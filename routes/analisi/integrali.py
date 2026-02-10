# ---------------------------------------------------------------------------
#  ROUTE: /api/integrale  – Calcolo integrale indefinito con passaggi
# ---------------------------------------------------------------------------

from flask import Blueprint, request, jsonify
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)
from sympy.printing.latex import latex

integrali_bp = Blueprint("integrali", __name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Normalizza \frac{a}{b}  →  (a)/(b)   per semplificare il parsing SymPy
# ---------------------------------------------------------------------------
def normalize_frac_syntax(expr: str) -> str:
    import re
    # Sostituzioni iterate finché troviamo ancora \frac
    frac_pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    while re.search(frac_pattern, expr):
        expr = re.sub(frac_pattern, r"(\1)/(\2)", expr)
    return expr


def _latex_to_sympy_string(expr: str) -> str:
    """
    Converte una stringa LaTeX minimale in una stringa interpretabile da SymPy.
    Riutilizza alcune trasformazioni chiave già viste nel parser di studio_funzione.
    """
    expr = normalize_frac_syntax(expr)
    expr = expr.replace("^", "**")

    # Mappa funzioni standard
    _func_map = {
        r"\sin": "sin",
        r"\cos": "cos",
        r"\tan": "tan",
        r"\sec": "sec",
        r"\csc": "csc",
        r"\cot": "cot",
        r"\arcsin": "asin",
        r"\arccos": "acos",
        r"\arctan": "atan",
        r"\ln": "log",
        r"\log": "log",
        r"\sqrt": "sqrt",
        r"\exp": "exp",
        r"\abs": "Abs",
    }
    for latex_name, sympy_name in _func_map.items():
        expr = expr.replace(latex_name, sympy_name)

    # Sostituisci \cdot con *
    expr = expr.replace(r"\cdot", "*")

    # \sqrt{…}  →  sqrt(…)
    import re

    expr = re.sub(r"sqrt\{([^{}]+)\}", r"sqrt(\1)", expr)
    expr = re.sub(r"sqrt([0-9]+)", r"sqrt(\1)", expr)

    return expr


def _parse_sympy_expression(expr: str):
    """
    Parsing sicuro con dizionario di funzioni consentite.
    """
    allowed_funcs = {
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "sec": sp.sec,
        "csc": sp.csc,
        "cot": sp.cot,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "exp": sp.exp,
        "Abs": sp.Abs,
        "E": sp.E,
    }
    transformations = standard_transformations + (implicit_multiplication_application,)
    return parse_expr(expr, local_dict=allowed_funcs, transformations=transformations)


# =========================================================================== #
# --------------------- MOTORE DI INTEGRAZIONE CON PASSAGGI ------------------ #
# =========================================================================== #
#  • Ogni strategia aggiunge step LaTeX al log.
#  • La funzione principale `integrate_with_steps(expr, x)` restituisce
#    (primitiva, steps) e viene richiamata dal route.
# --------------------------------------------------------------------------- #

# --- utilità per registrare uno step --------------------------------------- #
def _add_step(steps, title, tex):
    steps.append({"title": title, "content": tex})

# --- formule dirette (immediate) e generalizzate -------------------------- #
#  Le Wild() consentono di catturare costanti o funzioni generiche dove serve
A, Cst = sp.symbols('A Cst')          # costante “a” o generico
f = sp.Function('f')

_known_rules = [
    # potenze x^n  (n ≠ -1)
    *[
        (sp.Symbol('x')**n,
         sp.Symbol('x')**(n + 1)/(n + 1),
         []) for n in range(-5, 6) if n != -1
    ],

    # 1/x
    (1/sp.Symbol('x'), sp.log(sp.Abs(sp.Symbol('x'))), []),

    # radice √x
    (sp.sqrt(sp.Symbol('x')), sp.Rational(2, 3)*sp.sqrt(sp.Symbol('x'))**3, []),

    # a^x  con a costante
    (A**sp.Symbol('x'), A**sp.Symbol('x')/sp.log(A), [sp.Ne(A, 0), sp.Ne(A, 1)]),

    # e^x
    (sp.E**sp.Symbol('x'), sp.E**sp.Symbol('x'), []),

    # sin x  /  cos x
    (sp.sin(sp.Symbol('x')), -sp.cos(sp.Symbol('x')), []),
    (sp.cos(sp.Symbol('x')),  sp.sin(sp.Symbol('x')), []),

    # 1/(1+x²)   → arctan
    (1/(1 + sp.Symbol('x')**2), sp.atan(sp.Symbol('x')), []),

    # 1/cos²x = sec²x  → tan
    (1/sp.cos(sp.Symbol('x'))**2, sp.tan(sp.Symbol('x')), []),

    # 1/sin²x = csc²x  → -cot
    (1/sp.sin(sp.Symbol('x'))**2, -sp.cot(sp.Symbol('x')), []),

    # 1/√(1 - x²)  → arcsin
    (1/sp.sqrt(1 - sp.Symbol('x')**2), sp.asin(sp.Symbol('x')), []),

    # -1/√(1 - x²)  → arccos
    (-1/sp.sqrt(1 - sp.Symbol('x')**2), sp.acos(sp.Symbol('x')), []),

    # ------ GENERALIZZAZIONI (f(x))^a f'(x) = (f)^{a+1}/(a+1) -------------
    ( (f(sp.Symbol('x')))**A * sp.diff(f(sp.Symbol('x')), sp.Symbol('x')),
      (f(sp.Symbol('x')))**(A + 1)/(A + 1),
      [sp.Ne(A, -1)] ),

    # f'(x)/f(x)   → ln|f(x)|
    ( sp.diff(f(sp.Symbol('x')), sp.Symbol('x')) / f(sp.Symbol('x')),
      sp.log( sp.Abs( f(sp.Symbol('x')) ) ),
      []),

    # (1/a) * f'(x)/f(x) → (1/a) * log|f(x)|
    ( (1/A) * sp.diff(f(sp.Symbol('x')), sp.Symbol('x')) / f(sp.Symbol('x')),
      (1/A) * sp.log(sp.Abs(f(sp.Symbol('x')))),
      [sp.Ne(A, 0)] ),

    # f'(x)/(a + f(x)^2) → (1/a) * arctan(f(x)/sqrt(a))
    ( sp.diff(f(sp.Symbol('x')), sp.Symbol('x')) / (A + f(sp.Symbol('x'))**2),
      (1/A) * sp.atan(f(sp.Symbol('x')) / sp.sqrt(A)) if A != 1 else sp.atan(f(sp.Symbol('x'))),
      [sp.Ne(A, 0)] ),

    # f'(x)/(a - f(x)^2) → (1/(2√a)) * log| (√a + f(x)) / (√a - f(x)) |
    ( sp.diff(f(sp.Symbol('x')), sp.Symbol('x')) / (A - f(sp.Symbol('x'))**2),
      (1/(2*sp.sqrt(A))) * sp.log(sp.Abs((sp.sqrt(A) + f(sp.Symbol('x'))) / (sp.sqrt(A) - f(sp.Symbol('x'))))),
      [sp.Ne(A, 0)] ),

    # f'(x)/(1+f(x)²) → arctan f(x)
    ( sp.diff(f(sp.Symbol('x')), sp.Symbol('x')) / (1 + f(sp.Symbol('x'))**2),
      sp.atan( f(sp.Symbol('x')) ),
      []),

    # f'(x)/cos²(f(x)) → tan f(x)
    ( sp.diff(f(sp.Symbol('x')), sp.Symbol('x')) / sp.cos( f(sp.Symbol('x')) )**2,
      sp.tan( f(sp.Symbol('x')) ),
      []),

    # f'(x)/sin²(f(x)) → -cot f(x)
    ( sp.diff(f(sp.Symbol('x')), sp.Symbol('x')) / sp.sin( f(sp.Symbol('x')) )**2,
      -sp.cot( f(sp.Symbol('x')) ),
      []),

    # f'(x)/√(1 - f(x)²) → arcsin f(x)
    ( sp.diff(f(sp.Symbol('x')), sp.Symbol('x')) / sp.sqrt(1 - f(sp.Symbol('x'))**2),
      sp.asin( f(sp.Symbol('x')) ),
      []),

    # -f'(x)/√(1 - f(x)²) → arccos f(x)
    ( -sp.diff(f(sp.Symbol('x')), sp.Symbol('x')) / sp.sqrt(1 - f(sp.Symbol('x'))**2),
      sp.acos( f(sp.Symbol('x')) ),
      []),
]

def _match_known_integral(expr, x):
    for pattern, prim, conds in _known_rules:
        m = expr.match(pattern)
        if m is not None and all(c.subs(m) for c in conds):
            return prim.subs(m)
    return None

# --- riconoscimento √(a^2 - x^2) ------------------------------------------ #
def _match_sqrt_a2_minus_x2(expr, x):
    if expr.is_Pow and expr.exp == sp.Rational(1, 2):
        A = sp.Wild('A', exclude=[x])
        m = expr.base.match(A**2 - x**2)
        if m:
            return sp.simplify(m[A])
    return None

# --- suggeritore di strategia --------------------------------------------- #
def _suggest_strategy(expr, x):
    if expr.is_Add:
        return "linearità"
    if _match_known_integral(expr, x):
        return "formula"
    if expr.is_Mul:
        num, den = expr.as_numer_denom()
        if den.has(x) and sp.diff(den, x).equals(num):
            return "sostituzione"
    if expr.is_rational_function(x):
        return "razionale"
    if _match_sqrt_a2_minus_x2(expr, x) is not None:
        return "trig_sin"
    return "default"

# --- strategie ------------------------------------------------------------- #
def _integrate_formula(expr, x, steps):
    prim = _match_known_integral(expr, x)
    _add_step(steps, "Formula diretta:", fr"\int {latex(expr)}\,dx = {latex(prim)} + C")
    return prim

def _integrate_linearity(expr, x, steps):
    """Applica la linearità: spezza la somma ed integra ogni termine."""
    terms = expr.as_ordered_terms()
    # Step di scomposizione esplicita
    split_latex = (
        r"\int " + latex(expr) + r"\,dx \;=\; "
        + " + ".join([r"\int " + latex(t) + r"\,dx" for t in terms])
    )
    _add_step(steps, "Linearità – scomposizione della somma:", split_latex)
    # Mostra la nuova forma da integrare
    _add_step(
        steps,
        "Nuovo integrale da calcolare:",
        " + ".join([r"\int " + latex(t) + r"\,dx" for t in terms])
    )

    total = 0
    # Integra ricorsivamente ciascun termine
    for t in terms:
        prim_t, sub_steps = integrate_with_steps(t, x)
        steps.extend(sub_steps)   # mostra tutti i passaggi del termine
        total += prim_t
    return total

def _integrate_sost(expr, x, steps):
    num, den = expr.as_numer_denom()
    u = sp.Symbol('u')
    du_dx = sp.diff(den, x)
    _add_step(steps, "Sostituzione:", fr"Poniamo \(u = {latex(den)}\) ⇒ \(du = {latex(du_dx)}\,dx\)")
    integrand = num/den * 1/du_dx
    prim_u = sp.integrate(integrand, x).subs(den, u)
    prim = prim_u.subs(u, den)
    _add_step(steps, "Risultato sostituzione:", latex(prim))
    return prim

def _integrate_rational(expr, x, steps):
    """
    Integrazione di funzione razionale:

    • Semplifica/cancella fattori comuni (sp.cancel)
    • Se gradi uguali o num > den → divisione polinomiale,
      poi integra ricorsivamente quoziente e resto/den.
    • Altrimenti (grado num < grado den) → prova decomposizione
      in fratti semplici; se la decomposizione non cambia l’espressione,
      delega direttamente alla ricorsione sull’espressione stessa.
    """
    # 0) Semplificazioni
    simplified = sp.cancel(expr)
    if simplified != expr:
        _add_step(steps, "Semplificazione tramite fattorizzazione:", latex(simplified))
    expr = simplified

    num, den = sp.fraction(sp.together(expr))
    deg_num, deg_den = sp.degree(num, x), sp.degree(den, x)

    # Caso A: divisione polinomiale necessaria
    if deg_num >= deg_den:
        q, r = sp.div(num, den, domain='QQ')
        division_latex = (
            fr"{latex(expr)} = {latex(q)} + "
            fr"\dfrac{{{latex(r)}}}{{{latex(den)}}}"
        )
        _add_step(steps, "Divisione polinomiale:", division_latex)
        # Mostra il nuovo integrale di quoziente + parte propria
        _add_step(
            steps,
            "Nuovo integrale da calcolare:",
            fr"\int {latex(q)}\,dx \;+\; \int {latex(proper)}\,dx"
        )

        # Integra quoziente e parte propria R/den con passaggi completi
        prim_q, st_q = integrate_with_steps(q, x)
        steps.extend(st_q)

        proper = sp.simplify(r / den)
        prim_prop, st_prop = integrate_with_steps(proper, x)
        steps.extend(st_prop)

        return prim_q + prim_prop

    # Caso B: già propria – prova fratti semplici
    apart_expr = sp.apart(expr, x, full=False)

    # Se la decomposizione cambia l'espressione, mostrala
    if apart_expr != expr:
        _add_step(steps, "Decomposizione in fratti semplici:", latex(apart_expr))
        _add_step(
            steps,
            "Nuovo integrale da calcolare:",
            r" + ".join([r"\int " + latex(term) + r"\,dx" for term in apart_expr.as_ordered_terms()])
        )
        prim, st_apart = integrate_with_steps(apart_expr, x)
        steps.extend(st_apart)
        return prim

    # Se nessuna decomposizione, prova SymPy di default
    prim = sp.integrate(expr, x)
    _add_step(steps, "Metodo di fallback (SymPy):", latex(prim))
    return prim

def _integrate_trig_sin(expr, x, steps):
    a = _match_sqrt_a2_minus_x2(expr, x)
    θ = sp.symbols('θ')
    subs = {x: a*sp.sin(θ)}
    dx = sp.diff(a*sp.sin(θ), θ)
    _add_step(steps, "Sostituzione trigonometrica:", fr"\(x = {latex(a)}\sin\theta\), \(dx = {latex(dx)}\)")
    new_int = sp.simplify(expr.subs(subs) * dx)
    prim_θ = sp.integrate(new_int, θ)
    _add_step(steps, "Integrazione in θ:", latex(prim_θ))
    prim = prim_θ.subs(θ, sp.asin(x/a))
    _add_step(steps, "Risostituzione:", latex(prim))
    return prim

def _integrate_default(expr, x, steps):
    prim = sp.integrate(expr, x)
    _add_step(steps, "Metodo di default (SymPy):", latex(prim))
    return prim

# --- orchestratore --------------------------------------------------------- #
def integrate_with_steps(expr, x):
    steps = []
    strat = _suggest_strategy(expr, x)
    if strat == "formula":
        prim = _integrate_formula(expr, x, steps)
    elif strat == "linearità":
        prim = _integrate_linearity(expr, x, steps)
    elif strat == "sostituzione":
        prim = _integrate_sost(expr, x, steps)
    elif strat == "razionale":
        prim = _integrate_rational(expr, x, steps)
    elif strat == "trig_sin":
        prim = _integrate_trig_sin(expr, x, steps)
    else:
        prim = _integrate_default(expr, x, steps)
    return prim, steps

# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@integrali_bp.route("/api/integrale", methods=["POST"])
def calcola_integrale():
    data = request.get_json(silent=True) or {}
    integrand_latex = (data.get("integrand") or "").strip()

    if not integrand_latex:
        return jsonify({"success": False, "error": "Integrando mancante."}), 400

    # --------------------------- Parsing & integrazione --------------------
    try:
        sympy_string = _latex_to_sympy_string(integrand_latex)
        f_expr = _parse_sympy_expression(sympy_string)
    except Exception as e:
        return jsonify({"success": False, "error": f"Parsing error: {str(e)}"}), 400

    x = sp.symbols("x")

    # --------------------------- Step LaTeX --------------------------------
    steps = []

    # 1) Integrale inserito
    integral_latex = r"\displaystyle \int " + latex(f_expr) + r"\, dx"
    steps.append({"title": "Integrale inserito:", "content": integral_latex})

    # ---------------------------------------------------------
    # Integra con passaggi dettagliati
    # ---------------------------------------------------------
    try:
        primitive, extra_steps = integrate_with_steps(f_expr, x)
    except Exception as e:
        return jsonify({"success": False, "error": f'Errore integrazione: {str(e)}'}), 400
    steps.extend(extra_steps)

    # Step conclusivo: risultato finale
    steps.append({
        "title": "Risultato finale:",
        "content": r"\displaystyle \int " + latex(f_expr) + r"\, dx \;=\; " + latex(primitive) + r" + C"
    })

    # Risposta finale
    return jsonify(
        {
            "success": True,
            "result": latex(primitive),
            "steps": steps,
        }
    )