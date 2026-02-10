"""
Blueprint per la risoluzione di sistemi lineari quadrati (2 – 4 equazioni)
mediante eliminazione di Gauss, con output “step‑by‑step” in LaTeX.

JSON in ingresso:
{
  "n": 3,                           # dimensione (2‑4)
  "equations": [[...], ...]        # lista di liste (n righe, n+1 colonne)
}

JSON in uscita:
{
  "success": true,
  "steps": [ {title:str, content:str}, ... ],
  "result": {
      "compatible": bool,
      "unique": bool,
      "rankA": int,
      "rankAug": int,
      "solution": str | null          # LaTeX della soluzione se unica / parametrica
  }
}
"""
from __future__ import annotations

from flask import Blueprint, request, jsonify
from fractions import Fraction
from typing import List
import re
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)
from tokenize import TokenError

_transformations = standard_transformations + (implicit_multiplication_application,)

syslin_bp = Blueprint("sistemi_lineari_bp", __name__)

# ---------- utility ------------------------------------------------------ #
def _to_tex_num(x):
    if isinstance(x, Fraction):
        if x.denominator == 1:
            return str(x.numerator)
        else:
            return fr"\frac{{{x.numerator}}}{{{x.denominator}}}"
    if isinstance(x, sp.Basic):
        return sp.latex(sp.nsimplify(x))
    try:
        if int(x) == x:
            return str(int(x))
    except Exception:
        pass
    return str(x)

def latex_augmented(A: sp.Matrix, b: sp.Matrix, p: int) -> str:
    """LaTeX per matrice aumentata [A | b] con parentesi quadre."""
    n = A.rows
    cols_tex = []
    for i in range(n):
        row_vals = [_to_tex_num(A[i, j]) for j in range(p)]
        row_vals.append(_to_tex_num(b[i, 0]))
        cols_tex.append(" & ".join(row_vals))
    body = r" \\ ".join(cols_tex)
    return r"\left[\,\begin{array}{%s|c}%s\end{array}\,\right]" % ("c"*p, body)  # n col + | + 1

def latex_cases_system(eqs: List[List[str]]) -> str:
    """
    Restituisce il sistema in forma
      { aligned
        lhs_1 &= rhs_1 \\
        lhs_2 &= rhs_2 \\ …
    con allineamento sull’uguale.
    """
    rows_tex = [f"{lhs} &= {rhs}" for lhs, rhs in eqs]
    body = r" \\ ".join(rows_tex)
    return r"\left\{\,\begin{aligned}" + body + r"\end{aligned}\right."

def _pivot_zeros(A: sp.Matrix, k_sym):
    """
    Restituisce l’insieme dei valori di k che annullano i pivot
    della matrice A (già in forma a gradini).
    """
    # --- pivot e denominatori calcolati direttamente su 'A' --------------
    crit = set()

    # 1) individua i pivot della forma a gradini: primo elemento non‑nullo di ogni riga
    for i in range(A.rows):
        row = A.row(i)
        pivot_expr = None
        for elem in row:
            if elem != 0:
                pivot_expr = sp.simplify(elem)
                break
        if pivot_expr is not None and k_sym in pivot_expr.free_symbols:
            crit.update(sp.solve(sp.Eq(pivot_expr, 0), k_sym))

    # 2) analizza tutti gli entry per denominatori contenenti k (anche non‑pivot)
    for entry in A:
        num, den = sp.fraction(entry)
        if den.free_symbols and k_sym in den.free_symbols:
            crit.update(sp.solve(sp.Eq(den, 0), k_sym))

    return {sp.nsimplify(v) for v in crit}

# ---------- helper: fattorizza in k se possibile (solo su ℝ) -------------
def _factor_in_k(expr, k_sym):
    """
    Se l’espressione contiene k, prova a fattorizzarla su ℚ (⊂ ℝ).
    Restituisce la forma fattorizzata solo se non banale,
    altrimenti l’espressione originale.
    """
    if k_sym not in expr.free_symbols:
        return expr
    fact = sp.factor(expr)
    return fact if fact != expr else expr

# ---------- route -------------------------------------------------------- #
@syslin_bp.route("/api/sistemi-lineari", methods=["POST"])
def sistemi_lineari_route():
    data = request.get_json(silent=True) or {}
    n = data.get("n")
    equations = data.get("equations")

    # --- basic validation
    if not isinstance(n, int) or n not in (2, 3, 4):
        return jsonify(success=False, error="Parametro 'n' mancante o non valido (2–4)."), 400
    if (not isinstance(equations, list) or len(equations) != n or
        any(not isinstance(row, list) or len(row) not in (2, n + 1) for row in equations)):
        return jsonify(
            success=False,
            error="Ogni riga deve contenere due stringhe (lhs, rhs) o n+1 coefficienti numerici."
        ), 400

    # elenco variabili ammesse in ordine
    _allowed_var_names = ['x', 'y', 'z', 'w']

    # raccolta simboli effettivamente presenti nelle LHS (ignoriamo k)
    vars_found_ordered = []
    for lhs_rhs in equations:
        # formato simbolico -> [lhs, rhs]; formato numerico -> [coeffs..., b]
        if len(lhs_rhs) != 2:
            # numerico: nessuna variabile da scansionare
            continue
        lhs_str = lhs_rhs[0]
        if not isinstance(lhs_str, str) or not lhs_str.strip():
            continue
        lhs_expr = _latex_frac_to_python(lhs_str)
        try:
            expr = parse_expr(lhs_expr,
                              transformations=_transformations,
                              local_dict={nm: sp.symbols(nm) for nm in _allowed_var_names})
        except (SyntaxError, TokenError):
            # se non parse-abile, passa oltre
            continue
        for name in _allowed_var_names:
            sym = sp.symbols(name)
            if sym in expr.free_symbols and sym not in vars_found_ordered:
                vars_found_ordered.append(sym)

    # variabili effettive (p colonne). Se nessuna variabile trovata usiamo x di default.
    vars_syms = vars_found_ordered if vars_found_ordered else [sp.symbols('x')]
    p = len(vars_syms)
    if p > 4:
        return jsonify(success=False, error="Sono ammesse al massimo 4 variabili (x,y,z,w)."), 400

    # parse coefficients (int, frac, or symbol k)
    k = sp.symbols('k')
    A_list = []
    b_list = []

    symbolic_input = len(equations[0]) == 2  # left/right expressions

    if symbolic_input:
        # parse each lhs = rhs expression pair
        for lhs_str, rhs_str in equations:
            try:
                locals_map = {'k': k, **{str(v): v for v in vars_syms}}
                lhs_py = _latex_frac_to_python(lhs_str)
                rhs_py = _latex_frac_to_python(rhs_str)

                lhs = parse_expr(
                    lhs_py,
                    local_dict=locals_map,
                    global_dict={**sp.__dict__, **locals_map},
                    transformations=_transformations
                )
                rhs = parse_expr(
                    rhs_py,
                    local_dict=locals_map,
                    global_dict={**sp.__dict__, **locals_map},
                    transformations=_transformations
                )
                # Verifica che compaiano solo le variabili x,y,z,w e k
                allowed_syms = set(vars_syms) | {k}
                offending = (lhs.free_symbols | rhs.free_symbols) - allowed_syms
                if offending:
                    bad = ", ".join(str(s) for s in offending)
                    return jsonify(
                        success=False,
                        error=f"Sono ammesse solo, nella giusta dimensione, le variabili: x, y, z, w e k (trovate: {bad})."
                    ), 400
            except Exception:
                return jsonify(success=False, error="Impossibile interpretare le espressioni del sistema."), 400

            eq = sp.expand(lhs - rhs)               # porta tutto a sinistra
            # Verifica che il sistema sia lineare (nessuna variabile al denominatore o grado > 1)
            num, den = eq.as_numer_denom()
            # Controlla denominatore
            if den.free_symbols & set(list(vars_syms) + [k]):
                return jsonify(
                    success=False,
                    error="Il sistema non è lineare: variabili al denominatore."
                ), 400
            # Controlla grado del numeratore (supporta anche il parametro simbolico k)
            try:
                pol = sp.Poly(num, *vars_syms, domain='EX')  # 'EX' tollera coefficienti simbolici
                if pol.total_degree() > 1:
                    return jsonify(
                        success=False,
                        error="Il sistema non è lineare: grado > 1."
                    ), 400
            except sp.PolynomialError:
                return jsonify(
                    success=False,
                    error="Il sistema non è polinomiale in forma lineare."
                ), 400
            coeffs = [eq.coeff(v) for v in vars_syms]
            const_term = -eq.subs({v: 0 for v in vars_syms})
            A_list.append(coeffs)
            b_list.append(const_term)
    else:
        # coefficiente numerici già forniti
        for row in equations:
            if len(row) != p + 1:
                return jsonify(success=False,
                               error=f"In formato numerico servono {p}+1 valori per riga (trovati {len(row)})."), 400
            A_list.append([_parse_entry(tok, k) for tok in row[:-1]])
            b_list.append(_parse_entry(row[-1], k))

    A = sp.Matrix(A_list)
    b = sp.Matrix(b_list)

    steps = []

    # step 0: sistema originale
    steps.append({
        "title": "Sistema inserito:",
        "content": latex_cases_system(equations)
    })

    # step 1: matrice aumentata
    steps.append({
        "title": "Matrice aumentata:",
        "content": r"\left[A\,|\,\mathbf{b}\right] \;=\; " + latex_augmented(A, b, p)
    })

    # step 2: forma a gradini (Gauss)
    Aug = A.row_join(b)
    Aug_ech = Aug.echelon_form()
    A_ech = Aug_ech[:, :p]
    b_ech = Aug_ech[:, p:]
    # fattorizza eventuali polinomi in k (se non banali)
    A_fact = A_ech.applyfunc(lambda e: _factor_in_k(e, k))
    b_fact = b_ech.applyfunc(lambda e: _factor_in_k(e, k))

    steps.append({
        "title": "Forma a gradini (eliminazione di Gauss):",
        "content": r"\left[A\,|\,\mathbf{b}\right] \;=\; " + latex_augmented(A_fact, b_fact, p)
    })

    # --- Analisi del parametro k sulla matrice semplificata -------------
    has_k = k in A_ech.free_symbols or k in b_ech.free_symbols
    crit_vals = sorted(_pivot_zeros(Aug_ech, k)) if has_k else []

    if has_k:
        if crit_vals:
            crit_tex = ", ".join(sp.latex(v) for v in crit_vals)
            steps.append({
                "title": "Analisi pivot in funzione di $k$:",
                "content": fr"\text{{Valori critici: }} {crit_tex}"
            })
        else:
            steps.append({
                "title": "Analisi pivot in funzione di $k$:",
                "content": r"\text{Nessun pivot dipende da }k."
            })

    # --- Studio dei casi in funzione di k ---------------------------------
    cases = [None] + crit_vals  # None = caso generico
    for kval in cases:
        A_k = A if kval is None else A.subs({k: kval})
        b_k = b if kval is None else b.subs({k: kval})
        rankA = int(A_k.rank())
        rankAug = int(A_k.row_join(b_k).rank())
        compatible = (rankA == rankAug)
        unique = compatible and rankA == p

        if kval is None:
            if crit_vals:
                not_eq = ", ".join(sp.latex(v) for v in crit_vals)
                case_title = fr"Caso $k\neq{not_eq}$:"
            else:
                case_title = r"Caso generico:"
        else:
            case_title = fr"Caso $k={sp.latex(kval)}$:"

        # --- Studio per ciascun caso (sintesi + soluzioni) -----------------
        # etichetta del caso
        steps.append({
            "title": case_title,
            "content": ""
        })  # placeholder, verrà sostituito subito dopo

        # sintesi compatibilità e tipo di soluzione
        if compatible and unique:
            content_line = (
                fr"\operatorname{{rank}}(A) = \operatorname{{rank}}[A|b] = {p}"
                r" \;\Longrightarrow\; \text{soluzione unica}"
            )
        elif compatible:
            diff = p - rankA
            content_line = (
                fr"\operatorname{{rank}}(A) = \operatorname{{rank}}[A|b] = {rankA} < {p}"
                fr" \;\Longrightarrow\; \infty^{{{diff}}} \text{{ soluzioni}}"
            )
        else:
            content_line = (
                fr"\operatorname{{rank}}(A) = {rankA} < \operatorname{{rank}}[A|b] = {rankAug}"
                r" \;\Longrightarrow\; \text{sistema incompatibile}"
            )

        # aggiorna il passo precedente con il contenuto corretto
        steps[-1]["content"] = content_line

        # se non compatibile, salto direttamente al prossimo caso
        if not compatible:
            continue

        # se unica, aggiungo la soluzione unica
        if unique:
            sol_vec = A_k.solve(b_k)
            sol_tex = (
                r"\mathbf{x}=\begin{bmatrix}"
                + r"\\ ".join(sp.latex(e) for e in sol_vec)
                + r"\end{bmatrix}"
            )
            steps.append({
                "title": "Soluzione unica:",
                "content": sol_tex
            })
        else:
            # famiglia parametrica invariata
            sol_vec, _ = A_k.gauss_jordan_solve(b_k)
            free_syms = sorted(
                list(sol_vec.free_symbols - set(vars_syms) - {k}),
                key=lambda s: s.name
            )
            mapping, c_syms = {}, []
            if len(free_syms) == 1:
                new = sp.symbols('c')
                mapping[free_syms[0]] = new
                c_syms.append(new)
            else:
                for j, s0 in enumerate(free_syms, 1):
                    new = sp.symbols(f'c_{j}')
                    mapping[s0] = new
                    c_syms.append(new)
            sol_vec = sol_vec.subs(mapping)
            args = ", ".join(sp.latex(s) for s in c_syms)
            fam_tex = (
                r"\mathbf{x}(" + args + r")=\begin{bmatrix}"
                + r"\\ ".join(sp.latex(e) for e in sol_vec)
                + r"\end{bmatrix}"
                + rf"\;({args}\in\mathbb{{R}})"
            )
            steps.append({
                "title": "Famiglia di soluzioni:",
                "content": fam_tex
            })

    # --- risposta JSON ----------------------------------------------------
    return jsonify(success=True, steps=steps)

# ---------- helper: convert LaTeX \frac{a}{b} to (a)/(b) -------------
_frac_pattern = re.compile(r'\\frac\{([^{}]+)\}\{([^{}]+)\}')

def _latex_frac_to_python(expr: str) -> str:
    """
    Converte ricorsivamente tutti i \frac{num}{den} in  (num)/(den).
    Mantiene le parentesi per preservare la precedenza.
    """
    # normalizza \left...\right delimiters -> plain delimiters
    expr = expr.replace(r'\left(', '(').replace(r'\right)', ')')
    expr = expr.replace(r'\left[', '[').replace(r'\right]', ']')
    expr = expr.replace(r'\left\{', '{').replace(r'\right\}', '}')
    # converte caret (^) in esponenziazione Python (**)
    expr = expr.replace('^', '**')

    while '\\frac' in expr:
        expr, n = _frac_pattern.subn(r'(\1)/(\2)', expr)
        if n == 0:
            break
    return expr

# ---------- helper parse ------------------------------------------------- #
def _parse_entry(token: str, sym_k):
    """Parse a coefficient/RHS token: int, frac, float, or 'k'."""
    if isinstance(token, (int, float, Fraction)):
        return token
    if isinstance(token, str):
        token = token.strip()
        if token == 'k':
            return sym_k
        if token.startswith('\\frac'):
            # \frac{a}{b}
            m = re.match(r'^\\frac\{(-?\d+)\}\{(-?\d+)\}$', token.replace(' ', ''))
            if m:
                return Fraction(int(m.group(1)), int(m.group(2)))
        try:
            return Fraction(token)
        except Exception:
            pass
    raise ValueError(f"Token non riconosciuto: {token}")
