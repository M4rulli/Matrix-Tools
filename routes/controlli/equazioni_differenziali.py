def normalize_latex_like(expr: str) -> str:
    """
    Normalizza una sintassi LaTeX‑like controllata in sintassi SymPy‑style.
    Supporta funzioni elementari per E.D. lineari a coefficienti costanti.
    """
    if not isinstance(expr, str):
        return expr

    # Funzioni elementari (LaTeX → SymPy)
    replacements = {
        r"\sin": "sin",
        r"\cos": "cos",
        r"\tan": "tan",
        r"\exp": "exp",
        r"\ln": "log",
        r"\log": "log",
        r"\pi": "pi",
        r"\sqrt": "sqrt",
    }
    for k, v in replacements.items():
        expr = expr.replace(k, v)

    # Frazioni LaTeX: \frac{a}{b} -> (a)/(b)
    expr = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', expr)

    # Frazioni compatte: \frac12 -> 1/2
    expr = re.sub(r'\\frac\s*([0-9]+)\s*([0-9]+)', r'\1/\2', expr)

    # ------------------------------------------------------------
    #  Radici: \sqrt{a} -> sqrt(a),  \sqrt2 -> sqrt(2)
    # ------------------------------------------------------------
    expr = re.sub(r"\\\\sqrt\s*\{([^}]*)\}", r"sqrt(\1)", expr)
    expr = re.sub(r"\\\\sqrt\s*([0-9]+)", r"sqrt(\1)", expr)

    # Costanti e simboli
    expr = expr.replace("π", "pi")

    # Operatori
    expr = expr.replace(r"\cdot", "*")
    expr = expr.replace("^", "**")

    # Parentesi LaTeX
    expr = expr.replace(r"\left", "")
    expr = expr.replace(r"\right", "")
    expr = expr.replace("{", "(").replace("}", ")")

    # Moltiplicazioni implicite tra parentesi: )(  -> )*(
    expr = re.sub(r'\)\s*\(', r')*(', expr)

    # Inserisce parentesi automatiche: cos t → cos(t), exp -t → exp(-t), sqrt t → sqrt(t)
    expr = re.sub(r'\b(sin|cos|tan|exp|log|sqrt)\s+([a-zA-Z0-9\-\+]+)', r'\1(\2)', expr)

    # (d±k)(...)y -> (d±k)(...)*y
    expr = re.sub(r'\)\s*y\b', r')*y', expr)

    # Rimuove spazi inutili
    expr = expr.replace(" ", "")

    # Moltiplicazione implicita numero-lettera: 1/2t -> 1/2*t, 2t -> 2*t
    expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)

    # Moltiplicazione implicita parentesi-lettera: )(t) o )t -> )*t (generico)
    expr = re.sub(r'\)([a-zA-Z])', r')*\1', expr)

    print("[DEBUG][normalize_latex_like] ->", expr)
    return expr
from flask import Blueprint, request, jsonify
import sympy as sp
from sympy import (
    symbols, Function, Eq, dsolve, Derivative, exp, simplify, collect,
    cos, sin, latex, expand, linear_eq_to_matrix
)
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)
import re

equazioni_differenziali_bp = Blueprint("equazioni_differenziali", __name__)

def split_phase(expr, t):
    """
    Trasforma R*sin(ω t + φ) o R*cos(ω t + φ) in A*sin(ω t) + B*cos(ω t),
    usando le identità:
    - sin(ωt + φ) = sin(ωt)cos(φ) + cos(ωt)sin(φ)
    - cos(ωt + φ) = cos(ωt)cos(φ) - sin(ωt)sin(φ)
    """
    import sympy as sp
    from sympy import sin, cos, collect

    print("[DEBUG][split_phase] Input expression:", expr)

    try:
        expr = sp.expand_mul(expr)

        # Prova a fare pattern matching su R*sin(ω t + φ)
        a, b, R = sp.Wild('a'), sp.Wild('b'), sp.Wild('R')
        match = expr.match(R * sin(a * t + b))
        if match:
            R_val = match[R]
            omega = match[a]
            phi = match[b]
            A = sp.simplify(R_val * sp.cos(phi))
            B = sp.simplify(R_val * sp.sin(phi))
            result = A * sin(omega * t) + B * cos(omega * t)
            print("[DEBUG][split_phase] Matched sin(ωt + φ): A =", A, ", B =", B)
            return result

        # Prova a fare pattern matching su R*cos(ω t + φ)
        match = expr.match(R * cos(a * t + b))
        if match:
            R_val = match[R]
            omega = match[a]
            phi = match[b]
            A = sp.simplify(R_val * sp.cos(phi))
            B = sp.simplify(-R_val * sp.sin(phi))
            result = A * cos(omega * t) + B * sin(omega * t)
            print("[DEBUG][split_phase] Matched cos(ωt + φ): A =", A, ", B =", B)
            return result

        # fallback: restituisci l'espressione originale
        print("[DEBUG][split_phase] Nessun match diretto.")
        return expr

    except Exception as e:
        print("[DEBUG][split_phase] Exception:", str(e))
        return expr
    

def get_annichilatore(rhs_expr):
    """
    Restituisce l'annichilatore A(Δ) (oggetto SymPy Poly)
    per un termine u(t) qualsiasi, in base alla tabella continua.
    Se u(t)=u1+u2+..., torna il prodotto dei singoli fattori.
    """
    # Use only 'd' as the internal operator. Δ is used only for LaTeX output.
    t, d = sp.symbols("t d")
    a, b, n = sp.Wild('a'), sp.Wild('b'), sp.Wild('n')

    def ann_term(term):
        # Remove a numeric coefficient at the beginning (positive or negative)
        coeff, core = term.as_independent(t, as_Add=False)
        if coeff != 1:
            term = core
        if term.could_extract_minus_sign():
            term = -term
        # constant 1
        if term == 1:
            return d
        # linear term t   (k = 1  →  d^{1+1} = d² )
        if term == t:
            return d**2
        # polynomial t^k
        if term.is_Pow and term.base == t and term.exp.is_Integer and term.exp >= 0:
            k = int(term.exp)
            return d**(k + 1)
        # e^{at} * t^k
        m = term.match(t**n * sp.exp(a * t))
        if m and m[n].is_Integer and m[n] >= 0:
            k = int(m[n])
            A = (d - m[a])**(k + 1)
            return A
        # e^{at}
        m = term.match(sp.exp(a * t))
        if m:
            return d - m[a]
        # cos(bt) / sin(bt) * t^k
        for trig in (sp.cos, sp.sin):
            m = term.match(t**n * trig(b * t))
            if m and m[n].is_Integer and m[n] >= 0:
                k = int(m[n])
                return (d**2 + m[b]**2)**(k + 1)
            m = term.match(trig(b * t))
            if m:
                return d**2 + m[b]**2
        # e^{at} cos(bt) / sin(bt) * t^k
        for trig in (sp.cos, sp.sin):
            m = term.match(t**n * sp.exp(a*t) * trig(b*t))
            if m and m[n].is_Integer and m[n] >= 0:
                k = int(m[n])
                return ((d - m[a])**2 + m[b]**2)**(k + 1)
            m = term.match(sp.exp(a*t) * trig(b*t))
            if m:
                return (d - m[a])**2 + m[b]**2
        return None  # not recognized

    # decompose u(t) into sum of addends
    factors = []
    for addend in rhs_expr.as_ordered_terms():
        A_i = ann_term(addend)
        if A_i is None:
            return r"\text{Non determinato automaticamente}"
        factors.append(A_i)   # keep factored form

    # product of all factors (keep factored form for internal logic)
    A_total = sp.Mul(*factors, evaluate=False)

    # For LaTeX only: simplify obvious products (e.g. d*d**2 -> d**3),
    # but do NOT change the internal A_total used for computations.
    A_show = sp.simplify(A_total)

    latex_in_Delta = sp.latex(A_show).replace("d", r"\Delta")
    return A_total, latex_in_Delta


@equazioni_differenziali_bp.route('/api/equazioni_differenziali', methods=['POST'])
def api_equazione_differenziale():
    data = request.get_json()

    equation = data.get("equazione", "")
    condizioni = data.get("condizioniIniziali", [])

    print("[DEBUG] condizioniIniziali ricevute:", condizioni)

    if not equation.strip():
        return jsonify({
            "success": False,
            "error": "Nessuna equazione differenziale fornita."
        })

    result = solve_differential_equation(
        equation,
        condizioni_iniziali=condizioni
    )

    return jsonify(result)


def normalize_derivatives(equation_str):
    # Replace all Unicode Δ (and LaTeX \Delta) with ASCII d for internal logic
    equation_str = equation_str.replace(r"\Delta", "Δ")
    equation_str = equation_str.replace("Δ", "d")
    # Se la forma è semplicemente "d^n" o "d**n", converti in forma completa
    match = re.match(r'^\s*d\^(\d+)\s*$', equation_str)
    if match:
        n = match.group(1)
        equation_str = f"d^{n}y/dt^{n}"
    match_pow = re.match(r'^\s*d\*\*(\d+)\s*$', equation_str)
    if match_pow:
        n = match_pow.group(1)
        equation_str = f"d**{n}y/dt**{n}"

    # Prima gestione delle potenze nelle derivate (es. dy^2/dt^2 → d2y/dt2)
    equation_str = re.sub(r'd\^(\d+)y\s*/\s*dt\^(\d+)', r'd\1y/dt\2', equation_str)
    equation_str = re.sub(r'dy\^(\d+)/dt\^(\d+)', r'd\1y/dt\2', equation_str)

    # ------------------------------------------------------------------
    #  Gestione dell'operatore Δ (U+0394) inteso come derivata d/dt
    # ------------------------------------------------------------------
    # (No Δ logic remains; all Δ have become d)

    patterns = [
        (r'd10y\s*/\s*dt10', 'Derivative(y, (t, 10))'),
        (r'd9y\s*/\s*dt9', 'Derivative(y, (t, 9))'),
        (r'd8y\s*/\s*dt8', 'Derivative(y, (t, 8))'),
        (r'd7y\s*/\s*dt7', 'Derivative(y, (t, 7))'),
        (r'd6y\s*/\s*dt6', 'Derivative(y, (t, 6))'),
        (r'd5y\s*/\s*dt5', 'Derivative(y, (t, 5))'),
        (r'd4y\s*/\s*dt4', 'Derivative(y, (t, 4))'),
        (r'd3y\s*/\s*dt3', 'Derivative(y, (t, 3))'),
        (r'd2y\s*/\s*dt2', 'Derivative(y, (t, 2))'),
        (r'dy\s*/\s*dt', 'Derivative(y, t)'),
        (r'y', 'y')
    ]

    equation_str = equation_str.replace('^', '**')
    for pattern, replacement in patterns:
        equation_str = re.sub(pattern, replacement, equation_str)
    return equation_str

def solve_differential_equation(equation_str, condizioni_iniziali=None):
    try:
        latex_steps = []
        t = symbols('t')
        y = Function('y')(t)

        # --------------------------------------------------
        # Normalizzazione input condizioni iniziali
        # --------------------------------------------------
        if condizioni_iniziali is None:
            condizioni_iniziali = []

        # --------------------------------------------------
        # Normalize all \Delta (LaTeX) and Δ (Unicode) to d, then parse
        equation_str = normalize_derivatives(equation_str)
        print("[DEBUG] Equazione normalizzata:", equation_str)

        if not equation_str.strip() or equation_str.strip() in ["0 = 0", "0=0"]:
            raise ValueError("Nessun polinomio P(d) è stato inserito.")

        transformations = standard_transformations + (implicit_multiplication_application,)
        local_dict = {
            'y': y, 't': t,
            'd': sp.Symbol('d'),
            'Derivative': Derivative,
            'exp': exp, 'cos': cos, 'sin': sin,
            'sqrt': sp.sqrt,
            'e': exp(1),
            'pi': sp.pi,
            'log': sp.log,
        }

        lhs_str, rhs_str = equation_str.split('=', 1)

        # Normalizza ENTRAMBI i lati (LaTeX → SymPy)
        lhs_str = normalize_latex_like(lhs_str)
        rhs_str = normalize_latex_like(rhs_str)

        lhs_expr = parse_expr(lhs_str.strip(), transformations=transformations, local_dict=local_dict)
        rhs_expr = parse_expr(rhs_str.strip(), transformations=transformations, local_dict=local_dict)
        print("[DEBUG] LHS parsed:", lhs_expr)
        print("[DEBUG] RHS parsed:", rhs_expr)

        # --------------------------------------------------
        # Estrai P(d) dal lato sinistro PRIMA della conversione in Derivative
        # --------------------------------------------------
        d_sym = sp.Symbol("d")
        P_operator = lhs_expr

        # rimuove y(t)
        P_operator = P_operator.replace(
            lambda e: e == y,
            lambda e: 1
        )

        # rimuove eventuali Derivative(y, ...) residui (difensivo)
        P_operator = P_operator.replace(
            lambda e: isinstance(e, sp.Derivative) and e.expr == y,
            lambda e: d_sym ** e.derivative_count
        )

        P_operator = sp.expand(P_operator)

        # Esplicita l'azione di d come operatore differenziale: d^n * y -> Derivative(y, (t, n))
        def convert_d_operator(expr):
            if not expr.has(sp.Symbol('d')):
                return expr
            expr = sp.expand(expr)
            d = sp.Symbol('d')
            result = 0
            for term in expr.as_ordered_terms():
                if term.has(y):
                    coeffs = term.as_coeff_mul()
                    base_factors = coeffs[1]
                    power = 0
                    for f in base_factors:
                        if f == d:
                            power += 1
                        elif isinstance(f, sp.Pow) and f.base == d:
                            power += f.exp
                    if power > 0:
                        new_term = coeffs[0] * Derivative(y, (t, power))
                        result += new_term
                    else:
                        result += term
                else:
                    result += term
            return result

        lhs_expr = convert_d_operator(lhs_expr)
        rhs_expr = convert_d_operator(rhs_expr)
        print("[DEBUG] LHS dopo conversione d:", lhs_expr)
        print("[DEBUG] RHS dopo conversione d:", rhs_expr)

        # --------------------------------------------------
        # CONDIZIONI INIZIALI STANDARD
        # --------------------------------------------------
        ordine = 0
        for deriv in lhs_expr.atoms(sp.Derivative):
            if deriv.expr == y:
                ordine = max(ordine, deriv.derivative_count)

        # Se sono fornite condizioni iniziali, devono essere pari all’ordine
        if condizioni_iniziali:
            if len(condizioni_iniziali) != ordine:
                return {
                    "success": False,
                    "error": f"Servono {ordine} condizioni iniziali, ne sono state fornite {len(condizioni_iniziali)}."
                }

        # Espansione e semplificazione
        lhs_expr = expand(lhs_expr)
        rhs_expr = expand(rhs_expr)

        # Caso speciale: termine forzante zero → solo soluzione omogenea
        if rhs_expr == 0:
            # Risolvo solo l’omogenea
            homogeneous_sol = dsolve(Eq(lhs_expr, 0), y)
            # Estraggo direttamente la soluzione (non lista)
            homogeneous_rhs = collect(simplify(homogeneous_sol.rhs), t)
            homogeneous_latex = f"y_o(t) = {latex(homogeneous_rhs)}"

            # Aggiungo lo step per l’omogenea pura
            latex_steps.append({
                "title": "Soluzione dell'equazione omogenea:",
                "content": homogeneous_latex.replace("C_", "c_")
            })

            return {"success": True, "latex": latex_steps}

        original_eq = Eq(lhs_expr, rhs_expr)
        print("[DEBUG] Equazione completa:", original_eq)

        # Calcolo dell'annichilatore A(Δ)
        try:
            print("[DEBUG] Calcolo annichilatore...")
            u_func = rhs_expr
            A_expr, latex_in_Delta = get_annichilatore(u_func)
            # latex_in_Delta is already d→\Delta replaced
            latex_steps.insert(1, {
                "title": "Annichilatore \\( A(\\Delta) \\) tale che \\( A(\\Delta) u(t) = 0 \\):",
                "content": rf"A(\Delta) = {latex_in_Delta}"
            })
        except Exception as e:
            print("[DEBUG] Errore annichilatore:", str(e))
            latex_steps.insert(1, {
                "title": r"Annichilatore $A(\Delta)$:",
                "content": "Errore durante la determinazione"
            })
        d_sym = sp.Symbol("d")
        print("[DEBUG] A_expr (versione SymPy):", A_expr)

        # ------------------------------------------------------------------
        #  Estrai dal lato sinistro SOLO il polinomio P(Δ), rimuovendo
        #  l'eventuale "* y" o "* y(t)" aggiunti dal frontend.
        #  Esempi:
        #     "d + 1 * y"      -> "d + 1"
        #     "d**2*y"         -> "d**2"
        #     "Δ - 2  *  y(t)" -> "Δ - 2"
        # ------------------------------------------------------------------
        lhs_clean = lhs_str.strip()
        # remove any "* y" or "*y" or "* y(t)" at the end
        lhs_clean = re.sub(r'\*\s*y\s*(?:\(\s*t\s*\))?\s*$', '', lhs_clean)
        # also remove implicit multiplication "y" at the end
        lhs_clean = re.sub(r'\s*y\s*(?:\(\s*t\s*\))?\s*$', '', lhs_clean)

        P_poly_str = lhs_clean if lhs_clean else "0"   # fallback
        P_poly_sym = parse_expr(
            P_poly_str,
            transformations=transformations,
            local_dict={'d': d_sym}
        )

        # --------------------------------------------------
        # COSTRUISCI SOLO IL POLINOMIO OPERATORE (NO y, NO Derivative)
        # --------------------------------------------------
        full_operator = sp.expand(A_expr * P_operator)
        # sicurezza: rimuove eventuali Derivative(y, t, ...) rimasti per errore
        full_operator = full_operator.replace(
            lambda e: isinstance(e, sp.Derivative),
            lambda e: sp.Symbol("d") ** e.derivative_count
        )
        # Safety assertion: ensure no Δ remains internally
        assert not full_operator.has(sp.Symbol("Δ"))
        y_func = Function('y')(t)
        def apply_polynomial_operator_to_y(poly, y, d, t):
            # poly: polynomial in d only
            # returns the sum of terms as derivatives of y(t)
            poly = sp.expand(poly)
            result = 0
            for term in poly.as_ordered_terms():
                coeff, monom = term.as_coeff_Mul()
                if monom == 1:
                    result += coeff * y
                else:
                    # find power of d
                    if monom.has(d):
                        if isinstance(monom, sp.Pow) and monom.base == d:
                            power = monom.exp
                        elif monom == d:
                            power = 1
                        else:
                            if monom.is_Mul:
                                power = 0
                                for f in monom.args:
                                    if isinstance(f, sp.Pow) and f.base == d:
                                        power += f.exp
                                    elif f == d:
                                        power += 1
                                rest = monom / (d**power)
                                result += coeff * rest * Derivative(y, (t, power))
                                continue
                            else:
                                power = 0
                        result += coeff * Derivative(y, (t, power))
                    else:
                        result += coeff * monom * y
            return result

        applied_expr = apply_polynomial_operator_to_y(full_operator, y_func, d_sym, t)

        print("[DEBUG] Operatore A(d)*P(d) [solo operatore]:", full_operator)
        extended_lhs = applied_expr
        extended_eq  = Eq(extended_lhs, 0)

        # --------------------------------------------------
        # COSTRUZIONE VISIVA CORRETTA: (A(Δ)P(Δ)) y(t) = 0
        # --------------------------------------------------
        # Build (A(d)P(d)) as a pure polynomial in d, then substitute d→Δ for LaTeX
        full_poly = sp.expand(A_expr * P_operator)
        # Ensure no Derivative objects are present in polynomial
        assert not full_poly.has(sp.Derivative)
        full_poly_latex = sp.latex(full_poly).replace("d", r"\Delta")
        visual_latex = rf"\left({full_poly_latex}\right) y(t) = 0"

        try:
            ext_sol = dsolve(extended_eq, y)
            ext_rhs = expand(ext_sol.rhs)
            # Rimpiazza le costanti C_i con α_i per coerenza visiva
            alpha_symbols = [sp.Symbol(f"\\alpha_{{{i}}}") for i in range(1, 10)]
            for i, alpha in enumerate(alpha_symbols, start=1):
                ext_rhs = ext_rhs.subs(sp.Symbol(f'C{i}'), alpha)
            print("[DEBUG] Soluzione calcolata:", ext_rhs)
            ext_latex = rf"y_{{o,e}}(t) = {latex(ext_rhs)}"
        except Exception as e:
            ext_latex = "Non risolta automaticamente"

        latex_steps.insert(2, {
            "title": "Equazione omogenea estesa \\( A(\\Delta)P(\\Delta)y(t)=0 \\):",
            "content": rf"{visual_latex}"
        })
        latex_steps.insert(3, {
            "title": "Soluzione dell'omogenea estesa:",
            "content": ext_latex
        })

        # Tentativo di soluzione
        try:
            print("[DEBUG] Tentativo di risolvere l'equazione completa...")
            original_sol = dsolve(original_eq, y)
            if isinstance(original_sol, list):
                original_rhs_list = [collect(simplify(sol.rhs), t) for sol in original_sol]
                original_latex = r" \\ \text{Oppure} \\ ".join([f"y(t) = {latex(rhs)}" for rhs in original_rhs_list])
                original_rhs = original_rhs_list[0]  # assume primo come principale
            else:
                original_rhs = collect(simplify(original_sol.rhs), t)
                original_latex = f"y(t) = {latex(original_rhs)}"
        except Exception as e:
            return {
                'success': False,
                'error': f"Impossibile risolvere l'equazione: {str(e)}",
                'debug': f"Equazione normalizzata: {equation_str}"
            }

        # Soluzione omogenea (prova sempre, fallback su errore)
        try:
            print("[DEBUG] Tentativo di risolvere l'omogenea...")
            homogeneous_sol = dsolve(Eq(original_eq.lhs, 0), y)
            if isinstance(homogeneous_sol, list):
                homogeneous_rhs_list = [collect(simplify(sol.rhs), t) for sol in homogeneous_sol]
                homogeneous_latex = r" \\ \text{Oppure} \\ ".join([f"y_h(t) = {latex(rhs)}" for rhs in homogeneous_rhs_list])
                homogeneous_rhs = homogeneous_rhs_list[0]  # assume primo come principale
            else:
                homogeneous_rhs = collect(simplify(homogeneous_sol.rhs), t)
                homogeneous_latex = f"y_h(t) = {latex(homogeneous_rhs)}"
            homogeneous_latex = homogeneous_latex.replace("C_", "c_")
        except Exception as e:
            print("[DEBUG] Fallita soluzione dell'omogenea:", str(e))
            homogeneous_latex = "Non disponibile per equazioni non lineari"
            homogeneous_rhs = 0  # fallback per evitare errore

        # Calcolo della soluzione particolare automaticamente via dsolve
        try:
            particular_sol = dsolve(original_eq, y)
            # estrai l'RHS e sottrai la soluzione omogenea per ottenere la particolare
            if isinstance(particular_sol, list):
                particular_rhs = simplify(particular_sol[0].rhs - homogeneous_rhs)
            else:
                particular_rhs = simplify(particular_sol.rhs - homogeneous_rhs)
            # Se la soluzione particolare è in fase, la espande in forme sin+cos
            particular_latex = f"y_p(t) = {latex(particular_rhs)}"
            try:
                split = split_phase(particular_rhs, t)
                if split != particular_rhs:
                    split_latex = latex(split)
                    particular_latex = f"y_p(t) = {latex(particular_rhs)} = {split_latex}"
                else:
                    particular_latex = f"y_p(t) = {latex(particular_rhs)}"
            except Exception:
                pass
        except Exception:
            particular_latex = "Soluzione particolare non determinabile automaticamente"
        latex_steps.append({
            "title": "Soluzione particolare:",
            "content": particular_latex
        })

        latex_steps.insert(0, {
            "title": "Soluzione dell'equazione omogenea:",
            "content": homogeneous_latex.replace("y_h(t)", "y_{o}(t)")
        })

        # Calcola la soluzione generale y(t) = y_o(t) + y_p(t)
        try:
            split = split_phase(particular_rhs, t)
            rhs_to_use = split if split != particular_rhs else particular_rhs
        except Exception:
            rhs_to_use = particular_rhs

        y_general = expand(homogeneous_rhs + rhs_to_use)
        # Rinomina simboli C1...C9 in c_1...c_9 direttamente nell'espressione
        rename_dict = {sp.Symbol(f"C{i}"): sp.Symbol(f"c_{i}") for i in range(1, 10)}
        y_general = y_general.xreplace(rename_dict)
        # Genera il LaTeX con i simboli già corretti
        y_general_latex = f"y(t) = {latex(y_general)}"
        latex_steps.append({
            "title": "Soluzione generale esplicita:",
            "content": y_general_latex
        })

        # --------------------------------------------------
        # APPLICAZIONE CONDIZIONI INIZIALI (se presenti)
        # --------------------------------------------------
        if condizioni_iniziali:
            # --------------------------------------------------
            # Controllo di compatibilità numero condizioni iniziali
            # --------------------------------------------------
            if len(condizioni_iniziali) > ordine:
                return {
                    "success": False,
                    "error": (
                        f"Numero di condizioni iniziali incompatibile: "
                        f"ne sono state fornite {len(condizioni_iniziali)}, "
                        f"ma l'equazione è di ordine {ordine}."
                    )
                }
            # Ricava le costanti libere c_i
            constants = sorted(
                [s for s in y_general.free_symbols if s.name.startswith("c_")],
                key=lambda s: s.name
            )

            eqs = []
            for i in range(ordine):
                deriv_val = y_general.diff(t, i).subs(t, 0)
                rhs_val = sp.sympify(condizioni_iniziali[i])
                eqs.append(sp.Eq(deriv_val, rhs_val))

            sol_consts = sp.solve(eqs, constants, dict=True)

            if sol_consts:
                y_finale = y_general.subs(sol_consts[0])
                latex_steps.append({
                    "title": "Sistema delle condizioni iniziali:",
                    "content": r"\begin{cases}" + r"\\ ".join(latex(eq) for eq in eqs) + r"\end{cases}"
                })
                latex_steps.append({
                    "title": "Soluzione con condizioni iniziali:",
                    "content": f"y(t) = {latex(y_finale)}"
                })

                # --------------------------------------------------
                # Applica le soluzioni simboliche alla forma generale
                # --------------------------------------------------
                # --------------------------------------------------
                # SCOMPOSIZIONE FINALE CORRETTA (BASATA SUI DATI REALI)
                # risposta libera  = termini che dipendono dalle condizioni iniziali fornite
                # risposta forzata = tutti gli altri termini
                # --------------------------------------------------
                y_finale = sp.expand(y_finale)

                # simboli REALI delle condizioni iniziali (es. y_0, y_1, y_2)
                cond_symbols = [
                    sp.sympify(c) for c in condizioni_iniziali
                    if isinstance(c, str)
                ]

                terms = y_finale.as_ordered_terms()

                parte_libera = sum(
                    term for term in terms
                    if any(sym in term.free_symbols for sym in cond_symbols)
                )

                parte_forzata = sp.simplify(y_finale - parte_libera)

                # Due step separati (stile corso): y_f e y_l
                latex_steps.append({
                    "title": "Risposta forzata:",
                    "content": rf"y_{{f}}(t) = {sp.latex(parte_forzata)}"
                })
                latex_steps.append({
                    "title": "Risposta libera:",
                    "content": rf"y_{{l}}(t) = {sp.latex(parte_libera)}"
                })

                return {
                    "success": True,
                    "latex": latex_steps
                }

        return {
            "success": True,
            "latex": latex_steps
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'debug': f"Input: {equation_str}"
        }
