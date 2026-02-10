from flask import Blueprint, request, jsonify
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)
from sympy.functions.elementary.miscellaneous import real_root
def force_real_roots(expr):
    """
    Recursively replace rational powers with real_root on the real branch.
    """
    from sympy import Pow
    return expr.replace(
        lambda e: isinstance(e, Pow) and e.exp.is_Rational,
        lambda e: Pow(real_root(e.base, e.exp.q), e.exp.p, evaluate=False)
    )
from sympy.calculus.util import singularities, continuous_domain, periodicity
from sympy import Complement, S, factor_terms, limit, nsimplify, oo, sign, simplify, solve_univariate_inequality, symbols
from sympy import Interval
from sympy import Pow, Ne, And, Union, FiniteSet, solveset, S
from sympy.sets.sets import Intersection
from sympy import Piecewise
from sympy import Intersection, Union, And, S
from sympy.calculus.util import AccumulationBounds
from sympy import re, Abs, S, pi

# ----------------------------------------------------------------------
# Helper: normalizza \frac23 → \frac{2}{3}
import re
def normalize_frac_syntax(expr: str) -> str:
    # Match patterns like \frac23 and replace with \frac{2}{3}
    expr = re.sub(r'\\frac(\d)(\d)', r'\\frac{\1}{\2}', expr)
    return expr


def safe_domain(expr, x):
    """
    Calcola un dominio più rigoroso per espressioni con potenze f(x)**g(x).
    - Esclude gli zeri della base quando l'esponente può rendere la potenza ill‑definita.
    - Impone base ≥ 0 se l'esponente non è garantito intero.
    """
    from sympy.solvers.solveset import solveset
    try:
        base_domain = continuous_domain(expr, x, S.Reals)
    except NotImplementedError:
        # Il dominio non può essere calcolato in modo elementare (es. disuguaglianza trascendentale):
        # ripieghiamo su ℝ senza bloccare il flusso.
        base_domain = S.Reals
    except Exception:
        # Qualsiasi altra eccezione imprevista: continua con dominio reale completo
        base_domain = S.Reals


    forbidden_points = sp.S.EmptySet   # zeri della base da escludere
    extra_positive = None              # intervalli in cui base ≥ 0

    for p in expr.atoms(Pow):
        base, expo = p.args
        # 1. Trova i punti in cui la base è zero
        zeros = solveset(base, x, domain=S.Reals)
        if isinstance(zeros, sp.FiniteSet):
            # Se l’esponente è negativo o simbolico → escludi i punti dove la base è 0
            if (expo.is_number and expo.evalf() < 0) or (expo.has(x)):
                forbidden_points = forbidden_points.union(zeros)

        # 2. Se l’esponente è razionale con denominatore dispari → accettiamo basi anche negative
        if expo.is_Rational and expo.q % 2 == 1:
            continue  # dominio completo ℝ

        # 3. Se l’esponente non è garantito intero → imponi base ≥ 0
        if expo.is_integer is False or (expo.has(x) and expo.is_integer is not True):
            try:
                pos_set = solveset(base >= 0, x, domain=S.Reals)
                extra_positive = pos_set if extra_positive is None else extra_positive.intersect(pos_set)
            except Exception:
                pass

    # Applica le restrizioni
    if isinstance(forbidden_points, sp.FiniteSet) and len(forbidden_points) > 0:
        base_domain = Complement(base_domain, forbidden_points)

    if extra_positive is not None:
        base_domain = base_domain.intersect(extra_positive)

    return base_domain

studio_funzione_bp = Blueprint("studio_funzione_bp", __name__)


# ----------------------------------------------------------------------
# Helper: trasforma \frac{a}{b} in (a)/(b)
# ----------------------------------------------------------------------
def _latex_frac_to_python(s: str) -> str:
    import re

    # Corregge casi errati come \frac(1)(3) → (1)/(3)
    s = re.sub(r'\\frac\s*\(\s*([^)]+?)\s*\)\s*\(\s*([^)]+?)\s*\)', r'(\1)/(\2)', s)

    # existing brace-based frac conversion
    pattern = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    while True:
        m = pattern.search(s)
        if not m:
            break
        a, b = m.group(1), m.group(2)
        s = s[: m.start()] + f"({a})/({b})" + s[m.end() :]

    # parentheses-based frac conversion: handle nested parentheses
    def convert_frac_paren(text: str) -> str:
        result = ""
        i = 0
        while True:
            idx = text.find(r"\frac", i)
            if idx == -1:
                result += text[i:]
                break
            result += text[i:idx]
            j = idx + len(r"\frac")
            # skip whitespace
            while j < len(text) and text[j].isspace():
                j += 1
            # parse numerator if '(' follows
            if j < len(text) and text[j] == "(":
                # find matching ')'
                count = 0
                for k in range(j, len(text)):
                    if text[k] == "(":
                        count += 1
                    elif text[k] == ")":
                        count -= 1
                        if count == 0:
                            num = text[j+1:k]
                            end_num = k
                            break
                # skip whitespace to denominator
                m = end_num + 1
                while m < len(text) and text[m].isspace():
                    m += 1
                if m < len(text) and text[m] == "(":
                    # parse denominator
                    count = 0
                    for n in range(m, len(text)):
                        if text[n] == "(":
                            count += 1
                        elif text[n] == ")":
                            count -= 1
                            if count == 0:
                                den = text[m+1:n]
                                end_den = n
                                break
                    # append converted frac
                    result += f"({num})/({den})"
                    i = end_den + 1
                    continue
            # fallback: just copy '\frac' and continue
            result += r"\frac"
            i = idx + len(r"\frac")
        return result

    s = convert_frac_paren(s)
    return s

def evaluate_inf_sign(expr):
    """
    Semplifica espressioni del tipo  ±oo * something
    valutando correttamente il segno reale di *something*.
    Esempi:
        -oo * (-1)**(1/3)  →  +oo
        -oo * (-8)**(2/3)  →  +oo
        +oo * (-3)**(4/5)  →  -oo
    Restituisce expr invariato se non riconosce il pattern.
    """

    # helper: segno reale di un valore simbolico/ numerico
    def _real_sign(val):
        # Caso Pow con base reale negativa e esponente razionale p/q (q dispari):
        # (-a)**(p/q) ha segno (-1)**p
        if isinstance(val, sp.Pow):
            base, exp = val.args
            if base.is_real and base < 0 and exp.is_Rational and exp.q % 2 == 1:
                return -1 if (exp.p % 2) else +1

        # Fallback: usa la parte reale numerica di val
        try:
            real_part = sp.re(val.evalf())
            if real_part > 0:
                return +1
            elif real_part < 0:
                return -1
            else:
                return 0
        except Exception:
            # Estremo fallback (simbolico)
            return sp.sign(val)

    # helper: elabora un singolo termine tipo ±oo * something
    def _eval_mul_with_inf(term):
        if not (isinstance(term, sp.Mul) and any(a in term.args for a in (sp.oo, -sp.oo))):
            return term  # non è del tipo che ci interessa

        oo_part = sp.oo if sp.oo in term.args else -sp.oo
        others  = [a for a in term.args if a not in (sp.oo, -sp.oo)]
        if not others:   # solo ±oo → ritorna tale e quale
            return oo_part

        rest = sp.Mul(*others)
        sgn  = _real_sign(rest)
        if sgn == 0:
            return sp.nan   # evenètualmente non definito
        return oo_part if sgn > 0 else -oo_part

    # Applica la trasformazione ricorsivamente
    return expr.replace(
        lambda e: isinstance(e, sp.Mul) and any(a in e.args for a in (sp.oo, -sp.oo)),
        _eval_mul_with_inf
    )

def sign_chart(expr, var, domain):
    """
    Return the set where expr >= 0 over the given domain using a sign-chart algorithm.
    """
    expr = force_real_roots(expr)
    from sympy import solveset, S, Interval, Union, FiniteSet, solve_univariate_inequality
    # Normalize domain to concrete Interval(s) to avoid ConditionSet in later intersects
    from sympy import ConditionSet, Intersection, Union as SymUnion
    # Helper to convert a ConditionSet to Interval(s)
    def _normalize_cs(cs):
        base = cs.base_set if hasattr(cs, 'base_set') else S.Reals
        try:
            sol_int = solve_univariate_inequality(cs.condition, var)
            return sol_int.intersect(base)
        except Exception:
            return base
    # Flatten domain
    parts = []
    if isinstance(domain, ConditionSet):
        parts.append(_normalize_cs(domain))
    elif isinstance(domain, Intersection) or isinstance(domain, SymUnion):
        for part in domain.args:
            if isinstance(part, ConditionSet):
                parts.append(_normalize_cs(part))
            else:
                parts.append(part)
    else:
        parts.append(domain)
    # Rebuild domain as Union of normalized parts
    domain = SymUnion(*parts) if len(parts) > 1 else parts[0]

    num, den = expr.as_numer_denom()
    zeros_num = solveset(num, var, domain=S.Reals)
    print("[sign_chart] zeros_num =", zeros_num)
    zeros_den = solveset(den, var, domain=S.Reals)
    print("[sign_chart] zeros_den =", zeros_den)
    # Only iterate finite zeros; skip ConditionSet or other non-FiniteSet
    from sympy import FiniteSet
    if isinstance(zeros_num, FiniteSet):
        zeros_num_list = list(zeros_num)
    else:
        zeros_num_list = []
    if isinstance(zeros_den, FiniteSet):
        zeros_den_list = list(zeros_den)
    else:
        zeros_den_list = []

    # Breakpoints are the finite zeros of numerator and denominator
    breakpoints = sorted(zeros_num_list + zeros_den_list, key=lambda p: float(p))
    points = [S.NegativeInfinity] + breakpoints + [S.Infinity]
    intervals = []
    for a, b in zip(points[:-1], points[1:]):
        I = Interval.open(a, b).intersect(domain)
        if I == S.EmptySet:
            continue
        # flatten I into individual intervals
        sub_intervals = I.args if isinstance(I, Union) else [I]
        for J in sub_intervals:
            # choose a test point inside J
            start, end = J.start, J.end
            if start.is_finite and end.is_finite:
                test = (start + end) / 2
            elif start.is_finite and end is S.Infinity:
                test = start + 1
            elif start is S.NegativeInfinity and end.is_finite:
                test = end - 1
            else:
                test = 0 if domain.contains(0) is S.true else 1
            val = expr.subs(var, test).evalf().as_real_imag()[0]
            if val >= 0:
                intervals.append(J)
    # include zeros of numerator (if not zeros of denominator)
    for p in zeros_num_list:
        if p not in zeros_den_list and domain.contains(p) is S.true:
            intervals.append(FiniteSet(p))
    return Union(*intervals) if intervals else S.EmptySet

# ----------------------------------------------------------------------
# API endpoint per lo studio della funzione
# ----------------------------------------------------------------------
@studio_funzione_bp.route("/api/studio-funzione", methods=["POST"])
def studio_funzione():
    data = request.get_json()
    if not data or "expr" not in data:
        return jsonify({"success": False, "error": "Missing 'expr' parameter"}), 400

    expr = data["expr"]

    # ------------------------------------------------------------
    # 1. Pre-processing LaTeX → stringa “Python-style”
    # Prima normalizza \frac23 → \frac{2}{3}
    expr = normalize_frac_syntax(expr)
    # Corregge casi errati come \frac(1)(3) → (1)/(3)
    import re
    expr = re.sub(r'\\frac\s*\(\s*([^)]+?)\s*\)\s*\(\s*([^)]+?)\s*\)', r'(\1)/(\2)', expr)
    expr = expr.replace("^", "**")
    # Rimuovi eventuali empty exponent braces e asterischi doppi vuoti (es. e^{}^{})
    expr = expr.replace("**{}", "")
    expr = expr.replace(r"\left", "").replace(r"\right", "")
    # Convert any remaining LaTeX braces to parentheses (fix exponent braces)
    expr = expr.replace("{", "(").replace("}", ")")

    # Mappa di funzioni LaTeX → SymPy
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
        r"\pi": "pi",
    }
    for latex_name, sympy_name in _func_map.items():
        expr = expr.replace(latex_name, sympy_name)

    # Import re per le sostituzioni
    import re

    # Supporto plain 'arctan(...)' → 'atan(...)'
    expr = re.sub(r"\barctan\(", "atan(", expr)

    forbidden_funcs = ["log", "sin", "cos", "tan", "asin", "acos", "atan", "sec", "csc", "cot", "exp"]
    for func in forbidden_funcs:
        if re.search(rf"\b{func}\b(?!\()", expr):
            return jsonify({"success": False, "error": f"La funzione '{func}' deve essere scritta con parentesi: ad es. '{func}(...)'"})

    # · → *
    expr = expr.replace(r"\cdot", "*")

    # \sqrt{…}  |  \sqrt2
    expr = re.sub(r"sqrt\{([^{}]+)\}", r"sqrt(\1)", expr)
    expr = re.sub(r"sqrt([0-9]+)", r"sqrt(\1)", expr)

    # Potenze con graffe:  x**{cos x} → x**(cos x)
    expr = re.sub(r"\*\*\s*\{([^{}]+)\}", r"**(\1)", expr)

    # cos**2(x)  |  cos**2 x
    expr = re.sub(
        r"\b(sin|cos|tan|sec|csc|cot|asin|acos|atan)\*\*([0-9]+)\s*\(\s*([^)]+?)\s*\)",
        r"(\1(\3))**\2",
        expr,
    )
    expr = re.sub(
        r"\b(sin|cos|tan|sec|csc|cot|asin|acos|atan)\*\*([0-9]+)\s*([A-Za-z_][A-Za-z0-9_]*)",
        r"(\1(\3))**\2",
        expr,
    )

    # log|…|  →  log(Abs(…))
    expr = re.sub(r"log\|([^|]+)\|", r"log(Abs(\1))", expr)

    # asin|x| & varianti
    expr = re.sub(r"\b(asin|acos|atan)\s*Abs\(([^()]*)\)", r"\1(Abs(\2))", expr)

    # |…| → Abs(…)
    expr = re.sub(r"\|([^|]+)\|", r"Abs(\1)", expr)
    expr = expr.replace("|", "")  # barre residue

    # sinx → sin(x)    —    (cos x) → (cos(x))
    expr = re.sub(
        r"\b(sin|cos|tan|sec|csc|cot|asin|acos|atan|log|sqrt|exp)\s*([A-Za-z_][A-Za-z0-9_]*)",
        r"\1(\2)",
        expr,
    )
    expr = re.sub(
        r"\(\s*(sin|cos|tan|sec|csc|cot|asin|acos|atan|log|sqrt|exp)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\)",
        r"(\1(\2))",
        expr,
    )

    # * implicito prima di funzioni / Abs
    expr = re.sub(
        r"(?<=[0-9A-Za-z_)\]])(?=\b(?:asin|acos|atan|sin|cos|tan|sec|csc|cot|log|sqrt|exp)\()",
        "*",
        expr,
    )
    expr = re.sub(r"(?<=[0-9A-Za-z_)\]])(?=Abs\()", "*", expr)
    expr = re.sub(r"\b(asin|acos|atan)\s*\*?\s*Abs\(([^()]*)\)", r"\1(Abs(\2))", expr)

    # Prima convertiamo tutti i \frac… in (a)/(b)
    expr = _latex_frac_to_python(expr)

    debug_expr = re.sub(r"Abs\(([^)]+)\)", r"|\1|", expr)
    print("[studio_funzione] normalized expression:", debug_expr)

    # ------------------------------------------------------------
    # 2. Parsing con SymPy
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
        "e": sp.E,
        "E": sp.E,
        "S": sp.S,
        # Tratta "pi" come variabile greca, non come costante
        "pi": sp.Symbol("pi", real=True),
    }

    # define real symbol for x and register for parsing
    x = sp.Symbol("x", real=True)
    allowed_funcs["x"] = x
 
    transformations = standard_transformations + (implicit_multiplication_application,)
    try:
        # ── Minima patch: rimuove Abs() per potenze razionali con denominatore dispari
        f_expr = parse_expr(expr, local_dict=allowed_funcs, transformations=transformations)
        f_expr = f_expr.replace(
            lambda e: (
                isinstance(e, Pow)
                and isinstance(e.base, Abs)
                and e.exp.is_Rational
                and e.exp.q % 2 == 1
                and e.exp.p >= 0
            ),
            lambda e: Pow(e.base.args[0], e.exp, evaluate=False)
        )
        print("[studio_funzione] parsed expression:", f_expr)
    except Exception as e:
        return jsonify({"success": False, "error": f"Parsing error: {str(e)}"}), 400

    # alias for real symbol
    x_real = x
    steps = []

    # ------------------------------------------------------------
    # 0. Funzione inserita
    latex_fun = sp.latex(f_expr)
    latex_fun = latex_fun.replace(r"\log", r"\ln")
    latex_fun = latex_fun.replace(r"\infty", r"+\infty")
    latex_fun = latex_fun.replace(r"-+\infty", r"-\infty")
    latex_fun = re.sub(r"Abs\(([^)]+)\)", r"|\1|", latex_fun)
    steps.append({"title": "Funzione inserita:", "content": rf"f(x)= {latex_fun}"})
    print("[studio_funzione] function expression:", f_expr)
    # ------------------------------------------------------------
    # Riscrittura della funzione se contiene Abs
    abs_piecewise_expr = None

    # ------------------------------------------------------------
    # 1. Periodicità (modificato)
    # ------------------------------------------------------------
    periodic = False
    try:
        a = periodicity(f_expr, x)
        space = a/20
        check = []
        for i in range(20):
            check.append(f_expr.subs(x, i*space).evalf())
        nperiod = 1
        startidx = 0
        p1 = 1
        p2 = startidx
        while p1 < 20:
            if check[p1] == check[p2]:
                final = p1
                while p1 < 20 and p2 < final and check[p1] == check[p2]:
                    p1 += 1
                    p2 += 1
                    if final - p2 == 1:
                        nperiod += 1
                        break
            p1 += 1
        new_period = a / nperiod
        periodic = True
        T = new_period
        period_latex = sp.latex(new_period)
        steps.append({
            "title": "Periodicità:",
            "content": rf"\text{{La funzione è periodica di periodo }}{period_latex}." ,
        })
    except Exception:
        periodic = False

    # ------------------------------------------------------------
    # 2. Dominio (ristretto se periodica)
    # ------------------------------------------------------------
    try:
        full_dom = safe_domain(f_expr, x)
    except Exception:
        try:
            sing_try = sp.singularities(f_expr, x)
            full_dom = S.Reals if sing_try is S.EmptySet else Complement(S.Reals, sing_try)
        except Exception:
            full_dom = S.Reals

    if periodic:
        # restringi al primo periodo [0, T], escludendo singolarità interne
        try:
            zeros_cos = solveset(sp.cos(x), x, domain=Interval(0, T))
            sing_points = list(zeros_cos) if isinstance(zeros_cos, FiniteSet) else []
        except Exception:
            sing_points = []
        dom = Interval(0, T)
        if sing_points:
            dom = dom - FiniteSet(*sing_points)
    else:
        dom = full_dom

    dom_latex = sp.latex(dom).replace(r"\infty", r"+\infty").replace(r"-+\infty", r"-\infty")
    steps.append({"title": "Dominio:", "content": rf"{{Dom}}(f)= {dom_latex}"})

    def rewrite_abs_piecewise(expr, x, dom_f):
        # Handle any |u|**exp hidden in the expression by rewriting it as a two‑branch Piecewise
        # Handle any |u|**exp hidden in the expression by rewriting it as a two-branch Piecewise
        def _abs_pow_to_piecewise(e):
            inner = e.base.args[0]
            exp = e.exp
            # branch for x >= 0
            pos = inner**exp
            # branch for x < 0
            neg = (-inner)**exp
            return sp.Piecewise((pos, inner >= 0), (neg, inner < 0), evaluate=False)

        expr = expr.replace(
            lambda e: isinstance(e, sp.Pow) and isinstance(e.base, sp.Abs),
            _abs_pow_to_piecewise
        )
        abs_vars = list(expr.atoms(sp.Abs))
        replacements = {}

        for abs_expr in abs_vars:
            inner = abs_expr.args[0]
            try:
                abs_zeros = solveset(inner, x, domain=S.Reals)
            except Exception:
                abs_zeros = sp.S.EmptySet
            branches = []
        
            if isinstance(abs_zeros, FiniteSet) and len(abs_zeros) == 1:
                a = list(abs_zeros)[0]

                left_expr  = -inner      # −x
                right_expr =  inner      #  x
    
                cond_left  = x < a
                cond_right = x >= a

                branches.append((right_expr, cond_right))
                branches.append((left_expr,  cond_left))
            else:
                cond_pos = inner >= 0
                cond_neg = inner < 0
                branches.append((inner, cond_pos))
                branches.append((-inner, cond_neg))
            # Evita che SymPy normalizzi l'ultimo ramo con True
            piecewise_expr = sp.Piecewise(*branches, evaluate=False)
            replacements[abs_expr] = piecewise_expr
        return expr.xreplace(replacements)

    # --- Single-phase piecewise rewrite with per-branch domain intersection ---
    if f_expr.has(Abs):
        # 1) special-case: solitary Abs(...)**exp
        if isinstance(f_expr, sp.Pow) and isinstance(f_expr.base, sp.Abs):
            inner = f_expr.base.args[0]
            exp = f_expr.exp
            expr_pw = sp.Piecewise(
                (inner**exp, inner >= 0),
                ((-inner)**exp, inner < 0),
                evaluate=False
            )
        else:
            # general rewrite
            expr_pw = rewrite_abs_piecewise(f_expr, x, dom)
        # ensure we have a top-level Piecewise
        if not isinstance(expr_pw, sp.Piecewise):
            # lift any sub-Piecewise nodes (e.g. log(Piecewise(...))) to top-level
            pw_nodes = list(expr_pw.atoms(sp.Piecewise))
            if pw_nodes:
                pw = pw_nodes[0]
                lifted = []
                for sub_expr, cond in pw.args:
                    new_expr = expr_pw.xreplace({pw: sub_expr})
                    lifted.append((new_expr, cond))
                expr_pw = sp.Piecewise(*lifted, evaluate=False)
            else:
                expr_pw = sp.Piecewise((expr_pw, True))
        # 2) for each branch, compute domain and intersect with branch condition
        from sympy import Contains

        # --- Flatten nested Piecewise in expr_pw to get atomic branches ---
        from sympy import Piecewise, And
        # start with top-level branches
        branches = list(expr_pw.args)
        flattened = []
        # iterative flattening
        for expr_i, cond_i in branches:
            to_process = [(expr_i, cond_i)]
            while to_process:
                e, c = to_process.pop()
                # find nested Piecewise
                pwnodes = [pw for pw in e.atoms(Piecewise)]
                if pwnodes:
                    pw = pwnodes[0]
                    for sub_e, sub_c in pw.args:
                        new_expr = e.xreplace({pw: sub_e})
                        new_cond = And(c, sub_c)
                        to_process.append((new_expr, new_cond))
                else:
                    flattened.append((e, c))
        # now build final branches from flattened list
        final_branches = []
        branch_domains = []
        for branch_expr, branch_cond in flattened:
            print("[debug][Abs-PW] branch_expr =", branch_expr)
            print("[debug][Abs-PW] branch_cond =", branch_cond)
            # Calcola la condizione di ramo suddividendo con solveset e intersezione
            # Estrai le singole disuguaglianze dalla condizione combinata
            conds = branch_cond.args if isinstance(branch_cond, And) else [branch_cond]
            cond_set = dom  # parte dal dominio principale
            for cond_i in conds:
                try:
                    s = solveset(cond_i, x, domain=dom)
                except Exception:
                    s = S.Reals
                cond_set = cond_set.intersect(s)
            print("[debug][Abs-PW] combined cond_set =", cond_set)
            branch_domain = cond_set
            print("[debug][Abs-PW] dom =", dom)
            print("[debug][Abs-PW] cond_set ∩ dom =", branch_domain)
            # Skip branches whose domain is empty
            if branch_domain == S.EmptySet:
                continue
            cond_expr = Contains(x, branch_domain)
            final_branches.append((branch_expr, cond_expr))
            branch_domains.append(branch_domain)
        # 3) build final piecewise and global domain
        abs_piecewise_expr = sp.Piecewise(*final_branches)

        # produce LaTeX
        abs_pw_latex = sp.latex(abs_piecewise_expr).replace("for", "se").replace("otherwise", "altrimenti")
        piece_lines = []
        for (expr_i, _), dom_i in zip(final_branches, branch_domains):
            expr_tex = sp.latex(expr_i).replace(r"\log", r"\ln")
            dom_tex  = sp.latex(dom_i).replace(r"\infty", r"+\infty").replace(r"-+\infty", r"-\infty")
            piece_lines.append(rf"{expr_tex}\quad \text{{se }} x \in {dom_tex}")

        abs_pw_latex = (
            r"\left\{\begin{array}{l}"
            + r"\\[6pt]".join(piece_lines) +
            r"\end{array}\right."
        )
        steps.append({
            "title": "Riscrittura a tratti:",
            "content": rf"f(x) = {abs_pw_latex}"
        })

    # ------------------------------------------------------------
    # 2. Parità
    # ------------------------------------------------------------
    try:
        f_x = f_expr
        f_mx = f_expr.subs(x, -x)
        diff = sp.simplify(f_x - f_mx)
        sum_ = sp.simplify(f_x + f_mx)
        if diff == 0:
            parity_latex = r"\text{La funzione è pari.}"
        elif sum_ == 0:
            parity_latex = r"\text{La funzione è dispari.}"
        else:
            parity_latex = r"\text{La funzione non è né pari né dispari.}"
    except Exception:
        parity_latex = r"\text{Parità non determinabile.}"

    steps.append({"title": "Parità:", "content": parity_latex})

    # ------------------------------------------------------------
    # 4. Intersezione con l'asse y (f(0))
    # ------------------------------------------------------------
    try:
        y0_val = sp.simplify(f_expr.subs(x, 0))
        # Simplify (-1)**(odd rational) to -1
        y0_val = y0_val.replace(
            lambda e: isinstance(e, sp.Pow)
                      and e.base == -1
                      and e.exp.is_Rational
                      and e.exp.q % 2 == 1,
            lambda e: -1
        )
        if dom.contains(0) is not sp.S.true:
            raise ValueError("0 non nel dominio")

        y_content = rf"f(0)= {sp.latex(y0_val)}"

    except Exception:
        y_content = r"\text{Nessuna intersezione con l'asse $y$.}"

    steps.append({"title": "Intersezione con $y$:", "content": y_content})

    zeros_set = sp.S.EmptySet  # default, sovrascritto se calcolato correttamente
    # ------------------------------------------------------------
    # 5. Intersezione con l'asse x (zeri)
    try:
        zeros_set = sp.solveset(f_expr, x, domain=sp.S.Reals)
        # Zeri esplicitamente determinati
        if isinstance(zeros_set, sp.FiniteSet) and all(z.is_real and z.is_number for z in zeros_set):
            zeros_latex = sp.latex(zeros_set)
            zeros_content = rf"f(x) = 0, \quad \text{{per }} x \in {zeros_latex}"
            steps.append({"title": "Intersezione con $x$ (zeri della funzione):", "content": zeros_content})
        # Nessuna intersezione reale ⇒ mostra comunque lo step
        elif zeros_set == sp.S.EmptySet:
            zeros_content = r"\text{Nessuna intersezione con l'asse } x."
            steps.append({"title": "Intersezione con $x$ (zeri della funzione):", "content": zeros_content})
        # Zeri non elementari
        else:
            zeros_latex = r"\text{Zeri non esprimibili in forma elementare.}"
            zeros_content = zeros_latex
            steps.append({"title": "Intersezione con $x$ (zeri della funzione):", "content": zeros_content})
    except Exception:
        zeros_latex = r"\text{Zeri non determinabili con i metodi attuali.}"
        zeros_content = zeros_latex
        steps.append({"title": "Intersezione con $x$ (zeri della funzione):", "content": zeros_content})

    # ------------------------------------------------------------
    # 6. Studio del segno: f(x) > 0
    try:
        # Risolve direttamente l'inequazione f_expr > 0 sul dominio
        sol = solveset(f_expr > 0, x, domain=dom)
        sign_latex = sp.latex(sol)
        # Mostra '+' davanti a infinito
        sign_latex = sign_latex.replace(r"\infty", r"+\infty").replace(r"-+\infty", r"-\infty")
        latex_positive = rf"f(x) > 0, \quad \text{{per }} x \in {sign_latex}"
        # Se l'insieme positivo è vuoto, aggiungi la riga extra
        if sol == sp.S.EmptySet:
            latex_positive += r" \quad \Rightarrow \quad f(x) < 0 \quad \forall x \in \mathbb{R}"
        sign_content = latex_positive
    except Exception:
        sign_content = r"\text{Studio del segno non determinabile in modo elementare.}"
    steps.append({"title": "Studio del segno:", "content": sign_content})

    # ------------------------------------------------------------
    # 7. Limiti
    # ------------------------------------------------------------
    limit_lines = []
    try:
        sing_set = sp.singularities(f_expr, x)
    except Exception:
        sing_set = sp.S.EmptySet

    fin_sing = [p for p in sing_set if p.is_real] if isinstance(sing_set, sp.FiniteSet) else []

    # Prepara i punti di singolarità da riutilizzare nello step degli asintoti verticali
    punti_singolarita = set(fin_sing)

    def format_latex_limit(val):
        if val == sp.oo:
            return r"+\infty"
        elif val == -sp.oo:
            return r"-\infty"
        return sp.latex(val)

    # Limiti per x → ±∞
    for infinity_symbol, label in [(sp.oo, r"+\infty"), (-sp.oo, r"-\infty")]:
        # Per +∞, testiamo un punto molto grande (es. 10**6); per -∞, molto piccolo (es. -10**6)
        test_point = 10**6 if infinity_symbol == sp.oo else -10**6
        if not dom.contains(sp.Float(test_point)):
            continue
        from sympy.functions.elementary.miscellaneous import real_root
        # Handle real fractional powers x**(p/q) correctly at ±∞
        expr_for_limit = f_expr
        if isinstance(f_expr, sp.Pow) and f_expr.exp.is_Rational:
            p, q = f_expr.exp.p, f_expr.exp.q
            expr_for_limit = real_root(x, q)**p
        try:
            lim_val = sp.limit(expr_for_limit, x, infinity_symbol)
            # Normalizza ±∞ e segni
            lim_val = evaluate_inf_sign(lim_val)
        except Exception:
            # non è possibile calcolare il limite a questo estremo
            continue
        if isinstance(lim_val, AccumulationBounds):
            limit_lines.append(
                rf"\displaystyle\lim_{{x\to {label}}} f(x)=\text{{non esiste (oscillazione infinita)}}"
            )
        else:
            limit_lines.append(
                rf"\displaystyle\lim_{{x\to {label}}} f(x)={format_latex_limit(lim_val)}"
            )

    from sympy import Union

    # Estremi di accumulazione ai bordi del dominio
    if isinstance(dom, Union):
        intervals = sorted(dom.args, key=lambda i: i.start if isinstance(i, sp.Interval) else -sp.oo)
    else:
        intervals = [dom]

    # Aggiunta degli estremi aperti come punti di singolarità
    for interv in intervals:
        if isinstance(interv, sp.Interval):
            if interv.left_open and interv.start.is_finite:
                punti_singolarita.add(interv.start)
            if interv.right_open and interv.end.is_finite:
                punti_singolarita.add(interv.end)

    for interv in intervals:
        if isinstance(interv, sp.Interval):
            a, b = interv.start, interv.end
            left_open, right_open = interv.left_open, interv.right_open

            # Se a è un estremo finito ed è aperto ⇒ limite da destra (x → a⁺)
            if a.is_finite and left_open:
                try:
                    lim_right = sp.limit(f_expr, x, a, dir="+")
                    # Normalizza con evaluate_inf_sign
                    lim_right = evaluate_inf_sign(lim_right)
                    limit_lines.append(
                        rf"\displaystyle\lim_{{x\to {format_latex_limit(a)}^+}} f(x)={format_latex_limit(lim_right)}"
                    )
                except Exception:
                    limit_lines.append(
                        rf"\displaystyle\lim_{{x\to {format_latex_limit(a)}^+}} f(x)=\nexists"
                    )

            # Se b è un estremo finito ed è aperto ⇒ limite da sinistra (x → b⁻)
            if b.is_finite and right_open:
                try:
                    lim_left = sp.limit(f_expr, x, b, dir="-")
                    # Normalizza con evaluate_inf_sign
                    lim_left = evaluate_inf_sign(lim_left)
                    limit_lines.append(
                        rf"\displaystyle\lim_{{x\to {format_latex_limit(b)}^-}} f(x)={format_latex_limit(lim_left)}"
                    )
                except Exception:
                    limit_lines.append(
                        rf"\displaystyle\lim_{{x\to {format_latex_limit(b)}^-}} f(x)=\nexists"
                    )

    limits_content = r"\begin{align*}" + r"\\ ".join(limit_lines) + r"\end{align*}" if limit_lines else r"\text{Nessun limite notevole da segnalare.}"
    steps.append({"title": "Limiti nei punti di accumulazione e agli estremi:", "content": limits_content})

    # ------------------------------------------------------------
    # 8. Asintoti
    # ------------------------------------------------------------
    from collections import defaultdict

    asintoti = []
    # Limiti laterali in punti di singolarità per asintoti verticali
    for p in punti_singolarita:
        try:
            lim_sx = sp.limit(f_expr, x, p, dir="-")
        except Exception:
            lim_sx = None
        try:
            lim_dx = sp.limit(f_expr, x, p, dir="+")
        except Exception:
            lim_dx = None
        # Classifica asintoto verticale
        if lim_sx in (sp.oo, -sp.oo) and lim_dx in (sp.oo, -sp.oo):
            asintoti.append(rf"x={sp.latex(p)} \quad \text{{Asintoto verticale bilatero}}")
        elif lim_sx in (sp.oo, -sp.oo):
            asintoti.append(rf"x={sp.latex(p)} \quad \text{{Asintoto verticale sinistro}}")
        elif lim_dx in (sp.oo, -sp.oo):
            asintoti.append(rf"x={sp.latex(p)} \quad \text{{Asintoto verticale destro}}")

    # Asintoti orizzontali: limiti per x→±∞
    horizontal_vals = []
    for inf, label in [(sp.oo, r"+\infty"), (-sp.oo, r"-\infty")]:
        try:
            lim_inf = sp.limit(f_expr, x, inf)
        except Exception:
            lim_inf = None
        # Solo se limite esiste ed è finito
        if lim_inf is not None and lim_inf not in (sp.oo, -sp.oo) and getattr(lim_inf, "is_real", False):
            try:
                horizontal_vals.append(float(lim_inf))
                asintoti.append(
                    rf"y={sp.latex(lim_inf)} \quad \text{{Asintoto orizzontale per }}x\to {label}"
                )
            except (TypeError, ValueError):
                # Limite non numerico (p.es. AccumulationBounds) ⇒ non esiste
                asintoti.append(
                    rf"\displaystyle\lim_{{x\to {label}}} f(x) = \nexists "
                    r"\quad\text{(nessun asintoto orizzontale)}"
                )

    # Componi il contenuto
    if asintoti:
        content_asintoti = r"\begin{align*}" + r"\\ ".join(asintoti) + r"\end{align*}"
    else:
        content_asintoti = r"\text{Nessun asintoto verticale o orizzontale individuato.}"
    steps.append({"title": "Asintoti:", "content": content_asintoti})

    # 9. Asintoto obliquo (dopo Asintoti)
    oblique_lines = []
    oblique_list = []  # raccolta (m, q) da passare al frontend
    # Blocchi protettivi per x→+∞ e x→-∞
    for inf, label, m_name, q_name in [
        (sp.oo, r"+\infty", "m_{+}", "q_{+}"),
        (-sp.oo, r"-\infty", "m_{-}", "q_{-}")
    ]:
        try:
            try:
                m_val = sp.limit(f_expr / x, x, inf)
                if not m_val.is_real:
                    raise ValueError("m non reale")
            except Exception as e:
                oblique_lines.append(rf"\text{{Per }} x \to {label}:")
                oblique_lines.append(rf"\displaystyle\lim_{{x \to {label}}} \frac{{f(x)}}{{x}} = \nexists")
                oblique_lines.append(r"\text{Pertanto non è possibile determinare il coefficiente angolare } m.")
                continue
            else:
                if m_val == 0:
                    # Giustificazione formale per asintoto orizzontale
                    oblique_lines.append(
                        rf"{m_name} = \lim_{{x\to {label}}}\frac{{f(x)}}{{x}} = 0"
                    )
                    oblique_lines.append(
                        r"\text{Poiché } m_{{" + label + r"}} = 0\text{, non esiste asintoto obliquo.}"
                    )
                    continue
                oblique_lines.append(rf"{m_name} = \lim_{{x\to {label}}}\frac{{f(x)}}{{x}} = {sp.latex(m_val)}")
                try:
                    q_val = sp.limit(f_expr - m_val * x, x, inf)
                    if not q_val.is_real:
                        raise ValueError("q non reale")
                except Exception as e:
                    oblique_lines.append(rf"\displaystyle\lim_{{x \to {label}}} \frac{{f(x)}}{{x}} = {sp.latex(m_val)}")
                    oblique_lines.append(r"\displaystyle\lim_{x \to " + label + r"} \left[f(x) - m x\right] = \nexists")
                    oblique_lines.append(r"\text{Pertanto non è possibile determinare l'intercetta } q.")
                    continue
                else:
                    if isinstance(q_val, AccumulationBounds):
                        q_line = rf"{q_name} = \lim_{{x\to {label}}}(f(x)-{m_name}x) \text{{ oscilla tra }}[{sp.latex(q_val.min)},\,{sp.latex(q_val.max)}]"
                        y_line = r"\text{Termine noto non esistente ⇒ asintoto obliquo non individuabile}"
                    else:
                        q_line = rf"{q_name} = \lim_{{x\to {label}}}(f(x)-{m_name}x) = {sp.latex(q_val)}"
                        y_expr = sp.simplify(m_val * x + q_val)
                        y_line = rf"y_{{{label}}} = {sp.latex(y_expr)} \quad \text{{asintoto obliquo per }}x\to {label}"
                        # salva per il frontend
                        if all(v.is_real and v.is_finite for v in (m_val, q_val)):
                            oblique_list.append({
                                "type": "oblique",
                                "m": float(m_val),
                                "q": float(q_val)
                            })
                    oblique_lines.extend([q_line, y_line])
        except Exception:
            continue

    if oblique_lines:
        obliquo_content = r"\begin{array}{l}" + r"\\ ".join(oblique_lines) + r"\end{array}"
    else:
        obliquo_content = r"\text{Nessun asintoto obliquo individuato.}"

    steps.append({"title": "Asintoto obliquo:", "content": obliquo_content})

    # ------------------------------------------------------------
    # 10. Continuità e Discontinuità
    # ------------------------------------------------------------
    from collections import defaultdict

    discontinuities = []
    point_limits = defaultdict(dict)
    # Raggruppo i limiti sinistro/destro già in limit_lines
    import re
    for line in limit_lines:
        m = re.match(
            r".*\\lim_\{x\\to\s*(.*?)\^([+-])\}.*?f\(x\)\s*=\s*(.+)",
            line
        )
        if m:
            a, dir_, val = m.groups()
            key = a.replace(" ", "").strip()
            val = val.replace(r"\displaystyle", "").strip()
            if dir_ == "-":
                point_limits[key]["left"] = val
            else:
                point_limits[key]["right"] = val

    # Classificazione secondo le 3 specie con giustificazioni
    for a, lims in point_limits.items():
        l = lims.get("left")
        r = lims.get("right")
        # Limiti laterali
        lim_sx = l or r"\nexists"
        lim_dx = r or r"\nexists"

        if l is None or r is None:
            # Seconda specie: almeno un limite è infinito o non esiste
            discontinuities.append(
                rf"x={a}: \lim_{{x\to {a}^-}}f(x)={lim_sx},\ \lim_{{x\to {a}^+}}f(x)={lim_dx},\ "
                r"\text{discontinuità di seconda specie (limite mancante o non finito)}"
            )
        elif l != r:
            if any(s in (l, r) for s in [r"+\infty", r"-\infty", r"\infty"]):
                # Seconda specie: almeno un limite è infinito o non esiste
                discontinuities.append(
                    rf"x={a}: \lim_{{x\to {a}^-}}f(x)={l},\ \lim_{{x\to {a}^+}}f(x)={r},\ "
                    r"\text{almeno uno dei limiti è infinito ⇒ discontinuità di seconda specie}"
                )
            else:
                # Prima specie: limiti finiti ma diversi
                discontinuities.append(
                    rf"x={a}: \lim_{{x\to {a}^-}}f(x)={l},\ \lim_{{x\to {a}^+}}f(x)={r},\ "
                    r"\text{limiti sinistro e destro non coincidono ⇒ discontinuità di prima specie}"
                )
        else:
            # Limiti uguali (l == r)
            # Se sono ±∞ ⇒ discontinuità di seconda specie (limite infinito bilatero)
            if any(s in l for s in [r"+\infty", r"-\infty", r"\infty"]):
                discontinuities.append(
                    rf"x={a}: \lim_{{x\to {a}^\pm}}f(x)={l},\ "
                    r"\text{limite bilatero infinito ⇒ discontinuità di seconda specie}"
                )
                continue  # passa al prossimo punto

            # Altrimenti sono finiti e coincidenti → controlla valore della funzione
            # Convert LaTeX fraction and pi to Python syntax
            py_a = _latex_frac_to_python(a)
            py_a = py_a.replace("\\pi", "pi")
            xa = sp.sympify(py_a)
            in_domain = dom.contains(xa) is sp.S.true
            if not in_domain:
                # Punto fuori dal dominio ma limiti coincidenti ⇒ eliminabile
                discontinuities.append(
                    rf"x={a}: \lim_{{x\to {a}^\pm}}f(x)={l},\ "
                    r"\text{punto fuori dal dominio ⇒ discontinuità eliminabile (terza specie)}"
                )
            else:
                try:
                    fa = f_expr.subs(x, xa)
                    fa_tex = sp.latex(fa)
                    if fa_tex != l:
                        discontinuities.append(
                            rf"x={a}: \lim_{{x\to {a}^\pm}}f(x)={l},\ f({a})={fa_tex},\ "
                            r"\text{valore diverso ⇨ discontinuità eliminabile (terza specie)}"
                        )
                    # Se fa_tex == l ⇒ punto di continuità, nessuna discontinuità
                except Exception:
                    discontinuities.append(
                        rf"x={a}: \lim_{{x\to {a}^\pm}}f(x)={l},\ f({a})\ \text{{non valutabile}},\ "
                        r"\text{⇒ discontinuità eliminabile (terza specie)}"
                    )

    if discontinuities:
        disc_content = r"\begin{align*}" + r"\\ ".join(discontinuities) + r"\end{align*}"
    else:
        disc_content = r"\text{La funzione è continua } \forall x \in Dom(f)."

    steps.append({"title": "Continuità e Discontinuità:", "content": disc_content})

    # ------------------------------------------------------------
    # 11. Derivata della funzione a tratti
    # ------------------------------------------------------------
    # Usa la riscrittura piecewise calcolata sopra (abs_piecewise_expr)
    from sympy import Eq, Piecewise
    # Determina la funzione da derivare: preferisci abs_piecewise_expr
    f_pw = abs_piecewise_expr if abs_piecewise_expr is not None else f_expr
    # Calcola la derivata di f_pw
    f_prime_raw = sp.diff(f_pw, x).simplify()
    # Avvolgi in Piecewise se necessario
    if isinstance(f_prime_raw, Piecewise):
        f_prime_pw = f_prime_raw
    else:
        f_prime_pw = Piecewise((f_prime_raw, True))
    # Rimuovi eventuali branch centrali non derivabili (Eq)
    cleaned = []
    if isinstance(f_prime_pw, Piecewise):
        for expr_i, cond_i in f_prime_pw.args:
            if not cond_i.has(Eq):
                expr_s = sp.simplify(expr_i)
                cleaned.append((expr_s, cond_i))
        f_prime_pw = Piecewise(*cleaned)
    # Se non è Piecewise, f_prime_pw resta com'è
    # Prepara LaTeX per output
    # Se la derivata è Piecewise con un solo ramo, mostra solo l'espressione (senza le parentesi graffe)
    if isinstance(f_prime_pw, Piecewise) and len(f_prime_pw.args) == 1:
        expr0 = f_prime_pw.args[0][0].simplify()
        deriv_tex = (
            sp.latex(expr0)
            .replace(r"\log", r"\ln")
            .replace(r"\infty", r"+\infty")
            .replace(r"-+\infty", r"-\infty")
        )
        deriv_content = rf"f'(x) = {deriv_tex}"
    else:
        f_prime_pw_latex = (
            sp.latex(f_prime_pw.simplify())
            .replace("for", "se")
            .replace("otherwise", "altrimenti")
            .replace(r"\log", r"\ln")
            .replace(r"\infty", r"+\infty")
            .replace(r"-+\infty", r"-\infty")
        )
        deriv_content = rf"f'(x) = {f_prime_pw_latex}"
    steps.append({"title": "Derivata:", "content": deriv_content})

    # ------------------------------------------------------------
    # 12. Punti di non derivabilità
    # ------------------------------------------------------------
    try:
        # Calcola il dominio di f e di f'
        f_prime_sym = f_prime_pw
        try:
            # Calcola la continuità di f' solo all’interno del dominio ristretto
            cont_dom_fprime = continuous_domain(f_prime_sym, x, dom)
        except Exception:
            cont_dom_fprime = dom
        # raw ND = dominio ristretto \ Dom(f')
        raw_nd = dom - cont_dom_fprime
        # Insieme dei punti di non derivabilità
        nondiff_set = raw_nd
        # ------------------------------------------------------------------
        # Aggiunta dei punti di raccordo (junction points) delle funzioni
        # Piecewise: se il punto appartiene al dominio di f, deve essere
        # considerato candidato punto non derivabile (es. punto angoloso).
        # Se invece il punto NON appartiene al dominio di f, viene ignorato.
        # ------------------------------------------------------------------
        junction_pts = set()
        if isinstance(f_pw, sp.Piecewise):
            for _, cond in f_pw.args:
                # Caso Contains(x, <set>)
                if isinstance(cond, sp.Contains):
                    set_i = cond.args[1]
                    # Se è unione, scorre ogni parte
                    parts = set_i.args if isinstance(set_i, sp.Union) else [set_i]
                    for part in parts:
                        if isinstance(part, sp.Interval):
                            if part.start.is_finite:
                                junction_pts.add(part.start)
                            if part.end.is_finite:
                                junction_pts.add(part.end)
                        elif isinstance(part, sp.FiniteSet):
                            junction_pts.update(part)
                # Caso disuguaglianze semplici (x < a, x <= a, x > a, x >= a)
                elif cond.is_Relational and cond.lhs in (x, x_real) and cond.rhs.is_real:
                    junction_pts.add(cond.rhs)

        # Mantieni solo i punti che appartengono al dominio principale di f
        junction_pts = [p for p in junction_pts if dom.contains(p) is sp.S.true]

        # Aggiungi i punti di raccordo al set di non derivabilità, evitando duplicati
        if junction_pts:
            nondiff_set = nondiff_set.union(sp.FiniteSet(*junction_pts))
        # ------------------------------------------------------------------
        # Formula LaTeX come R \ (Dom(f) \ Dom(f'))
        nd_tex = sp.latex(nondiff_set)
        nd_content = rf"""N_D = \mathbb{{R}} \setminus \bigl(\mathrm{{Dom}}(f)\setminus\mathrm{{Dom}}(f')\bigr)
= {nd_tex}"""
    except Exception as e:
        nd_content = r"\text{Non derivabilità non valutabile con i metodi attuali.}"

    steps.append({"title": "Punti di non derivabilità:", "content": nd_content})

    # Estrazione dei punti di non derivabilità in lista Python
    try:
        # 'nondiff_set' è il SymPy Set calcolato sopra
        punti_non_derivabilita = sorted(
            pt for pt in nondiff_set
            if getattr(pt, 'is_real', False) and pt.is_number
        )
    except Exception:
        punti_non_derivabilita = []

    # ------------------------------------------------------------
    # STEP 13 – Classificazione punti di non derivabilità
    # ------------------------------------------------------------
    steps.append({
        "title": "Classificazione dei punti di non derivabilità:",
        "hidden": False,
        "latex": True,
        "content": ""
    })

    # Filtro sui punti per includere solo quelli nel dominio di f
    # Calcola il dominio di continuità della funzione originale
    x_real = sp.Symbol("x", real=True)
    f_expr_real = f_expr.subs(x, x_real)
    try:
        cont_dom_f = continuous_domain(f_expr_real, x_real, S.Reals)
    except Exception:
        cont_dom_f = S.Reals
    # Seleziona solo i punti non derivabili che appartengono al dominio di f
    valid_pts = [p for p in punti_non_derivabilita if cont_dom_f.contains(p) is S.true]
    # Se non ci sono punti da studiare, mostra il messaggio e ritorna
    if not valid_pts:
        steps[-1]["content"] = r"\text{La funzione è derivabile }\forall x\in \mathrm{Dom}(f)."
    else:
        f_prime_raw = sp.diff(f_expr_real, x_real)

        def latex_limit(val):
            # 1) Se contiene parte immaginaria non nulla
            try:
                im_val = sp.im(val).evalf()
                if im_val != 0:
                    return r"\nexists"
            except Exception:
                pass

            # 2) Infinito reale
            if val == sp.oo:
                return "+\\infty"
            if val == -sp.oo:
                return "-\\infty"
            if getattr(val, "is_infinite", False):
                if val.is_positive:
                    return "+\\infty"
                if val.is_negative:
                    return "-\\infty"
                s = str(val)
                return "-\\infty" if s.startswith("-") else "+\\infty"

            # 3) Valore finito reale
            return sp.latex(val)
        

        lines = []
        for punto in valid_pts:
            # Calcolo limiti unilaterali tramite rapporto incrementale
            h = sp.Symbol('h', real=True)
            x0_str = sp.latex(punto)
            # Rapporto incrementale base con x0
            ratio_tex = rf"\frac{{f\bigl({x0_str} + h\bigr) - f\bigl({x0_str}\bigr)}}{{h}}"
            lim_sx = sp.limit(
                (force_real_roots(f_expr).subs(x_real, punto + h) - force_real_roots(f_expr).subs(x_real, punto)) / h,
                h, 0, dir='-'
            )
            lim_dx = sp.limit(
                (force_real_roots(f_expr).subs(x_real, punto + h) - force_real_roots(f_expr).subs(x_real, punto)) / h,
                h, 0, dir='+'
            )

            lim_sx = evaluate_inf_sign(lim_sx)
            lim_dx = evaluate_inf_sign(lim_dx)

            # Classificazione
            try:
                # determinazione di limiti reali e finiti
                is_real_left = getattr(lim_sx, 'is_real', False)
                is_real_right = getattr(lim_dx, 'is_real', False)
                is_finite_left = is_real_left and getattr(lim_sx, 'is_finite', False)
                is_finite_right = is_real_right and getattr(lim_dx, 'is_finite', False)

                if is_finite_left and is_finite_right:
                    # entrambi i limiti finiti
                    if lim_sx != lim_dx:
                        descrizione = r"\text{Punto angoloso}"
                    else:
                        descrizione = r"\text{Derivata continua (limiti uguali)}"
                elif lim_sx in (sp.oo, -sp.oo) or lim_dx in (sp.oo, -sp.oo):
                    # almeno un limite infinito
                    if lim_sx in (sp.oo, -sp.oo) and lim_dx in (sp.oo, -sp.oo):
                        # entrambi infiniti
                        if lim_sx == lim_dx:
                            descrizione = r"\text{Flesso a tangente verticale}"
                        else:
                            descrizione = r"\text{Cuspide}"
                    else:
                        descrizione = r"\text{Punto non classificabile}"
                else:
                    # limiti non esistenti o non numerici
                    descrizione = r"\text{Punto non classificabile}"
            except Exception:
                descrizione = r"\text{Punto non classificabile}"

            # Preparazione del testo LaTeX per un singolo punto
            line = (
                rf"&\lim_{{h \to 0^-}} {ratio_tex} = {latex_limit(lim_sx)} "
                rf"&& \lim_{{h \to 0^+}} {ratio_tex} = {latex_limit(lim_dx)} "
                rf"&&\rightarrow {descrizione}~\text{{in }} x_0 = {x0_str}"
            )
            lines.append(line)

        # Unico blocco align* con tutti i punti uno sotto l'altro
        steps[-1]["content"] = r"\begin{align*}" + r"\\ ".join(lines) + r"\end{align*}"

    # ------------------------------------------------------------
    # 14. Monotonia – intervalli di crescita/decrescita (f′ ≥ 0 / f′ ≤ 0)
    # -------------------------------------------------------------------
    # Inizializza inc/dec: servono poi allo step 16
    inc_set = sp.S.EmptySet
    dec_set = sp.S.EmptySet

    try:
        # Usa la derivata già calcolata (può essere Piecewise)
        fprime_pw = f_prime_pw

        print("[debug] Derivata f':", fprime_pw)

        # Lista (expr, cond) dei rami
        if isinstance(fprime_pw, sp.Piecewise):
            branches = fprime_pw.args
        else:
            branches = [(fprime_pw, True)]

        for branch_expr, branch_cond in branches:
            # 1) Condizione del ramo → Set
            if branch_cond is True:
                cond_set = S.Reals
            elif isinstance(branch_cond, sp.Contains):
                # Extract the set directly from Contains(x, <set>)
                cond_set = branch_cond.args[1]
            else:
                try:
                    cond_set = solveset(branch_cond, x_real, domain=S.Reals)
                    # If solveset returns ConditionSet, fall back to ℝ
                    if isinstance(cond_set, sp.ConditionSet):
                        cond_set = S.Reals
                except Exception:
                    cond_set = S.Reals
            if cond_set == sp.S.EmptySet:
                continue

            # 2) Semplifica sign() nel ramo
            branch_simpl = branch_expr.replace(sign, lambda arg: arg / Abs(arg))
           
            # --- sostituisci x → x_real per coerenza con le variabili usate nei solver
            branch_simpl = branch_simpl.xreplace({x: x_real})

            # 3) Dominio del ramo
            try:
                # usa safe_domain per gestire correttamente potenze razionali
                branch_dom = safe_domain(branch_expr.xreplace({x: x_real}), x_real)
                print("[debug] Dominio del ramo:", branch_dom)
            except Exception:
                try:
                    branch_dom = continuous_domain(branch_expr.xreplace({x: x_real}), x_real, S.Reals)
                except Exception:
                    branch_dom = S.Reals

            working_dom = branch_dom.intersect(cond_set).intersect(dom)

            if working_dom == sp.S.EmptySet:
                continue

            # 4) Intervalli inc / dec sul dominio del ramo usando sign-chart
            inc_part = sign_chart(branch_simpl, x_real, working_dom)
            print("[debug] Intervallo di crescita:", inc_part)
            dec_part = sign_chart(-branch_simpl, x_real, working_dom)
            print("[debug] Intervallo di decrescita:", dec_part)

            inc_set = inc_set.union(inc_part)
            dec_set = dec_set.union(dec_part)
        # ----------------------------------------------------------
        # Escludi dal risultato finale di monotonia tutti i punti in
        # nondiff_set (punti di non derivabilità e punti di raccordo)
        # ----------------------------------------------------------
        try:
            if 'nondiff_set' in locals() and isinstance(nondiff_set, sp.Set):
                inc_set = Complement(inc_set, nondiff_set)
                dec_set = Complement(dec_set, nondiff_set)
        except Exception:
            pass

        # ---- LaTeX ----
        def _latex_set(s):
            return (sp.latex(s)
                    .replace(r"\infty", r"+\infty")
                    .replace(r"-+\infty", r"-\infty"))

        inc_tex = _latex_set(inc_set) if inc_set != sp.S.EmptySet else r"\varnothing"
        monotonia_content = (
            r"\begin{aligned}"
            rf"f'(x)\ge 0, \quad \text{{per }} x\in {inc_tex}"
            r"\end{aligned}"
        )

    except Exception as e:
        print("[debug] Monotonia error:", e)
        monotonia_content = r"\text{La monotonia non è stata determinata.}"

    steps.append({"title": "Monotonia:", "content": monotonia_content})
    # ------------------------------------------------------------
    # 16. Massimi e Minimi relativi (da monotonia)
    # --- semplificazione globale di f' (serve per step 16) ---
    extrema_pts = []  # verrà popolato nel try; resta lista vuota se fallisce
    # ------------------------------------------------------------
    try:
        # Insieme di DECRESCENZA: f′ ≤ 0
        # dec_set è già stato calcolato allo step 15 (Monotonia) e contiene il
        # dominio di decrescenza; non lo ricalcoliamo qui per evitare loss di info.

        # --- helper per estrarre intervalli da un SymPy Set ---
        def _intervals(s):
            if s == sp.S.EmptySet:
                return []
            if isinstance(s, sp.Union):
                return list(s.args)
            return [s]

        inc_intervals = [(iv, "inc") for iv in _intervals(inc_set)]
        dec_intervals = [(iv, "dec") for iv in _intervals(dec_set)]

        # Combina e ordina per inizio intervallo
        intervals_all = inc_intervals + dec_intervals
        intervals_all.sort(key=lambda t: t[0].start if isinstance(t[0], sp.Interval) else -sp.oo)

        extrema_pts = []   # (x0, tipo)  tipo ∈ {"max", "min"}

        # Scansiona i cambi di monotonia:  inc→dec ⇒ max  |  dec→inc ⇒ min
        for (iv1_int, t1), (iv2_int, t2) in zip(intervals_all, intervals_all[1:]):
            boundary = None

            # Caso  …][…  (gli estremi dei due intervalli coincidono)
            if isinstance(iv1_int, sp.Interval) and isinstance(iv2_int, sp.Interval) \
               and iv1_int.end == iv2_int.start:
                boundary = iv1_int.end

            # Caso di sovrapposizione in un singolo punto
            elif iv1_int.intersect(iv2_int):
                inter = iv1_int.intersect(iv2_int)
                if isinstance(inter, sp.FiniteSet):
                    boundary = list(inter)[0]

            if boundary is not None and boundary.is_real:
                # Salta se il punto non appartiene al dominio della funzione
                if dom.contains(boundary) is not sp.S.true:
                    continue
                if t1 == "inc" and t2 == "dec":
                    extrema_pts.append((boundary, "max"))
                elif t1 == "dec" and t2 == "inc":
                    extrema_pts.append((boundary, "min"))

        # --- Calcolo del valore di f in ciascun punto di estremo relativo ---
        extrema_lines = []

        # --- raccogli info ---
        info = []   # list of dicts: {"x": p, "val": f_val, "nature": "max"/"min"}
        for p, nature in extrema_pts:
            # ignora eventuali punti non nel dominio (sicurezza aggiuntiva)
            if dom.contains(p) is not sp.S.true:
                continue
            try:
                f_val = sp.simplify(f_expr_real.subs(x_real, p))
                # Simplify (-1)**(odd rational) to -1
                f_val = f_val.replace(
                    lambda e: isinstance(e, sp.Pow)
                              and e.base == -1
                              and e.exp.is_Rational
                              and e.exp.q % 2 == 1,
                    lambda e: -1
                )
            except Exception:
                f_val = sp.nan
            info.append({"x": p, "val": f_val, "nature": nature})

        # Classificazione: "massimo relativo" o "minimo relativo"
        for d in info:
            if d["nature"] == "max":
                label = r"\text{massimo relativo}"
            else:  # "min"
                label = r"\text{minimo relativo}"

            extrema_lines.append(
                rf"x_0 = {sp.latex(d['x'])},\; f(x_0)={sp.latex(d['val'])}: \; {label}"
            )
        print("[debug] Estremi trovati:", extrema_lines)


        # --- Output LaTeX ---
        if extrema_lines:
            extremi_content = r"\begin{align*}" + r"\\ ".join(extrema_lines) + r"\end{align*}"
        else:
            extremi_content = r"\text{Nessun massimo o minimo individuato.}"

    except Exception as e:
        extremi_content = r"\text{Determinazione degli estremi non riuscita.}"

    steps.append({
        "title": "Massimi e Minimi:",
        "content": extremi_content
    })

    # ------------------------------------------------------------
    # 17. Dati per il grafico (frontend: function‑plot.js)
    # ------------------------------------------------------------
    import re
    def to_js_func(expr: sp.Expr) -> str:
        """
        Converte una espressione SymPy in una stringa compatibile con math.js.
        - sqrt(x) → sqrt(x)
        - Pow(..., p/q) → nthRoot(base**p, q)
        - Abs(...) → abs(...)
        - ** → ^
        """
        def _convert_pow(e):
            if isinstance(e, sp.Pow):
                base, exp = e.args
                if exp == sp.Rational(1, 2):
                    return sp.Function("sqrt")(base)
                if exp.is_Rational:
                    return sp.Function("nthRoot")(base**exp.p, sp.Integer(exp.q))
            return e

        expr = expr.replace(lambda e: isinstance(e, sp.Pow), _convert_pow)
        js_str = str(expr)
        # Sostituisci pi con il valore numerico per renderlo riconoscibile in JS
        js_str = js_str.replace("pi", str(sp.pi.evalf()))
        js_str = js_str.replace("Abs", "abs")
        js_str = js_str.replace("**", "^")
        return js_str

    plot_data = {
        # stringa della funzione per il plotter JS
        "func": to_js_func(f_expr),
        # intervalli continui del dominio: [[a,b], ...]  (None = ±∞)
        "domain": [],
        # punti notevoli: continuiamo a inviarli (front‑end libero di usarli)
        "segments": [],       # lasciato per retro‑compatibilità (vuoto)
        "zeros":    [],
        "extrema":  [],
        "asymptotes": []
    }


    # Costruisci gli intervalli continui del dominio per il front‑end
    dom_intervals = list(dom.args) if isinstance(dom, sp.Union) else [dom]
    for iv in dom_intervals:
        if isinstance(iv, sp.Interval):
            a = float(iv.start) if iv.start.is_finite else None
            b = float(iv.end)   if iv.end.is_finite   else None
            plot_data["domain"].append([a, b])

    # 17.b  —  punti notevoli già disponibili
    if isinstance(zeros_set, sp.FiniteSet):
        plot_data["zeros"] = [float(z) for z in zeros_set if z.is_real]

    # Costruisci la lista di estremi con valori reali
    extrema_list = []
    for p, nature in extrema_pts:
        print("[debug] Estremo:", p, nature)
        if getattr(p, "is_real", False):
            try:
                # Evaluate f on the real branch for fractional powers
                y_val = force_real_roots(f_expr).subs(x, p)
                y_num = sp.N(y_val)
                # Includi solo se è un reale finito
                if y_num.is_real:
                    extrema_list.append({
                        "x": float(p),
                        "y": float(y_num),
                        "nature": nature
                    })
            except Exception as e:
                print("Il punto ", p, "non è valutabile:", e)
                continue
    plot_data["extrema"] = extrema_list
    print("[debug] Estremi per il plot:", plot_data["extrema"])

    plot_data["asymptotes"] = (
        [
            {"type": "vertical", "x": float(p)}
            for p in punti_singolarita if getattr(p, "is_real", False)
        ]
        + oblique_list
        + [{"type": "horizontal", "y": y} for y in horizontal_vals]
    )

    return jsonify({"success": True, "steps": steps, "plot": plot_data})