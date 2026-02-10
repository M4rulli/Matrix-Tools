from __future__ import annotations
from flask import Blueprint, request, jsonify
import sympy as sp, re
from typing import List, Tuple

cc_bp = Blueprint("condizioni_complementari_bp", __name__)

# Robust parsing is required because MathLive and other math editors emit
# various LaTeX forms for fractions (\frac{4}{3}, \frac43, etc.), inequalities,
# and variable subscripts. We normalize these for reliable parsing.

# ---------- parsing helpers --------------------------------------------- #
_var = re.compile(r"(x|y)_(\d+)")                    # variabili x_i o y_i

def parse_tex(expr:str)->sp.Expr:
    """‘3x_1-\\frac23x_2’ → sympy expression.
    Handles MathLive and compact LaTeX fraction/inequality/subscript variants.
    """
    _norm = expr.replace(" ", "")
    # Normalize \cdot and cdot to *
    _norm = _norm.replace(r"\cdot", "*")
    _norm = _norm.replace("cdot", "*")
    # Normalize x_{12} → x_12, y_{3} → y_3
    _norm = re.sub(r'([xy])_\{(\d+)\}', r'\1_\2', _norm)
    # Normalize inequalities BEFORE fraction rewriting
    _norm = _norm.replace(r"\leq", "<=").replace(r"\le", "<=")
    _norm = _norm.replace(r"\geq", ">=").replace(r"\ge", ">=")
    # Robust fraction rewriting: \frac{num}{den} and \frac43, \frac-12, etc.
    def _frac_curly(m):
        return f"({m.group(1)})/({m.group(2)})"
    def _frac_plain(m):
        # handles \frac43, \frac-1 2, etc.
        return f"({m.group(1)})/({m.group(2)})"
    _norm = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", _frac_curly, _norm)
    _norm = re.sub(r"\\frac([+-]?\d+)([+-]?\d+)", _frac_plain, _norm)
    _norm = re.sub(r"\\frac\s*([+-]?\d+)\s*([+-]?\d+)", _frac_plain, _norm)
    # Insert implicit multiplications: 3x_1 → 3*x_1, (2)x_3 → (2)*x_3
    _norm = re.sub(r'(?<=[0-9\)])([xy]_\d+)', r'*\1', _norm)
    # Build names dict from all variables found (use all matches)
    names = {f"{a}_{b}": sp.symbols(f"{a}_{b}") for a, b in _var.findall(_norm)}
    return sp.sympify(_norm, locals=names)

def latex_align(lines:List[str])->str:
    return r"\begin{aligned}" + r"\\ ".join(lines) + r"\end{aligned}"

# ---------- helper: strictly-positive check ---------------------------- #

def _is_strict_positive(val: sp.Expr, tol: float = 1e-12) -> bool:
    """Return True if numerical value is > 0 with a tolerance."""
    try:
        return float(val.evalf()) > tol
    except Exception:
        # symbolic, treat as not-positive
        return False

# ---------- helper: nonzero check (for complementarity, sign-agnostic) --------- #
def _is_nonzero(val: sp.Expr, tol: float = 1e-12) -> bool:
    """Return True if |val| > tol (used for complementarity, sign‑agnostic)."""
    try:
        return abs(float(val.evalf())) > tol
    except Exception:
        # symbolic or unevaluated: treat as non‑zero to enforce equality conservatively
        return True

# ---------- endpoint ----------------------------------------------------- #
@cc_bp.route("/api/condizioni-complementari", methods=["POST"])
def condizioni_complementari_route():
    data     = request.get_json(silent=True) or {}
    modo     = data.get("modo", "primale")          # 'primale' | 'duale'
    tipo     = data.get("tipo", "max")              # max / min del problema ORIGINALE
    obj_raw  = data.get("obj","")
    constr   = data.get("constraints",[])           # [[lhs,rhs,sign], ...]
    nonneg   = data.get("nonneg","")                # "x_1,x_3,x_4 >= 0"
    sol_raw  = data.get("sol","")                   # "x=(1,2,0)" oppure "y=(...)"

    def _validate_latex_var(token: str) -> bool:
        """Return True if token is exactly x_k or y_k with k≥1."""
        return bool(re.fullmatch(r"[xy]_[1-9]\d*", token))

    # ---------- basic syntactic sanity checks ----------
    # 1) constraints must contain a valid comparison symbol
    for lhs_txt, rhs_txt, sg in constr:
        if sg not in ("<=", ">=", "="):
            return jsonify(success=False,
                           error=f"Segno vincolo non valido: {sg}"), 400
        if not lhs_txt.strip() or not rhs_txt.strip():
            return jsonify(success=False,
                           error="Ogni vincolo deve avere LHS e RHS non vuoti."), 400

    # 2) non‑negativity string must be a comma‑separated list like x_1>=0 or x_2 \\ge 0
    if nonneg.strip():
        # split on commas, skip empty fragments
        pieces = [p.strip() for p in nonneg.split(',') if p.strip()]
        for piece in pieces:
            # ignore stray fragments without any variable
            if not re.search(r'[xy]_\d+', piece):
                continue
            # allow operators >=, <=, or LaTeX forms \ge, \le
            m = re.fullmatch(r"([xy]_[1-9]\d*)\s*(?:>=|<=|\\ge|\\le)\s*0", piece)
            if not m or not _validate_latex_var(m.group(1)):
                return jsonify(success=False,
                               error=f"Condizione di segno non riconosciuta: '{piece}'"), 400

    # 3) solution string must be like 'x=(a,b,...)' or 'y=(...)'
    sol_match = re.fullmatch(r"\s*([xy])\s*=\s*\(([^)]+)\)\s*", sol_raw)
    if not sol_match:
        return jsonify(success=False,
                       error="Formato soluzione proposto non riconosciuto."), 400

    # --- numero di vincoli ---------------------------------------------
    # se il client non passa 'n', ricavalo dalla lunghezza di constraints
    n_raw = data.get("n")
    try:
        n = int(n_raw) if n_raw is not None else len(constr)
    except (TypeError, ValueError):
        return jsonify(success=False,
                       error="Parametro 'n' non valido."), 400

    # aggiorna 'n' se non combacia con le constraints dichiarate
    if len(constr) != n:
        n = len(constr)

    # limiti attuali: per ora accettiamo da 1 a 6 vincoli (estendibile)
    if n < 1 or n > 6:
        return jsonify(success=False,
                       error="Numero di vincoli fuori intervallo (1–6)."), 400

    if modo not in ("primale","duale"):
        return jsonify(success=False,error="Flag 'modo' non valido"),400

    steps=[]

    # ---------------------------------------------------- #
    # STEP 0 : problema inserito (vincoli + segni + nonneg)
    # ---------------------------------------------------- #
    try:
        obj_expr = parse_tex(obj_raw)
        lhs = [parse_tex(c[0]) for c in constr]
        rhs = [parse_tex(c[1]) for c in constr]
        signs_list = [c[2] for c in constr]
    except Exception as e:
        return jsonify(success=False,error=f"Errore parsing: {e}"),400

    probl_lines=[(r"\text{%s } z ="%("min" if tipo=="min" else "max"))+sp.latex(obj_expr)+":"]
    for (L,R,S) in constr:
        sym=S.replace("<=",r"\le").replace(">=",r"\ge").replace("=",r"=")
        probl_lines.append(f"{L} {sym} {R}")
    probl_lines.append(nonneg.replace(">=",r"\ge").replace("<=",r"\le"))
    steps.append({"title":"Problema inserito:","content":latex_align(probl_lines)})

    # ---------- STEP 1 : costruzione del problema duale ------------------ #
    # Ricaviamo m (= n vincoli) e p (= # variabili originali)            
    # ricava il massimo indice delle variabili x_j presenti nella funzione obiettivo
    vars_in_obj = _var.findall(obj_raw)           # list of tuples like ('x', '3')
    if vars_in_obj:
        p = max(int(idx) for _, idx in vars_in_obj)
    else:
        p = 0
    m = n                                                 # numero di vincoli
    # coeff. matrice A e vettori c, b
    c_vec = sp.Matrix([obj_expr.coeff(sp.symbols(f"x_{j+1}")) for j in range(p)])
    A_mat = sp.zeros(m, p)
    for i, lhs_expr in enumerate(lhs):
        for j in range(p):
            A_mat[i, j] = lhs_expr.coeff(sp.symbols(f"x_{j+1}"))
    b_vec = sp.Matrix(rhs)

    # Impostiamo il dual nella forma esplicita
    y_syms = [sp.symbols(f"y_{i+1}") for i in range(m)]
    dual_obj_type = "min" if tipo=="max" else "max"
    dual_lines = [fr"\text{{{dual_obj_type}}}\; z_D = " + sp.latex((b_vec.T * sp.Matrix(y_syms))[0]) + ":"]
    # ----  direzione del vincolo duale per ciascuna variabile primale ----
    #  • variabile primale  ≥ 0  →  (min) “≤”, (max) “≥”
    #  • variabile primale  ≤ 0  →  (min) “≥”, (max) “≤”
    #  • variabile libera      →  “=”
    #
    # costruiamo una mappa  var_sign[x_j] -> {'>=', '<=', 'free'}
    var_sign = {f"x_{j+1}": "free" for j in range(p)}     # di default libera
    for piece in nonneg.split(","):
        piece = piece.strip()
        if not piece:
            continue
        m_match = re.match(r"(x_\d+)\s*([<>]=)\s*0", piece.replace(r"\ge",">=").replace(r"\le","<="))
        if m_match:
            v, sgn = m_match.groups()
            var_sign[v] = ">=" if sgn == ">=" else "<="

    dual_constraints: List[Tuple[sp.Expr, str, sp.Expr]] = []  # (lhs, latex_op, rhs)
    for j in range(p):
        var_name = f"x_{j+1}"
        lhs_expr = sum(A_mat[i, j] * y_syms[i] for i in range(m))

        sign_token = var_sign.get(var_name, "free")
        # Determine LaTeX operator
        if sign_token == "free":
            latex_op = "="
        else:
            # corretto mapping dei segni
            if tipo == "max":          # primale di MAX
                latex_op = r"\ge" if sign_token == ">=" else r"\le"
            else:                      # primale di MIN
                latex_op = r"\le" if sign_token == ">=" else r"\ge"
        # Store latex_op for logic and display
        dual_constraints.append((lhs_expr, latex_op, c_vec[j]))
        dual_lines.append(f"{sp.latex(lhs_expr)} {latex_op} {sp.latex(c_vec[j])}")

    # segni sulle variabili duali
    dual_var_sign: List[Tuple[sp.Symbol, str]] = []  # (y_i, sign_string)
    for sign, y in zip(signs_list, y_syms):
        if sign == "=":
            dual_var_sign.append((y, "free"))
        else:
            if tipo == "max":          # primale di MAX
                dual_var_sign.append((y, ">=0" if sign == "<=" else "<=0"))
            else:                      # primale di MIN
                dual_var_sign.append((y, "<=0" if sign == "<=" else ">=0"))
    # segno delle variabili duali in base sia al tipo di vincolo del primale che al tipo del problema
    dual_nonneg = []
    for sign, y in zip(signs_list, y_syms):
        if sign == "=":
            dual_nonneg.append(fr"{sp.latex(y)} \in \mathbb{{R}}")
        else:
            # regola generale:
            #  •  problema di MIN:  "<=" → y ≤ 0   ,  ">=" → y ≥ 0
            #  •  problema di MAX:  "<=" → y ≥ 0   ,  ">=" → y ≤ 0
            if tipo == "min":
                sym = r"\le 0" if sign == "<=" else r"\ge 0"
            else:  # tipo == "max"
                sym = r"\ge 0" if sign == "<=" else r"\le 0"
            dual_nonneg.append(fr"{sp.latex(y)} {sym}")
    dual_lines.append(",\; ".join(dual_nonneg))

    steps.append({"title": "Problema duale:", "content": latex_align(dual_lines)})

    # ---------------------------------------------------- #
    # STEP 2 : parsing della tupla proposta
    # ---------------------------------------------------- #
    # es. "x = (1,2,0)"  ->  vett = [1,2,0]
    # Pattern for signed rational or decimal number tokens (e.g. -2/3, 1.5, 4)
    NUM_TOKEN_RE = re.compile(r"[+-]?(?:\d+\/\d+|\d*\.\d+|\d+)")
    try:
        if not sol_raw:
            return jsonify(success=False,
                           error="Soluzione (campo 'sol') mancante."), 400
        var_letter = sol_raw.strip()[0]             # 'x' o 'y'
        tuple_body = sol_raw.split("=",1)[1]
        body_norm = tuple_body.replace(' ','')
        # normalize LaTeX \frac{a}{b} and \fracab -> a/b for solution strings too
        body_norm = re.sub(r"\\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", body_norm)
        body_norm = re.sub(r"\\\frac\s*([+-]?\d+)\s*([+-]?\d+)", r"\1/\2", body_norm)
        raw_nums = NUM_TOKEN_RE.findall(body_norm)
        expected_len = p if modo == "primale" else m
        if len(raw_nums) != expected_len:
            return jsonify(success=False,
                           error=f"Soluzione proposta di lunghezza {len(raw_nums)}, "
                                 f"ma attesa {expected_len}."), 400
        try:
            # parse each token (including fractions like "3/2") into a Sympy expression
            sol_vec = sp.Matrix([parse_tex(t) for t in raw_nums])
        except Exception:
            return jsonify(success=False, error="Valori numerici nella soluzione non validi."), 400
    except Exception:
        return jsonify(success=False,error="Formato soluzione proposto non riconosciuto"),400

    steps.append({
        "title": f"Soluzione proposta:",
        "content": fr"{var_letter} = {sp.latex(sol_vec.T)}"
    })

    # ---------------------------------------------------- #
    # STEP 3 : ammissibilità (vincoli + segno)
    # ---------------------------------------------------- #
    feas_checks = []
    feas = True

    if modo == "primale":
        # controllo sui vincoli PRIMALI
        subs_primale = {sp.symbols(f"x_{i+1}"): sol_vec[i] for i in range(len(sol_vec))}
        subs = subs_primale
        for (L, R, S) in constr:
            lhs_val = parse_tex(L).subs(subs)
            rhs_val = parse_tex(R).subs(subs)
            # choose comparison based on original constraint sign
            if S == "<=":
                ok = float(lhs_val.evalf()) <= float(rhs_val.evalf())
                sym_tex = r"\le"
            elif S == ">=":
                ok = float(lhs_val.evalf()) >= float(rhs_val.evalf())
                sym_tex = r"\ge"
            elif S == "=":
                ok = float(abs((lhs_val - rhs_val).evalf())) < 1e-12
                sym_tex = "="
            else:
                return jsonify(success=False, error=f"Segno vincolo non valido: {S}"), 400
            feas_checks.append(fr"{sp.latex(lhs_val)}\;{sym_tex}\;{sp.latex(rhs_val)}")
            feas &= bool(ok)
        # segni variabili primali (da nonneg string)
        for piece in nonneg.split(","):
            piece = piece.strip()
            if not piece:
                continue
            m_match = re.match(r"(x_\d+)\s*([<>]=)\s*0", piece.replace(r"\\ge",">=").replace(r"\\le","<="))
            if m_match:
                v, sgn = m_match.groups()
                val = subs_primale.get(sp.symbols(v), None)
                if val is None:
                    continue
                feas &= bool(val >= 0) if sgn == ">=" else bool(val <= 0)
    else:  # modo == "duale"
        # controllo sui vincoli DUALI
        subs_duale = {y_syms[i]: sol_vec[i] for i in range(len(sol_vec))}
        subs = subs_duale
        for lhs_dual, latex_op, rhs_dual in dual_constraints:
            lhs_val = lhs_dual.subs(subs_duale)
            rhs_val = rhs_dual
            sym_tex = latex_op
            if latex_op == r"\le":
                ok = float(lhs_val.evalf()) <= float(rhs_val.evalf())
            elif latex_op == r"\ge":
                ok = float(lhs_val.evalf()) >= float(rhs_val.evalf())
            else:  # equality
                ok = abs((lhs_val - rhs_val).evalf()) < 1e-12
            feas_checks.append(fr"{sp.latex(lhs_val)}\;{sym_tex}\;{sp.latex(rhs_val)}")
            feas &= bool(ok)
        # segni variabili duali
        for y, sign in dual_var_sign:
            val = subs_duale[y]
            if sign == ">=0":
                feas &= bool(val >= 0)
            elif sign == "<=0":
                feas &= bool(val <= 0)
            # 'free' no restriction

    steps.append({"title": "Verifica ammissibilità:",
                  "content": latex_align(feas_checks) +
                             (r"\quad\text{Ammissibile}" if feas else r"\quad\text{NON ammissibile}")})

    if not feas:
        # --- conclusione per soluzione non ammissibile (fase 1) --------------
        steps.append({
            "title": "Conclusione:",
            "content": "\\text{Soluzione } %s\\text{, non ammissibile }\\Rightarrow\\text{ non ottima.}\\\\[2em]" % var_letter
        })
        return jsonify(success=True, steps=steps, latex=steps)

    # ---------------------------------------------------- #
    # STEP 4 : condizioni di complementarità → sistema lineare
    # ---------------------------------------------------- #
    system_eqs: List[str] = []

    if modo == "duale":
        #  Si costruisce un sistema in x utilizzando la regola:
        #     se y_i ≠ 0 ⇒ il vincolo i‑esimo del primale deve valere in uguaglianza
        #     se il vincolo duale j‑esimo NON è in uguaglianza ⇒ x_j = 0
        #  (gli indici i, j sono rispettivamente quelli dei vincoli e delle variabili)
        #
        # uguaglianze forzate dai componenti y non‑zero
        for i, y_val in enumerate(sol_vec):
            if _is_nonzero(y_val):
                system_eqs.append(
                    fr"{sp.latex(lhs[i])} = {sp.latex(rhs[i])}"
                )

        # eventuali x_j = 0 (vincoli duali *non* in uguaglianza)
        subs_duale = {y_syms[i]: sol_vec[i] for i in range(len(sol_vec))}
        for j, (lhs_dual, op, rhs_dual) in enumerate(dual_constraints):
            lhs_val = lhs_dual.subs(subs_duale)
            is_eq   = abs(float((lhs_val - rhs_dual).evalf())) < 1e-12
            if not is_eq:
                system_eqs.append(fr"x_{{{j+1}}} = 0")
    else:
        # modo == primale  →  sistema nei moltiplicatori duali y
        # se x_j > 0 ⇒ vincolo j del duale in uguaglianza
        for j, x_val in enumerate(sol_vec):
            if _is_strict_positive(x_val):
                lhs_dual, _, rhs_dual = dual_constraints[j]
                system_eqs.append(fr"{sp.latex(lhs_dual)} = {sp.latex(rhs_dual)}")

        # se il vincolo primale i‑esimo NON è in uguaglianza ⇒ y_i = 0
        subs_primale = {sp.symbols(f"x_{k+1}"): sol_vec[k] for k in range(len(sol_vec))}
        for i, (L, R, S) in enumerate(constr):
            lhs_val = parse_tex(L).subs(subs_primale)
            rhs_val = parse_tex(R).subs(subs_primale)
            is_eq   = abs(float((lhs_val - rhs_val).evalf())) < 1e-12
            if not is_eq:
                system_eqs.append(fr"{sp.latex(y_syms[i])} = 0")

    if not system_eqs:
        system_eqs.append(r"\text{(nessuna equazione imposta)}")

    # ---- render system with \begin{cases} ----
    cases_tex = r"\begin{cases} " + r"\\ ".join(system_eqs) + r"\end{cases}"

    # ---- build SymPy equations and unknown list ----
    eq_list   = []
    unknowns  = []
    for rel in system_eqs:
        # split on '='
        if "=" in rel:
            lhs_tex, rhs_tex = rel.split("=", 1)
            lhs_tex, rhs_tex = lhs_tex.strip(), rhs_tex.strip()
            if not lhs_tex or not rhs_tex:
                continue  # guard against empty LHS or RHS
            lhs_expr = parse_tex(lhs_tex)
            rhs_expr = parse_tex(rhs_tex)
            eq_list.append(sp.Eq(lhs_expr, rhs_expr))
            # collect symbols appearing only on LHS
            unknowns.extend([s for s in lhs_expr.free_symbols if (s.name.startswith("x_") and modo=="duale")
                                                                or (s.name.startswith("y_") and modo=="primale")])
    unknowns = list(dict.fromkeys(unknowns))  # unique & order‑preserving

    sol_system = None
    if unknowns and eq_list:
        try:
            sol_list = sp.solve(eq_list, unknowns, dict=True)
            if sol_list:
                sol_system = sol_list[0]   # take the first solution
        except Exception:
            sol_system = None

    # STEP 4 – sistema e soluzione
    steps.append({
        "title": "Sistema dalle condizioni di complementarità:",
        "content": cases_tex
    })

    if sol_system is not None:
        # pretty print solution tuple
        sol_lines = []
        if modo == "duale":
            # build explicit list x_1,x_2,…,x_p (no ellipsis)
            tuple_vars = ", ".join(f"x_{{{j+1}}}" for j in range(p))
            tuple_vals = ", ".join(
                sp.latex(sol_system.get(sp.symbols(f"x_{j+1}"), 0)) for j in range(p)
            )
            sol_lines.append(fr"\left({tuple_vars}\right) = \left({tuple_vals}\right)")
        else:  # primale ⇒ soluzione in y
            tuple_vars = ", ".join(f"y_{{{i+1}}}" for i in range(m))
            tuple_vals = ", ".join(
                sp.latex(sol_system.get(sp.symbols(f"y_{i+1}"), 0)) for i in range(m)
            )
            sol_lines.append(fr"\left({tuple_vars}\right) = \left({tuple_vals}\right)")
        steps.append({
            "title": "Soluzione del sistema:",
            "content": latex_align(sol_lines)
        })

        # ---------- verifica ammissibilità della soluzione trovata ----------
        feas2 = True
        feas_checks2 = []
        if modo == "duale":
            # verifica sul primale con x‑solution
            subs_x = {sp.symbols(f"x_{j+1}"): sol_system.get(sp.symbols(f"x_{j+1}"), 0)
                      for j in range(p)}
            for (L,R,S) in constr:
                lhs_val = parse_tex(L).subs(subs_x)
                rhs_val = parse_tex(R).subs(subs_x)
                if S == "<=":
                    feas2 &= float(lhs_val.evalf()) <= float(rhs_val.evalf())
                    sym_tex = r"\le"
                elif S == ">=":
                    feas2 &= float(lhs_val.evalf()) >= float(rhs_val.evalf())
                    sym_tex = r"\ge"
                else:
                    feas2 &= abs(float((lhs_val-rhs_val).evalf())) < 1e-12
                    sym_tex = "="
                feas_checks2.append(fr"{sp.latex(lhs_val)}\;{sym_tex}\;{sp.latex(rhs_val)}")
        else:
            # verifica sul duale con y‑solution
            subs_y = {y_syms[i]: sol_system.get(y_syms[i], 0) for i in range(m)}
            for lhs_dual, op, rhs_dual in dual_constraints:
                lhs_val = lhs_dual.subs(subs_y)
                if op == r"\le":
                    feas2 &= float(lhs_val.evalf()) <= float(rhs_dual.evalf())
                elif op == r"\ge":
                    feas2 &= float(lhs_val.evalf()) >= float(rhs_dual.evalf())
                else:
                    feas2 &= abs(float((lhs_val-rhs_dual).evalf())) < 1e-12
                feas_checks2.append(fr"{sp.latex(lhs_val)}\;{op}\;{sp.latex(rhs_dual)}")

        steps.append({
            "title": "Verifica finale della soluzione:",
            "content": latex_align(feas_checks2) +
                       (r"\quad\text{Soluzione ammissibile. Ottimalità per }P\text{ e }D."
                        if feas2 else r"\quad\text{Soluzione NON ammissibile.}")
        })

        if not feas2:
            steps.append({
                "title": "Conclusione:",
                "content": "\\text{Soluzione } %s\\text{, non ammissibile }\\Rightarrow\\text{ non ottima.}\\\\[2em]" % var_letter
            })
            return jsonify(success=True, steps=steps, latex=steps)

    # conclusione
    if modo == "duale":
        conclusion_text = (
            r"\text{Ammissibile ⇒ }y\text{ ottima per }D,\;x\text{ ottima per }P."
        )
    else:
        conclusion_text = (
            r"\text{Ammissibile ⇒ }x\text{ ottima per }P,\;y\text{ ottima per }D."
        )
    steps.append({"title": "Conclusione:", "content": conclusion_text})

    return jsonify(success=True, steps=steps, latex=steps)