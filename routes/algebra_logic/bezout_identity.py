from flask import Blueprint, request, jsonify

# Helper to split a long LaTeX expression at the last operator before the limit
def split_long(expr, max_len=130):
    """
    Splits a LaTeX expression at the last operator before max_len.
    Returns a list of lines.
    """
    if len(expr) <= max_len:
        return [expr]
    ops = ['+', '-', r'\cdot', r'\times', '=', '&=']
    idx = 0
    splits = []
    current = expr
    while len(current) > max_len:
        # Find last operator before max_len
        cut = -1
        for op in ops:
            search_from = 0
            while True:
                pos = current.find(op, search_from)
                if pos == -1 or pos > max_len:
                    break
                cut = max(cut, pos + len(op) - 1)
                search_from = pos + 1
        if cut == -1 or cut < 10:  # don't split too early
            break
        splits.append(current[:cut+1].rstrip())
        current = current[cut+1:].lstrip()
    splits.append(current)
    return splits

identita_bezout_bp = Blueprint('identita_bezout_bp', __name__)

@identita_bezout_bp.route('/api/bezout-identity', methods=['POST'])
@identita_bezout_bp.route('/api/identita-bezout', methods=['POST'])
def identita_bezout_route():
    """
    Calcola l'identità di Bézout per interi a, b.
    """
    data = request.get_json() or {}
    # Validazione parametri
    try:
        a = int(data.get('a'))
        b = int(data.get('b'))
    except (TypeError, ValueError):
        return jsonify(success=False, error="Parametri mancanti o non interi 'a' e 'b'."), 400

    steps = []
    # Step 0: espressione iniziale
    expr0 = fr"{a}x + {b}y = \gcd({a},{b})"
    steps.append({"title": "Espressione iniziale:", "content": expr0})

    # Step 1: iterazioni algoritmo di Euclide esteso with tracciamento dei coefficienti di Bézout
    iterations = []
    r0, r1 = a, b
    # Bézout coefficients initialization
    s0, s1 = 1, 0
    t0, t1 = 0, 1
    while r1 != 0:
        q = r0 // r1
        r2 = r0 - q * r1
        # record division step
        iterations.append(fr"{r0} &= {q} \cdot {r1} + {r2}")
        # update Bézout coefficients
        s2 = s0 - q * s1
        t2 = t0 - q * t1
        # rotate variables
        r0, r1 = r1, r2
        s0, s1 = s1, s2
        t0, t1 = t1, t2
    gcd = r0
    # final Bézout coefficients
    old_s, old_t = s0, t0
    # Split each line for readability
    align_lines = []
    for it in iterations:
        align_lines.extend(split_long(it))
    align_body = " \\\\\n".join(align_lines)
    align_env = r"\begin{align*}" + "\n" + align_body + "\n" + r"\end{align*}"
    steps.append({"title": "Iterazioni algoritmo di Euclide esteso:", "content": align_env})

    # Step 2: Espressioni a ritroso:
    r0, r1 = a, b
    back_steps = []
    while r1 != 0:
        q = r0 // r1
        r2 = r0 - q * r1
        if r2 == 0:
            break
        back_steps.append(fr"{r2} &= {r0} - {q}\cdot {r1}")
        r0, r1 = r1, r2
    align_lines2 = []
    for bs in back_steps:
        align_lines2.extend(split_long(bs))
    align_body2 = " \\\\\n".join(align_lines2)
    align_env2 = r"\begin{align*}" + "\n" + align_body2 + "\n" + r"\end{align*}"
    steps.append({"title": f"Abbiamo che $\\gcd({a},{b}) = {gcd}$. Calcoliamo ora le espressioni a ritroso:", "content": align_env2})

    # Step 3: Sostituzioni successive
    # build mappa di substitution per r_i -> rhs
    mapping = []
    for line in back_steps[:-1]:
        parts = line.split("&=")
        lhs = parts[0].strip()
        rhs = parts[1].strip()
        mapping.append((lhs, rhs))
    # inizia with l'ultima equation (gcd)
    last_line = back_steps[-1]
    lhs, rhs = [p.strip() for p in last_line.split("&=")]
    # build l'elenco di sostituzioni progressive, with wrap che ripete sotto '='
    expr = rhs
    subs_lines = []
    # prima substitution: linea base
    parts = split_long(expr)
    if parts:
        subs_lines.append(f"{lhs} &= {parts[0]}")
        for cont in parts[1:]:
            subs_lines.append(f"& {cont}")
    # Apply the remaining substitutions backward
    for sub_lhs, sub_rhs in reversed(mapping):
        if sub_lhs in expr:
            expr = expr.replace(sub_lhs, f"({sub_rhs})")
            parts = split_long(expr)
            if parts:
                subs_lines.append(f"{lhs} &= {parts[0]}")
                for cont in parts[1:]:
                    subs_lines.append(f"& {cont}")
    # unisci le linee in un align centralizzato
    subs_body = " \\\\\n".join(subs_lines)
    subs_env = r"\begin{align*}" + "\n" + subs_body + "\n" + r"\end{align*}"
    steps.append({"title": "Sostituzioni successive:", "content": subs_env})

    # Step 5: Risultato finale
    s_str = f"({old_s})" if old_s < 0 else str(old_s)
    t_str = f"({old_t})" if old_t < 0 else str(old_t)
    identity_expr = fr"{a} \cdot {s_str} + {b} \cdot {t_str} = {gcd}"
    implication = fr"x = {old_s}, y = {old_t} \implies {identity_expr}"
    steps.append({"title": "Risultato finale:", "content": implication})

    return jsonify(success=True, latex=steps, result={"gcd": gcd, "coeffs": [old_s, old_t]})
