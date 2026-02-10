from flask import Blueprint, request, jsonify
import sympy as sp
import re

spectral_bp = Blueprint("decomposizione_spettrale", __name__)

def normalize_frac_syntax(expr: str) -> str:
    """
    Converts LaTeX-style \frac{a}{b} or \fracab into (a/b)
    """
    # First fix invalid \frac12 → \frac{1}{2}
    expr = re.sub(r'\\frac(\d)(\d)', r'\\frac{\1}{\2}', expr)
    # Then convert \frac{a}{b} into (a/b)
    expr = re.sub(r'\\frac\s*{([^{}]+)}\s*{([^{}]+)}', r'(\1/\2)', expr)
    return expr

def convert_sqrt_syntax(expr_str):
    """
    Converts all occurrences of sqrt(...) into (...**(1/2)).
    Example: "2sqrt(5/6)" → "2*(5/6)**(1/2)"
    """
    expr_str = re.sub(r'(\d)(?=sqrt)', r'\1*', expr_str)
    pattern = r'sqrt\s*\(([^()]+)\)'
    while re.search(pattern, expr_str):
        expr_str = re.sub(pattern, r'(\1)**(1/2)', expr_str)
    return expr_str

@spectral_bp.route("/api/decomposizione-spettrale", methods=["POST"])
def compute():
    s = sp.symbols('s')
    t = sp.symbols('t')

    raw_matrix = request.json["matrix"]
    converted = [[convert_sqrt_syntax(normalize_frac_syntax(e)) for e in row] for row in raw_matrix]
    parsed    = [[sp.sympify(e) for e in row] for row in converted]
    A         = sp.Matrix(parsed)
    n         = A.shape[0]

    # ---------- Passi 1‑4 : polinomio caratteristico ----------
    char_poly = A.charpoly(s).as_expr()
    factored  = sp.factor(char_poly)
    eigvals   = A.eigenvals()
    eig_list  = [f"\\lambda_{{{i+1}}} = {sp.latex(val)}"
                 for i, val in enumerate(eigvals.keys())]
    inv_P     = sp.apart(1 / char_poly, s)

    # ---------- Passo 5 : proiettori ----------
    inv_terms = inv_P.as_ordered_terms()
    blocks = {}
    for term in inv_terms:
        root = sp.solve(sp.Eq(sp.denom(term), 0), s)[0]
        blocks.setdefault(root, []).append(term)

    latex_fi, proj_lines, proj_expr_list = [], [], []
    I = sp.eye(n)
    for idx, (lam, terms) in enumerate(blocks.items(), 1):
        fi_expr = sp.simplify(sp.Add(*terms) * char_poly)
        denom   = fi_expr.subs(s, lam)
        ei_expr = sp.simplify(fi_expr / denom)
        latex_fi.append(rf"f_{idx}(s)=\dfrac{{{sp.latex(fi_expr)}}}{{P(s)}}")

        # e_i(A)
        ei_poly  = sp.Poly(sp.expand(ei_expr), s)
        coeffs   = list(reversed(ei_poly.all_coeffs()))
        Ei       = sp.zeros(n)
        latex_terms = []
        for k, coeff in enumerate(coeffs):
            coeff = sp.nsimplify(coeff, rational=True)
            if coeff == 0:
                continue
            sign_tex   = "-" if coeff < 0 else ""
            abs_coeff  = abs(coeff)
            coeff_tex  = sp.latex(abs_coeff)
            factor_tex = "I" if k == 0 else "A" if k == 1 else f"A^{k}"
            term_mat   = I if k == 0 else A if k == 1 else A**k
            Ei += coeff * term_mat
            if abs_coeff == 1:
                latex_terms.append(f"{sign_tex}{factor_tex}")
            else:
                latex_terms.append(f"{sign_tex}\\left({coeff_tex}\\right){factor_tex}")
        proj_expr_list.append(Ei)
        proj_lines.append(
            rf"E_{idx} = e_{idx}(A) = "
            + " + ".join(latex_terms).replace("+ -", "- ")
            + rf" = {sp.latex(Ei)}"
        )

    # ---------- Passi 6‑7 : D e N ----------
    # somma dei blocchi con partenza da matrice zero per evitare TypeError
    D_matrix = sum(
        (lam * proj_expr_list[i] for i, (lam, _) in enumerate(blocks.items())),
        sp.zeros(n)
    )
    N_matrix = A - D_matrix
    latex_D = (rf"D = " +
               " + ".join([f"{sp.latex(lam)} E_{i+1}" for i, (lam, _) in enumerate(blocks.items())])
               + rf" = {sp.latex(D_matrix)}")
    latex_N = rf"N = A - D = {sp.latex(N_matrix)}"

    # ---------- Passi 8‑11 : e^{Dt}, D^{t}, e^{At}, A^{t} ----------
    # indice di nilpotenza
    k = 1
    while not (N_matrix**k).equals(sp.zeros(*N_matrix.shape)):
        k += 1

    lambdas = [((Ei*D_matrix).trace()/Ei.trace()).simplify() for Ei in proj_expr_list]

    exp_Dt, Dt = sp.zeros(n), sp.zeros(n)
    latex_eDt_parts, latex_Dt_parts = [], []
    for i, lam in enumerate(lambdas, 1):
        Ei = proj_expr_list[i-1]
        exp_Dt += sp.exp(lam*t, evaluate=False) * Ei
        Dt     += sp.Pow(lam, t, evaluate=False) * Ei
        latex_eDt_parts.append(rf"e^{{{sp.latex(lam)}t}}E_{{{i}}}")
        latex_Dt_parts .append(rf"{sp.latex(lam)}^t E_{{{i}}}")

    exp_At = sum(((t**i/sp.factorial(i))*exp_Dt*(N_matrix**i) for i in range(k)), start=sp.zeros(n))
    At_mat = sum((sp.binomial(t,i)*Dt*(N_matrix**i) for i in range(k)), start=sp.zeros(n))

    # ---------- Assemble steps ----------
    steps = [
        {"title": "Matrice inserita:", "content": rf"A = {sp.latex(A)}"},
        {"title": "Polinomio caratteristico $P(s)$:", "content": "P(s) = " + sp.latex(char_poly)},
        {"title": "Fattorizzazione di $P(s)$:", "content": "P(s) = " + sp.latex(factored)},
        {"title": "Autovalori:", "content": ",\\ ".join(eig_list)},
        {"title": r"Decomposizione $\frac{1}{P(s)}$:", "content": rf"\dfrac{{1}}{{P(s)}} = {sp.latex(inv_P)}"},
        {"title": "Proiettori spettrali:", "content": r"\begin{align}" + " \\\\ ".join(proj_lines) + r"\end{align}"},
        {"title": "Matrice diagonalizzabile $D$:", "content": latex_D},
        {"title": "Parte nilpotente $N$:", "content": latex_N},
        {"title": r"Esponenziale della parte diagonale $e^{Dt}$:", "content": r"e^{Dt} = " + " + ".join(latex_eDt_parts)},
        {"title": r"Potenza della parte diagonale $D^t$:",  "content": r"D^t = " + " + ".join(latex_Dt_parts)},
        {"title": r"Esponenziale della matrice $e^{At}$:", "content": r"e^{At} = " + sp.latex(sp.simplify(exp_At))},
        {"title": r"Potenza della matrice $A^t$:",  "content": r"A^t = " + sp.latex(sp.simplify(At_mat))}
    ]

    return jsonify({
        "success": True,
        "latex": steps,
        "latex_fi": latex_fi,
        "latex_proj": proj_lines,
        "latex_D": latex_D,
        "latex_N": latex_N,
        "nilpotence_index": k,
        "N_matrix_raw": sp.srepr(N_matrix),
        "proj_expr_list_raw": [sp.srepr(ei) for ei in proj_expr_list]
    })


# New endpoint for exp_power_At
@spectral_bp.route("/api/exp_power_At", methods=["POST"])
def compute_exp_power_At():
    t = sp.symbols("t")
    A = sp.Matrix(request.json["matrix"])
    N = sp.sympify(request.json["N_matrix_raw"])
    proj_expr_list = [sp.sympify(e) for e in request.json["proj_expr_list_raw"]]

    n = A.shape[0]
    D = A - N

    # indice di nilpotenza
    k = 1
    while not (N**k).equals(sp.zeros(*N.shape)):
        k += 1

    # autovalori tramite proiettori
    lambdas = [((Ei*D).trace()/Ei.trace()).simplify() for Ei in proj_expr_list]

    # e^{Dt} e D^{t}
    exp_Dt = sp.zeros(n)
    Dt     = sp.zeros(n)
    latex_eDt_parts = []
    latex_Dt_parts  = []
    for i, lam in enumerate(lambdas, start=1):
        Ei = proj_expr_list[i-1]
        exp_Dt += sp.exp(lam*t, evaluate=False) * Ei
        Dt     += sp.Pow(lam, t, evaluate=False) * Ei
        latex_eDt_parts.append(rf"e^{{{sp.latex(lam)}t}}E_{{{i}}}")
        latex_Dt_parts .append(rf"{sp.latex(lam)}^t E_{{{i}}}")

    # e^{At}
    exp_At = sum(((t**i/sp.factorial(i))*exp_Dt*(N**i) for i in range(k)), start=sp.zeros(n))
    # A^{t}
    At     = sum((sp.binomial(t,i)*Dt*(N**i) for i in range(k)), start=sp.zeros(n))

    return jsonify({
        "success": True,
        "latex_steps": [
            rf"e^{{Dt}} = " + " + ".join(latex_eDt_parts),
            rf"D^t     = " + " + ".join(latex_Dt_parts),
            rf"e^{{At}} = {sp.latex(sp.simplify(exp_At))}",
            rf"A^t     = {sp.latex(sp.simplify(At))}"
        ],
        "nilpotence_index": k
    })