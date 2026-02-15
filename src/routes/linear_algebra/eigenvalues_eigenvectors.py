from __future__ import annotations

from flask import Blueprint, request, jsonify
from typing import List, Tuple, Set
from fractions import Fraction
import re
import sympy as sp

eigen_bp = Blueprint("autovalori_autovettori_bp", __name__)

# ---------- Utility: number → LaTeX -------------------------------------- #
def _to_tex_num(x):
    if isinstance(x, Fraction):
        if x.denominator == 1:
            return str(x.numerator)
        return fr"\frac{{{x.numerator}}}{{{x.denominator}}}"
    # SymPy numbers or Python numbers
    if isinstance(x, sp.Basic):
        # simplify + latex
        return sp.latex(sp.nsimplify(x))
    try:
        if int(x) == x:
            return str(int(x))
    except Exception:
        pass
    return str(x)

# ---------- Utility: vector → LaTeX -------------------------------------- #
def latex_vector(v: sp.Matrix) -> str:
    body = r" \\ ".join(_to_tex_num(c) for c in v)
    return r"\mathbf{v}=\begin{bmatrix} " + body + r" \end{bmatrix}"

# ---------- Utility: matrix → LaTeX -------------------------------------- #
def latex_matrix(mat: List[List], symbol_lambda: sp.Symbol | None = None) -> str:
    """
    Se symbol_lambda è fornito, gli elementi contenenti `symbol_lambda`
    vengono lasciati in forma LaTeX sympy.latex(elem).
    """
    rows_tex = []
    for row in mat:
        cols_tex = []
        for val in row:
            if symbol_lambda is not None and symbol_lambda in sp.sympify(val).free_symbols:
                cols_tex.append(sp.latex(val))
            else:
                cols_tex.append(_to_tex_num(val))
        rows_tex.append(" & ".join(cols_tex))
    body = r" \\ ".join(rows_tex)
    return fr"\begin{{bmatrix}} {body} \end{{bmatrix}}"

# ---------- ROUTE --------------------------------------------------------- #
@eigen_bp.route("/api/eigenvalues-eigenvectors", methods=["POST"])
@eigen_bp.route("/api/autovalori-autovettori", methods=["POST"])
def autovalori_autovettori_route():
    data = request.get_json(silent=True) or {}
    n = data.get("n")
    matrix = data.get("matrix")

    # ----- validation -----------------------------------------------------
    if not isinstance(n, int) or n not in (2, 3, 4):
        return jsonify(success=False, error="Parametro 'n' mancante o non valido (2–4)."), 400
    if (not isinstance(matrix, list) or
        len(matrix) != n or
        any(not isinstance(row, list) or len(row) != n for row in matrix)):
        return jsonify(success=False, error="Parametro 'matrix' non conforme (n×n)."), 400

    # conversion to Fractions so that SymPy keeps exact arithmetic
    mat_frac: List[List[Fraction]] = []
    for row in matrix:
        new_row: List[Fraction] = []
        for x in row:
            if isinstance(x, str) and x.startswith("\\frac"):
                clean = x.replace(" ", "")
                if clean.startswith("\\frac{"):
                    m = re.match(r"^\\frac\{(-?\d+)\}\{(-?\d+)\}$", clean)
                else:
                    m = re.match(r"^\\frac(-?\d+)(-?\d+)$", clean)
                if not m:
                    return jsonify(success=False, error=f"Formato frazione non valido: {x}"), 400
                num_str, den_str = m.groups()
                try:
                    new_row.append(Fraction(int(num_str), int(den_str)))
                except ZeroDivisionError:
                    return jsonify(success=False, error="Denominatore zero nella frazione."), 400
            else:
                try:
                    new_row.append(Fraction(x))
                except (TypeError, ValueError):
                    return jsonify(success=False, error="Valori non numerici nella matrice."), 400
        mat_frac.append(new_row)

    # build SymPy matrix
    A = sp.Matrix(mat_frac)
    lam = sp.symbols("λ")

    steps: List[dict] = []

    # Step 0 – Matr  iniziale
    steps.append({"title": "Matrice iniziale:", "content": latex_matrix(mat_frac)})

    try:
        # Step 1 – Polinomio caratteristico
        charpoly = A.charpoly(lam)
        char_expr = sp.expand(charpoly.as_expr())
        content1 = (
            fr"p_A(\lambda)=\det\!\left({latex_matrix((A - lam*sp.eye(n)).tolist(), lam)}\right)"
            fr"={sp.latex(char_expr)}"
        )
        fact_expr = sp.factor(char_expr)
        if fact_expr != char_expr:
            content1 += fr"={sp.latex(fact_expr)}"
        steps.append({"title": "Polinomio caratteristico:", "content": content1})

        # Step 2 – Elenco autovalori
        eigen_vals = [ev[0] for ev in A.eigenvects()]
        eig_list_tex = r",\;".join(fr"\lambda_{i+1}={sp.latex(val)}" for i, val in enumerate(eigen_vals))
        steps.append({"title": "Autovalori:", "content": eig_list_tex})

        # Step 3+ – Autospazi
        eigen_data = A.eigenvects()
        eigen_summary = []
        diag_ok = True
        for idx, (eig_val, alg_mult, eig_vecs) in enumerate(eigen_data, 1):
            geo_mult = len(eig_vecs)
            value_latex = sp.latex(eig_val)
            eigen_summary.append({
                "value": value_latex,  # store as LaTeX string, JSON‑serializable
                "alg_mult": int(alg_mult),
                "geo_mult": int(geo_mult)
            })
            if geo_mult != alg_mult:
                diag_ok = False

            expr_label = fr"\ker\!\left(A-\lambda_{idx} I\right)\;="
            B = (A - eig_val*sp.eye(n))
            B_rref, _ = B.rref()
            mat_latex = latex_matrix(B.tolist())
            rref_latex = latex_matrix(B_rref.tolist())
            vecs_latex = " ,\\; ".join(latex_vector(v) for v in eig_vecs) if geo_mult > 0 else ""
            content_ev = (
                fr"{expr_label}"
                fr"{mat_latex}"
                fr"\;\Longrightarrow\;"
                fr"{rref_latex}"
            )
            if vecs_latex:
                content_ev += fr"\;\Longrightarrow\;{vecs_latex}"
            steps.append({"title": fr"Autospazio per $\lambda_{idx}$", "content": content_ev})

        # --- Tabella riepilogativa MA / MG ----------------------------
        table_rows = [
            fr"{item['value']} & {item['alg_mult']} & {item['geo_mult']}"
            for item in eigen_summary
        ]
        table_latex = (
            r"\begin{array}{c|c|c}"
            r"\lambda & \text{MA} & \text{MG}\\\hline "
            + r"\\ ".join(table_rows) +
            r"\end{array}"
        )
        steps.append({
            "title": "Molteplicità algebriche e geometriche:",
            "content": table_latex
        })

        result = {
            "eigenvalues": eigen_summary,
            "diagonalizable": diag_ok
        }
        return jsonify(success=True, steps=steps, result=result)
    except Exception as exc:  # pragma: no cover - fallback error path
        return jsonify(success=False, error=f"Errore nel calcolo: {exc}"), 500
    finally:
        # placeholder to satisfy linters that require finally/except presence
        pass
