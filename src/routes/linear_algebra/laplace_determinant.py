"""
Blueprint per il calcolo del determinante tramite sviluppo di Laplace
fino a matrici 4 × 4, con selezione automatica di riga/colonna che
massimizza il numero di zeri (tie‑break: prima riga, poi colonne).

Il JSON d’ingresso atteso è:
{
  "n": 3,                 # dimensione (2–4)
  "matrix": [[...], ...]  # lista di liste numeriche n×n
}

La risposta è:
{
  "success": true,
  "latex": [ {title:str, content:str}, ... ],
  "result": det // numero intero / float
}
"""
from __future__ import annotations

from flask import Blueprint, request, jsonify
from typing import List, Tuple, Set
from fractions import Fraction
import re

det_laplace_bp = Blueprint("determinante_laplace_bp", __name__)


# ---------- Utilità LaTeX ------------------------------------------------- #
def _to_tex_num(x: float | int) -> str:
    # Support Fraction, float, int
    if isinstance(x, Fraction):
        if x.denominator == 1:
            return str(x.numerator)
        else:
            return fr"\frac{{{x.numerator}}}{{{x.denominator}}}"
    try:
        if int(x) == x:
            return str(int(x))
    except Exception:
        pass
    return str(x)


def latex_matrix(mat: List[List[float]], highlight: Set[Tuple[int, int]] | None = None) -> str:
    """
    Genera codice LaTeX di una matrice con eventuale evidenziazione
    (\color{red}{0}) delle posizioni in `highlight`.
    """
    highlight = highlight or set()
    rows_tex = []
    for i, row in enumerate(mat):
        cols_tex = []
        for j, val in enumerate(row):
            cell = _to_tex_num(val)
            if (i, j) in highlight and val == 0:
                cell = fr"\color{{red}}{{{cell}}}"
            cols_tex.append(cell)
        rows_tex.append(" & ".join(cols_tex))
    body = r" \\ ".join(rows_tex)
    return fr"\begin{{bmatrix}} {body} \end{{bmatrix}}"


# ---------- Algoritmo di ricerca row/column with più zeri ---------------- #
def best_row_or_col(mat: List[List[float]]) -> Tuple[str, int, Set[Tuple[int, int]]]:
    """
    Restituisce ('row'|'col', indice, posizioni_zero).
    In caso di parità, viene scelta:
      1) la prima riga con massimo zeri
      2) se nessuna riga “batte” le colonne, la prima colonna con max zeri
    """
    n = len(mat)
    row_zero_counts = [sum(1 for val in row if val == 0) for row in mat]
    col_zero_counts = [sum(1 for i in range(n) if mat[i][j] == 0) for j in range(n)]

    max_row_zeros = max(row_zero_counts)
    best_row_idx = row_zero_counts.index(max_row_zeros)

    max_col_zeros = max(col_zero_counts)
    best_col_idx = col_zero_counts.index(max_col_zeros)

    if max_row_zeros == 0 and max_col_zeros == 0:
        return "none", 0, set()  # Nessuno zero: nessuna evidenziazione

    if max_row_zeros >= max_col_zeros:  # prefer row
        pos = {(best_row_idx, j) for j, v in enumerate(mat[best_row_idx]) if v == 0}
        return "row", best_row_idx, pos
    else:
        pos = {(i, best_col_idx) for i in range(n) if mat[i][best_col_idx] == 0}
        return "col", best_col_idx, pos


# ---------- Determinante (ricorsivo) with passi ---------------------------- #
def determinant(mat: List[List[float]],
                steps: List[dict],
                depth: int = 0) -> float:
    """
    Calcola il determinante restituendo i passi LaTeX nello stesso
    array `steps`. Ogni livello di ricorsione aggiunge dettagli.
    """
    n = len(mat)

    if n == 2:
        a, b = mat[0]
        c, d = mat[1]
        det_val = a * d - b * c
        # Step LaTeX per 2×2 (only if non siamo nel livello più profondo)
        title = "Determinante 2×2:"
        content = (
            fr"\det\!\left({latex_matrix(mat)}\right)="
            fr"\left({_to_tex_num(a)}\right)\left({_to_tex_num(d)}\right)"
            fr"-\left({_to_tex_num(b)}\right)\left({_to_tex_num(c)}\right)"
            fr"={_to_tex_num(det_val)}"
        )
        steps.append({"title": title, "content": content})
        return det_val

    # --------------- Selezione row/column ottimale ---------------
    kind, idx, zeros_pos = best_row_or_col(mat)
    if depth == 0:
        if kind == "none":
            steps.append({
                "title": "Nessuno zero presente nella matrice:",
                "content": "Non sono presenti zeri: l'espansione di Laplace userà la prima riga arbitrariamente."
            })
        else:
            title_sel = f"Selezione della {'riga' if kind=='row' else 'colonna'} con più zeri:"
            content_sel = latex_matrix(mat, highlight=zeros_pos)
            steps.append({"title": title_sel, "content": content_sel})

    # --------------- Espansione di Laplace -------------------------
    terms_tex = []
    det_total: Fraction = Fraction(0, 1)
    n_range = range(n)

    def minor_matrix(m: List[List[float]], r: int, c: int) -> List[List[float]]:
        """Return the submatrix obtained by removing row r and column c."""
        return [ [m[i][j] for j in n_range if j != c]
                 for i in n_range if i != r ]

    # Prepara la lista di posizioni per row o column scelta
    positions = [(idx, j) for j in n_range] if kind == "row" else [(i, idx) for i in n_range]

    # Costruzione espressione espansa
    joined_terms = " + ".join(
        fr"{'-' if (-1)**(i+j) < 0 else ''}{_to_tex_num(mat[i][j])}"
        fr"\cdot\det\!\left({latex_matrix(minor_matrix(mat, i, j))}\right)"
        for i, j in positions if mat[i][j] != 0
    ).replace("+ -", "- ")

    if depth == 0:
        expr = fr"\det\!\left({latex_matrix(mat)}\right)= {joined_terms}"
        steps.append({"title": "Espansione di Laplace:", "content": expr})

    # Ora ciclo reale per computation numerico
    for i, j in positions:
        a_ij = mat[i][j]
        if a_ij == 0:  # termine nullo → salta
            continue
        sign = (-1) ** (i + j)
        sub = minor_matrix(mat, i, j)
        sub_det = determinant(sub, steps, depth + 1)
        det_total += sign * a_ij * sub_det
        sign_tex = "-" if sign < 0 else ""
        term_tex = (
            fr"{sign_tex}{_to_tex_num(a_ij)}"
            fr"\cdot\det\!\left({latex_matrix(sub)}\right)"
        )
        terms_tex.append(term_tex)

    if depth == 0:
        steps.append({
            "title": "Risultato finale:",
            "content": fr"\det\!\left({latex_matrix(mat)}\right)= {_to_tex_num(det_total)}"
        })

    return det_total


# ---------- ROUTE --------------------------------------------------------- #
@det_laplace_bp.route("/api/laplace-determinant", methods=["POST"])
@det_laplace_bp.route("/api/determinante-laplace", methods=["POST"])
def determinante_laplace_route():
    """
    Calcola il determinante (2×2 – 4×4) con sviluppo di Laplace.
    Algoritmo:
      • Passo 0: mostra la matrice inserita.
      • Passo 1: evidenzia riga/colonna con più zeri.
      • Passo 2: espansione di Laplace con sottomatrici inserite.
      • Passi successivi: determinanti ricorsivi delle sottomatrici.
    """
    data = request.get_json(silent=True) or {}
    n = data.get("n")
    matrix = data.get("matrix")

    # ----- Validazione ----------------------------------------------------
    if not isinstance(n, int) or n not in (2, 3, 4):
        return jsonify(success=False, error="Parametro 'n' mancante o non valido (2–4)."), 400
    if (not isinstance(matrix, list) or
        len(matrix) != n or
        any(not isinstance(row, list) or len(row) != n for row in matrix)):
        return jsonify(success=False, error="Parametro 'matrix' non conforme (n×n)."), 400
    # Conversione numerica with supporto a frazioni LaTeX (\frac)
    mat: List[List[Fraction]] = []
    for row in matrix:
        new_row: List[Fraction] = []
        for x in row:
            if isinstance(x, str) and x.startswith('\\frac'):
                clean = x.replace(' ', '')
                # Case 1: both numerator and denominator in braces: \frac{n}{d}
                if clean.startswith('\\frac{'):
                    m = re.match(r'^\\frac\{(-?\d+)\}\{(-?\d+)\}$', clean)
                else:
                    # Case 2: no braces, assume simple two integers: \fracnd
                    m = re.match(r'^\\frac(-?\d+)(-?\d+)$', clean)
                if not m:
                    return jsonify(success=False, error=f"Formato frazione non valido: {x}"), 400
                num_str, den_str = m.groups()
                try:
                    frac = Fraction(int(num_str), int(den_str))
                except ZeroDivisionError:
                    return jsonify(success=False, error="Denominatore zero nella frazione."), 400
                new_row.append(frac)
            else:
                try:
                    new_row.append(Fraction(x))
                except (TypeError, ValueError):
                    return jsonify(success=False, error="La matrice contiene valori non numerici."), 400
        mat.append(new_row)

    steps: List[dict] = []

    # Step 0 - Initial matrix
    steps.append({
        "title": "Matrice iniziale:",
        "content": latex_matrix(mat)
    })

    # Computation determinante with passi
    det_val = determinant(mat, steps)

    return jsonify(success=True, steps=steps, latex=steps, result=_to_tex_num(det_val))
 
 
