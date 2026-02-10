"""
Blueprint per il Metodo del Simplesso (modalità DIZIONARIO).

JSON d’ingresso previsto
------------------------
{
  "tipo"      : "max" | "min",          # massimizzazione / minimizzazione
  "metodo"    : "dizionario",           # (in futuro anche "tableau")
  "standard"  : "si" | "no",            # se “si” le disequazioni sono già in forma standard
  "n"         : 3,                      # numero di vincoli  (2–4 per ora)
  "obj"       : "3x_1 + 2x_2",          # funzione obiettivo
  "constraints": [                      # lista di lhs, rhs (stringhe mathlive)
        ["x_1 + 2x_2", "4"],
        ["3x_1 + x_2", "5"],
        ["x_1 + x_2" , "3"]
  ]
}

Risposta
--------
{
  "success": true,
  "latex"  : [ {title:str, content:str}, ... ],
  "steps"  : idem,
  "result" : "z^* = 8 (x_1=..., x_2=...)"   # solo quando calcoleremo l’ottimo
}
"""
from __future__ import annotations

from flask import Blueprint, request, jsonify
from fractions import Fraction
from typing import List
import sympy as sp
from itertools import count
# ---------- parser MathLive --------------------------------------------- #
import re

simplesso_bp = Blueprint("simplesso_bp", __name__)

# ---------- utility LaTeX ------------------------------------------------ #
def _tex_num(x: sp.Rational | int | float) -> str:
    if isinstance(x, (int, sp.Integer)):
        return str(int(x))
    if isinstance(x, sp.Rational):
        return fr"\frac{{{x.p}}}{{{x.q}}}" if x.q != 1 else str(x.p)
    return str(x)

def _latex_align(lines: List[str]) -> str:
    """Ritorna un ambiente aligned con le righe passate."""
    return r"\begin{aligned}" + r"\\ ".join(lines) + r"\end{aligned}"

def _dictionary_to_latex(base_vars, dict_expr, z_expr, nonbase_vars):
    """
    Restituisce una stringa LaTeX (aligned) del dizionario corrente.
      base_vars      : list[Symbol] variabili di base (nell’ordine)
      dict_expr      : dict{Symbol: Expr} mappa base -> RHS (in termini di non base)
      z_expr         : Expr, funzione obiettivo (espressa in non base)
      nonbase_vars   : list[Symbol]
    """
    rows = [ sp.latex(sp.Eq(b, dict_expr[b])) for b in base_vars ]
    rows.append("z = " + sp.latex(z_expr))
    return _latex_align(rows)

def _is_feasible(base_vars, dict_expr, nonbase_vars):
    """
    Dizionario ammissibile se, fissando tutte le variabili non di base a 0,
    il termine costante di ogni variabile di base è ≥ 0.
    """
    subs_zero = {v: 0 for v in nonbase_vars}
    for b in base_vars:
        const = sp.simplify(dict_expr[b].subs(subs_zero))
        if const < 0:
            return False
    return True

# ---------- parser MathLive --------------------------------------------- #
_var_re   = re.compile(r"x_(\d+)")
_frac_re  = re.compile(r"\\frac\{(-?\d+)\}\{(-?\d+)\}")

def _parse_expr(expr: str) -> sp.Expr:
    """Converte stringhe tipo '3x_1 + \frac12 x_3' in un’espressione sympy."""
    # gestisci frazioni LaTeX
    def _frac_sub(m: re.Match[str]) -> str:
        num, den = m.groups()
        return f"({num})/({den})"
    s = _frac_re.sub(_frac_sub, expr.replace(" ", ""))

    # trasforma \cdot in moltiplicazione
    s = s.replace(r"\cdot", "*")

    # inserisci il simbolo di moltiplicazione implicita: es. 3x_1 -> 3*x_1
    s = re.sub(r'(?<=[0-9\)])(x_\d+)', r'*\1', s)

    # sostituisci x_i → x_i (sympy symbols)
    tokens = _var_re.findall(s)
    syms   = {f"x_{i}": sp.symbols(f"x_{i}") for i in tokens}
    # NB: sympy non accetta _ nelle variabili senza quoting: usiamo x_1 come sympy Symbol name
    return sp.sympify(s, locals=syms)

# ---------- ROUTE -------------------------------------------------------- #
@simplesso_bp.route("/api/simplesso", methods=["POST"])
def simplesso_route():
    data = request.get_json(silent=True) or {}
    tipo      = data.get("tipo", "max")
    metodo    = data.get("metodo", "dizionario")
    standard  = data.get("standard", "no")
    n         = int(data.get("n", 0))
    obj_raw   = data.get("obj", "")
    constr    = data.get("constraints", [])

    if metodo != "dizionario":
        return jsonify(success=False, error="Solo modalità 'dizionario' al momento."), 400
    if n not in (2, 3, 4) or len(constr) != n:
        return jsonify(success=False, error="Numero di vincoli non valido."), 400

    # ------------ parsing -------------------------------------------------
    try:
        obj_expr = _parse_expr(obj_raw)
    except Exception as e:
        return jsonify(success=False, error=f"Errore nel parsing: {e}"), 400

    # --- Convert minimization to maximization --------------------------
    is_min = (tipo == "min")
    if is_min:
        obj_expr = -obj_expr   # we will treat everything as a max‑problem

    lhs_list = []
    rhs_vec = []
    signs = []
    for lhs_raw, rhs_raw, sg in constr:
        try:
            lhs_list.append(_parse_expr(lhs_raw))
            rhs_vec.append(_parse_expr(rhs_raw))
        except Exception as e:
            return jsonify(success=False, error=f"Errore nel parsing: {e}"), 400
        if sg not in ("<=", ">="):
            return jsonify(success=False, error="Segno vincolo non valido"), 400
        signs.append(sg)


    # colleziona tutte le variabili x_i presenti
    max_index = 0
    for expr in [obj_expr] + lhs_list:
        max_index = max(max_index,
                        *(int(m) for m in _var_re.findall(str(expr))) or [0])

    steps: List[dict] = []

    # ---------- STEP 0 : problema inserito --------------------------------
    align_lines = [ (r"\text{%s } z =" % ("min" if tipo=="min" else "max")) +
                    sp.latex(obj_expr) + ":" ]
    # sign = r"\le" if tipo=="max" else r"\ge"
    for lhs, rhs, sg in zip(lhs_list, rhs_vec, signs):
        sym = sg.replace("<=", r"\le").replace(">=", r"\ge")
        align_lines.append(f"{sp.latex(lhs)} {sym} {sp.latex(rhs)}")
    # elenco esplicito variabili non negative
    vars0 = ", ".join(f"x_{{{j}}}" for j in range(1, max_index + 1))
    align_lines.append(vars0 + " \\ge 0")

    steps.append({
        "title": "Problema inserito:",
        "content": _latex_align(align_lines)
    })

    # ---------- STEP 1 : forma standard -----------------------------------
    slack_syms = []
    std_lines  = []
    for i in range(n):
        sg   = signs[i]
        lhs  = lhs_list[i]
        rhs  = rhs_vec[i]

        # --- normalizza: rendi RHS non‑negativo --------------------------
        if rhs.free_symbols:   # dovrebbe essere costante; se non lo è, lo lasciamo così
            rhs_val = rhs
        else:
            rhs_val = sp.nsimplify(rhs)
            if rhs_val < 0:
                lhs  = -lhs
                rhs  = -rhs_val
                sg   = "<=" if sg == ">=" else ">="   # inverti il verso

        # salva le forme normalizzate
        lhs_list[i] = lhs
        rhs_vec[i]  = rhs
        signs[i]    = sg

        # --- aggiungi variabile di slack / surplus -----------------------
        slack_idx = max_index + len(slack_syms) + 1
        s_i = sp.symbols(f"x_{slack_idx}")
        slack_syms.append(s_i)

        if sg == "<=":
            expr_tex = fr"{sp.latex(lhs)} + {sp.latex(s_i)} = {sp.latex(rhs)}"
        else:  # ">="
            expr_tex = fr"{sp.latex(lhs)} - {sp.latex(s_i)} = {sp.latex(rhs)}"

        std_lines.append(expr_tex)

    # elenco esplicito variabili e slack non negative
    vars_std = ", ".join(f"x_{{{j}}}" for j in range(1, max_index + n + 1))
    std_lines.append(vars_std + " \\ge 0")

    steps.append({
        "title": "Forma standard (variabili di slack o surplus):",
        "content": _latex_align(std_lines)
    })

    # ---------- STEP 2 : dizionario iniziale ------------------------------
    # variabili di base = slack, non–base = originali
    dict_lines = []
    for i in range(n):
        bi  = rhs_vec[i]
        row = lhs_list[i]
        dict_eq = sp.Eq(slack_syms[i], bi - row)  # s_i = b_i − (combinazione di x)
        dict_lines.append(sp.latex(dict_eq))

    # funzione obiettivo (convertita a massimizzazione se necessario)
    z_expr   = obj_expr

    # esprimi z in funzione delle variabili NON di base (sostituisci le base)
    for b in slack_syms:                            # all slack are in base at start
        z_expr = z_expr.subs({b: rhs_vec[slack_syms.index(b)] - lhs_list[slack_syms.index(b)]})
    dict_lines.append(r"z = " + sp.latex(z_expr))

    steps.append({
        "title": "Dizionario iniziale:",
        "content": _latex_align(dict_lines)
    })

    # =======================================
    #  Iterazioni del metodo del simplesso
    # =======================================
    base_vars     = slack_syms[:]                 # variabili di base
    nonbase_vars  = [sp.symbols(f"x_{j}") for j in range(1, max_index+1)]
    dict_expr     = {b: rhs_vec[i] - lhs_list[i] for i, b in enumerate(base_vars)}

    # ---------------------------------------------------------
    #  Fase I: verifica se il dizionario iniziale è ammissibile
    # ---------------------------------------------------------
    def _pivot_dictionary(base_vars, nonbase_vars, dict_expr, obj_expr,
                          enter, leave):
        """
        Esegue un pivot (dictionary form) ed aggiorna in place:
            base_vars, nonbase_vars (liste)
            dict_expr  (dict base -> RHS)
            obj_expr   (sympy Expr)   → valore di ritorno
        Restituisce la nuova obj_expr.
        """
        pivot_eq = dict_expr[leave]
        a        = pivot_eq.coeff(enter)
        new_rhs  = (pivot_eq - a*enter) / (-a)   # enter espresso in NB
        # sostituisci nelle altre righe
        for b in base_vars:
            if b == leave:
                continue
            dict_expr[b] = dict_expr[b].subs({enter: new_rhs})
        obj_expr = obj_expr.subs({enter: new_rhs})
        # update liste variabili
        base_vars.remove(leave)
        nonbase_vars.remove(enter)
        base_vars.append(enter)
        nonbase_vars.append(leave)
        # aggiorna dizionario per la variabile entrante
        del dict_expr[leave]
        dict_expr[enter] = new_rhs
        return obj_expr

    if not _is_feasible(base_vars, dict_expr, nonbase_vars):
        steps.append({
            "title": "Dizionario iniziale NON ammissibile:",
            "content": _dictionary_to_latex(base_vars, dict_expr, z_expr, nonbase_vars)
        })
        # Interrompiamo subito l'esercizio: non gestiamo più la Fase I
        steps.append({
            "title": "Esercizio terminato:",
            "content": r"\text{Il dizionario iniziale non è ammissibile.}"
        })
        return jsonify(success=True, steps=steps, latex=steps)

    MAX_ITERS = 10
    for it in range(1, MAX_ITERS+1):
        # -------- scegliere variabile entrante ---------------------------
        coeffs = {v: z_expr.coeff(v) for v in nonbase_vars}
        # per max: scegli coeff > 0 massimo
        entering = None
        best_c   = 0
        for v, c in coeffs.items():
            if c > best_c:
                best_c = c
                entering = v
        if entering is None:  # ottimo trovato
            steps.append({
                "title": "Ottimalità raggiunta:",
                "content": _dictionary_to_latex(base_vars, dict_expr, z_expr, nonbase_vars)
            })
            break

        # -------- scegliere variabile uscente ----------------------------
        min_ratio = None
        leaving   = None
        for b in base_vars:
            expr   = dict_expr[b]
            a      = expr.coeff(entering)
            if a >= 0:  # coeff non “migliora” => ignora
                continue
            const = expr.subs({v: 0 for v in nonbase_vars})
            ratio = const / (-a)
            if ratio >= 0 and (min_ratio is None or ratio < min_ratio):
                min_ratio = ratio
                leaving   = b

        if leaving is None:
            steps.append({
                "title": "Problema illimitato:",
                "content": r"\text{Nessun vincolo impone un limite superiore} \Rightarrow \text{problema illimitato superiormente} \\[2em]"
            })
            break

        steps.append({
            "title": fr"Iterazione {it}. Entrante ${sp.latex(entering)}$, uscente ${sp.latex(leaving)}:$",
            "content": _dictionary_to_latex(base_vars, dict_expr, z_expr, nonbase_vars)
        })

        # -------- pivot ---------------------------------------------------
        pivot_eq = dict_expr[leaving]
        a        = pivot_eq.coeff(entering)
        # risolvi per entering
        new_rhs  = (pivot_eq - a*entering) / (-a)
        # sostituisci in tutte le altre righe
        for b in base_vars:
            if b == leaving:
                continue
            dict_expr[b] = dict_expr[b].subs({entering: new_rhs})
        z_expr = z_expr.subs({entering: new_rhs})
        # aggiorna dizionario: leaving diventa nonbase, entering base
        del dict_expr[leaving]
        dict_expr[entering] = new_rhs
        base_vars.remove(leaving)
        nonbase_vars.remove(entering)
        base_vars.append(entering)
        nonbase_vars.append(leaving)

        # mostra dizionario aggiornato
        steps.append({
            "title": fr"Dizionario aggiornato (iter {it}):",
            "content": _dictionary_to_latex(base_vars, dict_expr, z_expr, nonbase_vars)
        })
    # ----- fine iterazioni ------------------------------------------------

    if entering is None:  # ottimalità trovata
        solution_lines = []
        optimal_val = z_expr.subs({v: 0 for v in nonbase_vars})
        if is_min:
            optimal_val = -optimal_val
        for j in range(1, max_index+1):
            var = sp.symbols(f"x_{j}")
            if var in base_vars:
                val = dict_expr[var].subs({v: 0 for v in nonbase_vars})
            else:
                val = 0
            solution_lines.append(fr"{sp.latex(var)} = {sp.latex(val)}")
        tuple_vars = ", ".join(fr"x_{{{j}}}" for j in range(1, max_index+1))
        tuple_vals = ", ".join(sp.latex(
                dict_expr[sp.symbols(f"x_{j}")].subs({v: 0 for v in nonbase_vars})
            ) if sp.symbols(f"x_{j}") in base_vars else "0"
            for j in range(1, max_index+1)
        )
        tuple_line = fr"\left({tuple_vars}\right) = \left({tuple_vals}\right)"
        steps.append({
            "title": "Soluzione ottimale:",
            "content": _latex_align([tuple_line, fr"z^* = {sp.latex(optimal_val)}"])
        })

    # --- (da qui in poi: test di ammissibilità e pivot, non implementato) --

    return jsonify(success=True, steps=steps, latex=steps)