from sympy.parsing.sympy_parser import parse_expr
from sympy.logic.boolalg import SOPform
from sympy import symbols
import itertools
from flask import Blueprint, request, jsonify
from sympy.logic.inference import satisfiable

polinomi_booleani_bp = Blueprint("polinomi_booleani", __name__)


from sympy.logic.boolalg import (
    And, Or, Not, simplify_logic, to_dnf, is_dnf
)
from sympy.abc import _clash1

import sympy
import re



# Function per output LaTeX with apice per la negazione
def latex_with_prime(expr):
    """
    Converte l'output LaTeX di SymPy sostituendo le negazioni (~x) con la
    notazione tramite apice: x^{\prime}
    Inoltre sostituisce \doubleprime con '' per evitare problemi di rendering.
    """
    s = sympy.latex(expr)

    # \lnot x   --> x^{\prime}
    s = re.sub(r'\\lnot\s*([a-zA-Z])', r'{\1}^{\\prime}', s)
    # \neg x   --> x^{\prime}  (alternative symbol)
    s = re.sub(r'\\neg\s*([a-zA-Z])', r'{\1}^{\\prime}', s)

    return s


def parse_boolean_expr(expr_str):
    """
    Converte una stringa booleana (usando +, ·, ') in una espressione SymPy.
    Esempio: (x + y)·z → And(Or(x, y), z)
    """
    # --- remove spazi e PUNTO finale -------------------------------------------------
    expr_str = expr_str.strip().rstrip(".")
    # ----------------------------------------------------------------------------------

    # Remove LaTeX size delimiters
    expr_str = expr_str.replace(r'\left', '').replace(r'\right', '')
    # Convert logical symbols
    expr_str = expr_str.replace(r'\land', '&').replace(r'\lor', '|')
    # Convert prime notation for NOT
    expr_str = re.sub(r'\^\{\\prime\}', "'", expr_str)
    # Remove any remaining LaTeX backslashes
    expr_str = expr_str.replace('\\', '')
    # Remove stray braces
    expr_str = expr_str.replace('{', '').replace('}', '')
    # Normalize algebraic operators
    expr_str = expr_str.replace("·", "&").replace("+", "|")

    # --- new section: handle double NOT operators -------------------------------
    # 1. LaTeX x^{\prime\prime}  →  x
    expr_str = re.sub(r"\^\{\\prime\\prime\}", "", expr_str)
    # 2. Varianti singola macro: ^{\\doubleprime}  →  x
    expr_str = re.sub(r"\^\{\\doubleprime\}", "", expr_str)
    # 3. Token notation: ^doubleprime or '' -> empty
    expr_str = expr_str.replace("^doubleprime", "")
    expr_str = expr_str.replace("''", "")
    # ----------------------------------------------------------------------

    # --- costanti with apice --------------------------------------------
    # 1' -> 0   ,   0' -> 1
    expr_str = re.sub(r"\b1'", "0", expr_str)
    expr_str = re.sub(r"\b0'", "1", expr_str)

    # Negazioni di gruppo ( (), [], {} ) seguite da apice
    expr_str = apply_group_negations(expr_str)

    # Convert prime notation for NOT on single symbols: x' → ~x
    expr_str = re.sub(r"([A-Za-z0-9_])'", r"~\1", expr_str)
    # --- costanti with tilde (~) ----------------------------------------
    # ~1 -> 0   ,   ~0 -> 1  (anche quando seguite da simboli non alfabetici)
    expr_str = re.sub(r"~\s*1(?![0-9])", "0", expr_str)
    expr_str = re.sub(r"~\s*0(?![0-9])", "1", expr_str)
    # Corregge eventuali tilde poste dopo il simbolo (es. x~ → ~x)
    expr_str = re.sub(r"([A-Za-z0-9_])~", r"~\1", expr_str)

    # Elimina eventuali doppie negazioni rimaste (~~x  ->  x)
    while '~~' in expr_str:
        expr_str = expr_str.replace('~~', '')

    # Remove stray "~k" tokens
    expr_str = expr_str.replace("~k", "")
    from sympy.parsing.sympy_parser import parse_expr
    return parse_expr(expr_str, evaluate=False, local_dict=_clash1)


def apply_group_negations(expr_str):
    """
    Converte ogni blocco racchiuso da () seguito da un apostrofo
    in una negazione booleana:  (…)'  → ~(…)
    Gestisce correttamente annidamenti arbitrari.
    """
    while "'" in expr_str:
        p = expr_str.find("'")           # posizione del primo apice
        if p == 0:
            expr_str = expr_str[1:]      # apice isolato, scartalo
            continue

        close_char = expr_str[p - 1]
        if close_char != ')':            # non è un apice di gruppo → passa
            expr_str = expr_str.replace("'", "~", 1)
            continue

        # risali all'indietro per trovare l'apertura bilanciata
        depth = 0
        open_idx = None
        for j in range(p - 2, -1, -1):
            c = expr_str[j]
            if c == ')':
                depth += 1
            elif c == '(':
                if depth == 0:
                    open_idx = j
                    break
                depth -= 1
        if open_idx is None:
            # parentesi non bilanciate, remove l'apice e continuous
            expr_str = expr_str[:p] + expr_str[p + 1:]
            continue

        # replace il blocco with la forma negata
        inner = expr_str[open_idx:p]  # include parentesi di chiusura
        negated = f"~{inner}"
        expr_str = expr_str[:open_idx] + negated + expr_str[p + 1:]

    return expr_str


#
# ---------------------------------------------------------------------------
# ----------  CONSENSUS‑METHOD  :  Sum of all prime implicants  -------------
# ---------------------------------------------------------------------------
def _term_literals(term):
    """
    Convert an And-term (product) into a frozenset of literals represented as
    tuples (symbol, polarity) where polarity is True for positive literal and
    False for negated literal.
    """
    if isinstance(term, sympy.Symbol):
        return frozenset([(term, True)])
    if term.func is sympy.Not:          # single negated literal
        return frozenset([(term.args[0], False)])
    # general And
    lits = set()
    for arg in term.args:
        if arg.func is sympy.Not:
            lits.add((arg.args[0], False))
        else:
            lits.add((arg, True))
    return frozenset(lits)


def _literals_to_term(lits):
    """Inverse of _term_literals: from frozenset back to SymPy And product"""
    factors = []
    for sym, pos in sorted(lits, key=lambda t: t[0].name):
        factors.append(sym if pos else sympy.Not(sym))
    if not factors:
        return sympy.true
    if len(factors) == 1:
        return factors[0]
    return sympy.And(*factors)


def _consensus(lits1, lits2):
    """
    Classical consensus between two product terms.

    Given two literal‑sets *lits1* and *lits2* (frozensets produced by
    `_term_literals`) the consensus is defined **only** when the two terms

      • differ in the polarity of **exactly one** variable x_k
      • are *otherwise identical*.

    In that case the consensus is the common part (all literals but x_k),
    i.e.  αx_k + βx_k'  ≡  α      where α is the product of the common
    literals.

    If the conditions are not met, the function returns ``None``.
    """
    # --- individua le variables with polarità opposta --------------------
    diff_var = None
    for var, pol in lits1:
        if (var, pol) not in lits2 and (var, not pol) in lits2:
            # trovato var with polarità opposta
            if diff_var is not None:      # più di una differenza
                return None
            diff_var = var

    # Devono esserci *esattamente* una differenza di polarità
    if diff_var is None:
        return None

    # --- verifica che le restanti letterali coincidano ------------------
    base1 = {lit for lit in lits1 if lit[0] != diff_var}
    base2 = {lit for lit in lits2 if lit[0] != diff_var}
    if base1 != base2:                    # non stesso “alpha”
        return None

    # Consensus = prodotto dei letterali comuni (può essere vuoto → True)
    return frozenset(base1)


def _remove_redundant(terms):
    """Eliminate every term that is a superset of some other term."""
    minimal = set(terms)
    for t1 in terms:
        for t2 in terms:
            if t1 is t2:
                continue
            if t1.issuperset(t2):
                minimal.discard(t1)
                break
    return minimal


def compute_stip(fnd_expr):
    """
    Given a (not‑necessarily‑reduced) SOP expression, compute the **sum of all
    prime implicants** (s.t.i.p.) via the consensus method.

    Returns the resulting SymPy expression.
    """
    # 1) break into product terms
    if fnd_expr is sympy.true:
        return sympy.true
    if fnd_expr is sympy.false:
        return sympy.false

    if isinstance(fnd_expr, sympy.Or):
        init_terms = list(fnd_expr.args)
    else:   # single product
        init_terms = [fnd_expr]

    # convert to literal frozensets and drop duplications
    term_sets = {_term_literals(t) for t in init_terms}
    term_sets = _remove_redundant(term_sets)

    # 2) iterative consensus closure
    changed = True
    while changed:
        changed = False
        new_terms = set(term_sets)
        term_list = list(term_sets)
        for i in range(len(term_list)):
            for j in range(i + 1, len(term_list)):
                c = _consensus(term_list[i], term_list[j])
                if c is not None and c not in new_terms:
                    new_terms.add(c)
                    changed = True
        # remove redundant after each round
        term_sets = _remove_redundant(new_terms)

    # 3) back to SymPy Or expression
    prime_terms = [_literals_to_term(ts) for ts in sorted(
        term_sets, key=lambda s: sorted([lit[0].name for lit in s]))]
    if not prime_terms:
        return sympy.false
    if len(prime_terms) == 1:
        return prime_terms[0]
    return sympy.Or(*prime_terms)


# --- NEW: Compute all minimal SOP forms from the s.t.i.p. ---
def compute_minimal_forms(stip_expr):
    """
    Given the s.t.i.p. expression (sum of all prime implicants) return a list
    with *all* minimal SOP expressions equivalent to it.
    A minimal SOP is obtained by removing the largest possible number of
    implicants while preserving equivalence.
    The search is exponential in the number of implicants but that number is
    typically small in educational use‑cases (≤ 10).
    """
    # Single‑term or constant expressions are already minimal
    if stip_expr.is_Atom or isinstance(stip_expr, sympy.Not):
        return [stip_expr]

    terms = list(stip_expr.args) if isinstance(stip_expr, sympy.Or) else [stip_expr]
    m = len(terms)
    minimal, seen, best_size = [], set(), m

    # combinations starting from the smallest possible size
    for r in range(1, m + 1):
        if r > best_size:
            break
        for idxs in itertools.combinations(range(m), r):
            subset = [terms[i] for i in idxs]
            candidate = subset[0] if len(subset) == 1 else sympy.Or(*subset)
            # equivalence test via satisfiability (robust even when simplify_logic
            # cannot reduce candidate ^ stip_expr to the literal `False`).   The two
            # expressions are equivalent iff their XOR is *unsatisfiable*.
            from sympy.logic.inference import satisfiable
            if not satisfiable(candidate ^ stip_expr):
                # canonical (order‑independent) representation of the term set
                canon = frozenset(candidate.args) if isinstance(candidate, sympy.Or) else frozenset([candidate])
                # if we found a *smaller* solution set -> reset
                if r < best_size:
                    minimal, seen, best_size = [candidate], {canon}, r
                elif r == best_size:
                    # keep every syntactically different set of implicants
                    if canon not in seen:
                        minimal.append(candidate)
                        seen.add(canon)
        if minimal:      # we already found the minimal size → stop
            break
    return minimal


def to_fnd(expr_str, var_names=None):
    """
    Converte un'espressione booleana stringa nella sua Forma Normale Disgiuntiva completa (FND canonica).
    """
    expr = parse_boolean_expr(expr_str)
    # 1) Determina le variables che l'utente desidera visualizzare
    if var_names is None:
        vars_sym = sorted(expr.free_symbols, key=lambda s: s.name)
        var_names = [str(v) for v in vars_sym]
    else:
        # crea i simboli nell'order richiesto (anche if non compaiono in expr)
        vars_sym = symbols(" ".join(var_names))
        if not isinstance(vars_sym, (list, tuple)):
            vars_sym = [vars_sym]

    # 2) Prepara espressione Python direttamente dalla struttura SymPy
    expr_py = str(expr)
    expr_py = expr_py.replace('~', ' not ')
    expr_py = expr_py.replace('&', ' and ')
    expr_py = expr_py.replace('|', ' or ')
    expr_py = re.sub(r"\.+$", "", expr_py.strip())

    # 3) Build truth table e lista mintermini via eval()
    tt_rows = []
    minterms = []
    for combo in itertools.product([False, True], repeat=len(var_names)):
        env = { name: combo[i] for i, name in enumerate(var_names) }
        try:
            val = bool(eval(expr_py, {}, env))
        except Exception as exc:
            raise ValueError(f"Errore nel valutare l’espressione: {exc}")
        tt_rows.append((combo, val))
        if val:
            minterms.append([int(v) for v in combo])

    # 4) Costruzione tabella di verità in LaTeX
    header = " & ".join(var_names) + " & P(" + ",".join(var_names) + ") \\\\"
    rows = []
    for combo, res in tt_rows:
        row_str = " & ".join(str(int(v)) for v in combo) + f" & {int(res)} \\\\"
        rows.append(row_str)
    truth_table_latex = (
        "\n\\begin{array}{|" + "c|" * (len(var_names) + 1) + "}\n"
        "\\hline\n"
        + header + "\n\\hline\n"
        + "\n".join(rows) + "\n\\hline\n"
        "\\end{array}\n\n"
    )

    # 5) Build la FND canonica esatta (somma dei mintermini, non ridotta)
    if not minterms:
        fnd_exact = sympy.false
    elif len(minterms) == 2 ** len(vars_sym):
        fnd_exact = sympy.true
    else:
        product_terms = []
        for combo in minterms:
            literals = []
            for v, bit in zip(vars_sym, combo):
                literals.append(v if bit else ~v)
            product_terms.append(sympy.And(*literals))
        fnd_exact = sympy.Or(*product_terms)

    return fnd_exact, truth_table_latex


def render_latex(expr):
    """
    Ritorna una rappresentazione LaTeX leggibile per SymPy BooleanFunction
    """
    return sympy.latex(expr)


# Flask API endpoint
@polinomi_booleani_bp.route("/api/boolean-polynomials", methods=["POST"])
@polinomi_booleani_bp.route("/api/polinomi-booleani", methods=["POST"])
def polinomi_booleani_route():
    try:
        data = request.get_json()
        expr_str = data.get("polinomio", "")
        if not expr_str:
            raise ValueError("Espressione mancante.")

        n_vars = int(data.get("n_vars", 0))
        if n_vars:
            base_names = list("xyzuvw")  # extend if needed
            var_names = base_names[:n_vars]
        else:
            var_names = None

        fnd_expr, truth_table_latex = to_fnd(expr_str, var_names)

        # STIP (sum of all prime implicants) via consensus
        stip_expr = compute_stip(fnd_expr)

        # --- NEW: Compute minimal SOP forms and their LaTeX ---
        minimal_forms = compute_minimal_forms(stip_expr)
        # Present every minimal form on its own display‐line with a label
        numbered_lines = [
            fr"\text{{f.m.}}^{{{idx+1}}} \;=\; {latex_with_prime(mf)}"
            for idx, mf in enumerate(minimal_forms)
        ]
        # each line stacked with nice alignment
        minimal_latex = (
            r"\begin{aligned}" + "\n" +
            r" \\[1ex]".join(numbered_lines) +
            r"\end{aligned}"
        )

        # Step sequence: 0) tabella, 1) FND, 2) s.t.i.p., 3) minimal forms
        steps = [{
            "title": "Tabella di verità dell’espressione:",
            "content": truth_table_latex
        }, {
            "title": "Forma Normale Disgiuntiva (FND):",
            "content": latex_with_prime(fnd_expr)
        }, {
            "title": "Somma di tutti gli implicanti primi (s.t.i.p.):",
            "content": fr"\text{{s.t.i.p.}}(P)\;=\;{latex_with_prime(stip_expr)}"
        }]
        steps.append({
            "title": "Forme minimali equivalenti (f.m.):",
            "content": minimal_latex
        })

        return jsonify({
            "success": True,
            "result": sympy.sstr(fnd_expr),
            "latex": steps
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })