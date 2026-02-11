# routes/controlli/equazioni_differenze.py

from flask import Blueprint, request, jsonify
import sympy as sp
import re
import os
from sympy import symbols, Function, Eq, latex, simplify, expand, linear_eq_to_matrix
from sympy.core.function import AppliedUndef
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)

equazioni_differenze_bp = Blueprint("equazioni_differenze", __name__)

# ------------------------------------------------------------
#  Normalizzazione LaTeX-like -> SymPy-like (discreto)
# ------------------------------------------------------------
def normalize_latex_like_discrete(expr: str) -> str:
    r"""
    Normalizza una sintassi LaTeX-like controllata in sintassi SymPy-style.
    Per equazioni alle differenze con operatore shift (Δ y(t) = y(t+1)).
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

    # ------------------------------------------------------------
    #  1) Rimuovi token LaTeX di sizing PRIMA di toccare i backslash
    #     (altrimenti "\\left" diventa "left" e il parser lo interpreta
    #     come prodotto implicito l*e*f*t ...)
    # ------------------------------------------------------------
    # gestisci sia stringhe con doppio backslash ("\\\\left") che con singolo ("\\left")
    expr = re.sub(r"\\\\left\b", "", expr)
    expr = re.sub(r"\\\\right\b", "", expr)
    expr = re.sub(r"\\left\b", "", expr)
    expr = re.sub(r"\\right\b", "", expr)

    # difensivo: se per qualche motivo i backslash sono già stati rimossi
    # (non dovrebbe accadere prima del parsing)
    expr = re.sub(r"\bleft\b", "", expr)
    expr = re.sub(r"\bright\b", "", expr)

    # ------------------------------------------------------------
    #  2) Impulso unitario discreto: \delta_0( ... ) -> KroneckerDelta(t, k)
    #     Supporta forme:
    #       \delta_0(t), \delta_0(t-2), \delta_0(t+2)
    #       \delta_{0}(\left(t\right)) e varianti con/ senza backslash
    # ------------------------------------------------------------
    # normalizza indice: \delta_{0} -> \delta_0 (gestisci anche doppio backslash)
    expr = expr.replace(r"\\delta_{0}", r"\\delta_0")
    expr = expr.replace(r"\delta_{0}", r"\delta_0")

    # Prima gestisci i casi con shift (t-k) / (t+k) con o senza backslash.
    # Nota: dopo la rimozione di \left/\right restano parentesi normali.
    def _kd_from_sign(sign: str, num: str) -> str:
        k = int(num)
        # \delta_0(t-k) -> KroneckerDelta(t,k)
        # \delta_0(t+k) -> KroneckerDelta(t,-k) (così poi simplify_delta_t_nonneg lo annulla per t>=0)
        return f"KroneckerDelta(t,{k})" if sign == '-' else f"KroneckerDelta(t,{-k})"

    expr = re.sub(
        r"(?:\\delta_0|\\\\delta_0|\bdelta_0\b)\s*\(\s*t\s*([\+\-])\s*(\d+)\s*\)",
        lambda m: _kd_from_sign(m.group(1), m.group(2)),
        expr,
    )

    # Caso base: delta_0(t)
    expr = re.sub(
        r"(?:\\delta_0|\\\\delta_0|\bdelta_0\b)\s*\(\s*t\s*\)",
        "KroneckerDelta(t,0)",
        expr,
    )
    # Caso shorthand: delta_0 t / delta_0t (senza parentesi)
    expr = re.sub(
        r"(?:\\delta_0|\\\\delta_0|\bdelta_0\b)\s*t\b",
        "KroneckerDelta(t,0)",
        expr,
    )
    expr = re.sub(r"δ_0\s*t\b", "KroneckerDelta(t,0)", expr)
    expr = re.sub(r"δ_0t\b", "KroneckerDelta(t,0)", expr)
    expr = re.sub(r"delta_0t\b", "KroneckerDelta(t,0)", expr)

    # Unicode delta (se mai arrivasse): δ_0(t)
    expr = re.sub(r"δ_0\s*\(\s*t\s*\)", "KroneckerDelta(t,0)", expr)

    # ------------------------------------------------------------
    #  Radici: \sqrt{a} -> sqrt(a),  \sqrt2 -> sqrt(2)
    # ------------------------------------------------------------
    expr = re.sub(r"\\sqrt\s*\{([^}]*)\}", r"sqrt(\1)", expr)
    expr = re.sub(r"\\sqrt\s*([0-9]+)", r"sqrt(\1)", expr)

    # \frac{a}{b} -> (a)/(b)
    expr = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', expr)

    # \frac12 -> 1/2
    expr = re.sub(r'\\frac\s*([0-9]+)\s*([0-9]+)', r'\1/\2', expr)

    # simboli unicode
    expr = expr.replace("π", "pi")

    # operatori
    expr = expr.replace(r"\cdot", "*")
    expr = expr.replace("^", "**")

    # parentesi LaTeX
    expr = expr.replace(r"\left", "")
    expr = expr.replace(r"\right", "")
    expr = expr.replace("{", "(").replace("}", ")")

    # moltiplicazioni implicite tra parentesi: )(  -> )*(
    expr = re.sub(r'\)\s*\(', r')*(', expr)
    # ( ... )y  -> ( ... )*y
    expr = re.sub(r'\)\s*y\b', r')*y', expr)

    # Inserisce parentesi automatiche: cos t → cos(t), exp -t → exp(-t)
    expr = re.sub(r'\b(sin|cos|tan|exp|log|sqrt)\s*\(?\s*([a-zA-Z0-9_\-\+]+)\s*\)?', r'\1(\2)', expr)

    # rimuove spazi
    expr = expr.replace(" ", "")

    # Elimina eventuali backslash residui SOLO alla fine (difensivo).
    # Qui dovrebbero rimanere solo backslash "inermi"; se resta qualche comando non riconosciuto,
    # preferiamo comunque rimuoverli per non far fallire parse_expr.
    expr = expr.replace('\\\\', '')
    expr = expr.replace('\\', '')

    # Moltiplicazione implicita numero-lettera: 1/2k -> 1/2*k, 2k -> 2*k
    expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)

    # Moltiplicazione implicita numero-parantesi: 2(k+1) -> 2*(k+1)
    expr = re.sub(r'(\d)\(', r'\1*(', expr)

    # Moltiplicazione implicita parentesi-lettera: )k -> )*k
    expr = re.sub(r'\)([a-zA-Z])', r')*\1', expr)

    return expr

# ------------------------------------------------------------
#  Utilities: delta discreto con ipotesi t>=0 (causale)
# ------------------------------------------------------------

def simplify_delta_t_nonneg(expr: sp.Expr, t_sym: sp.Symbol) -> sp.Expr:
    """Semplifica KroneckerDelta assumendo t intero e t>=0.

    Regole principali (coerenti con il corso):
      - KD(t, k) con k < 0  -> 0 (l'impulso sarebbe a tempo negativo)
      - KD(t + m, 0) con m>0 -> 0 (equivalente a KD(t, -m))
      - KD(t + m, k) -> KD(t, k-m) e poi applica la regola k-m < 0 -> 0

    Nota: questa NON è un'identità su Z intero, ma una semplificazione sul dominio t>=0.
    """

    expr = sp.expand(expr)

    def _offset_from_t(x: sp.Expr):
        x = sp.expand(x)
        if x == t_sym:
            return sp.Integer(0)
        if x.is_Add and t_sym in x.free_symbols:
            k = sp.simplify(x - t_sym)
            if k.is_Integer:
                return sp.Integer(k)
        return None

    def _as_intsym(x: sp.Expr):
        x = sp.expand(x)
        return sp.Integer(x) if x.is_Integer else None

    def _rule_kd(kd: sp.KroneckerDelta):
        a, b = kd.args

        # Normalizza l'ordine: vogliamo preferibilmente KD(t, k) (stile corso)
        if a.is_Integer and (b == t_sym):
            kd = sp.KroneckerDelta(t_sym, a, evaluate=False)
            a, b = kd.args
        elif b.is_Integer and (a == t_sym):
            kd = sp.KroneckerDelta(t_sym, b, evaluate=False)
            a, b = kd.args

        ka = _offset_from_t(a)
        kb = _offset_from_t(b)
        ma = _as_intsym(a)
        mb = _as_intsym(b)

        # KD(t, k) con k<0 -> 0
        if a == t_sym and mb is not None and mb < 0:
            return sp.Integer(0)
        if b == t_sym and ma is not None and ma < 0:
            return sp.Integer(0)

        # KD(t+m, 0) -> KD(t, -m) -> 0 se m>0
        if ka is not None and mb == 0:
            return sp.Integer(0) if ka > 0 else sp.KroneckerDelta(t_sym, 0, evaluate=False)
        if kb is not None and ma == 0:
            return sp.Integer(0) if kb > 0 else sp.KroneckerDelta(t_sym, 0, evaluate=False)

        # KD(t+m, k) -> KD(t, k-m)
        if ka is not None and mb is not None:
            idx = sp.simplify(mb - ka)
            return sp.Integer(0) if idx < 0 else sp.KroneckerDelta(t_sym, idx, evaluate=False)
        if kb is not None and ma is not None:
            idx = sp.simplify(ma - kb)
            return sp.Integer(0) if idx < 0 else sp.KroneckerDelta(t_sym, idx, evaluate=False)

        return kd

    return expr.replace(lambda e: isinstance(e, sp.KroneckerDelta), lambda e: _rule_kd(e))


def apply_forward_shift_poly_to(expr: sp.Expr, P_d: sp.Expr, t_sym: sp.Symbol, d_sym: sp.Symbol) -> sp.Expr:
    """Applica un polinomio P(d) (con d = shift in avanti) a expr.

    Convenzione: d^k f(t) = f(t+k).
    Supporta P(d) come somma di monomi in d.
    """
    P_d = sp.expand(P_d)
    res = 0
    for term in P_d.as_ordered_terms():
        coeff, monom = term.as_coeff_Mul()
        power = 0
        if monom == 1:
            power = 0
        elif monom == d_sym:
            power = 1
        elif isinstance(monom, sp.Pow) and monom.base == d_sym and monom.exp.is_Integer:
            power = int(monom.exp)
        else:
            # casi più complessi (es. coeff*d^k) li gestiamo provando a estrarre d^k
            if monom.has(d_sym):
                monom_poly = sp.Poly(monom, d_sym)
                if monom_poly.is_univariate and len(monom_poly.monoms()) == 1:
                    power = monom_poly.monoms()[0][0]
                else:
                    raise ValueError(f"Monomio non supportato in P(d): {monom}")
            else:
                power = 0
        res += coeff * expr.subs(t_sym, t_sym + power)

    return simplify_delta_t_nonneg(sp.expand(res), t_sym)


def get_annichilatore_discrete(rhs_expr: sp.Expr, t_sym: sp.Symbol, d_sym: sp.Symbol):
    """Lookup-table minimale per annichilatori discreti (t>=0).

    Attualmente implementa in modo robusto i casi impulsivi:
      - c*KD(t,k) con k>=0  ->  d^(k+1)
      - KD(t,k) con k<0     ->  1 (perché sul dominio t>=0 è già 0)

    Se rhs è somma, il totale è il prodotto dei singoli fattori.
    """

    rhs_expr = simplify_delta_t_nonneg(sp.expand(rhs_expr), t_sym)

    def ann_term(term: sp.Expr):
        # stacca un coefficiente numerico
        coeff, core = term.as_independent(t_sym, as_Add=False)
        term = core

        # -------------------------------------------------
        #  1) Impulsi: KD(t,k)
        # -------------------------------------------------
        if isinstance(term, sp.KroneckerDelta):
            a, b = term.args
            if a == t_sym and b.is_Integer:
                k = int(b)
                if k < 0:
                    return sp.Integer(1)
                return d_sym ** (k + 1)
            if b == t_sym and a.is_Integer:
                k = int(a)
                if k < 0:
                    return sp.Integer(1)
                return d_sym ** (k + 1)
            return None

        # -------------------------------------------------
        #  Helpers for patterns
        # -------------------------------------------------
        aW = sp.Wild('aW')
        nW = sp.Wild('nW')
        pW = sp.Wild('pW')
        wW = sp.Wild('wW')

        def _is_nonneg_int(x):
            return x.is_Integer and int(x) >= 0

        def _base_trig_poly(p_val, w_val):
            # d^2 - 2 p cos(w) d + p^2
            return d_sym**2 - 2*sp.simplify(p_val*sp.cos(w_val))*d_sym + sp.simplify(p_val**2)

        # -------------------------------------------------
        #  2) Esponenziali discrete: a**t e t^n a**t
        #     (Δ - a) e (Δ - a)^{n+1}
        # -------------------------------------------------
        m = term.match(aW**t_sym)
        if m and not m[aW].has(t_sym) and m[aW] != 0:
            return d_sym - m[aW]

        m = term.match((t_sym**nW) * (aW**t_sym))
        if m and _is_nonneg_int(m[nW]) and not m[aW].has(t_sym) and m[aW] != 0:
            return (d_sym - m[aW])**(int(m[nW]) + 1)

        # -------------------------------------------------
        #  3) Trigonometrici: cos(w t), sin(w t)
        #     (Δ^2 - 2 cos(w) Δ + 1)
        #     e versioni con p^t: p^t cos(w t), p^t sin(w t)
        # -------------------------------------------------
        m = term.match(sp.cos(wW*t_sym))
        if m and not m[wW].has(t_sym):
            return _base_trig_poly(sp.Integer(1), m[wW])

        m = term.match(sp.sin(wW*t_sym))
        if m and not m[wW].has(t_sym):
            return _base_trig_poly(sp.Integer(1), m[wW])

        m = term.match((pW**t_sym) * sp.cos(wW*t_sym))
        if m and not m[wW].has(t_sym) and not m[pW].has(t_sym) and m[pW] != 0:
            return _base_trig_poly(m[pW], m[wW])

        m = term.match((pW**t_sym) * sp.sin(wW*t_sym))
        if m and not m[wW].has(t_sym) and not m[pW].has(t_sym) and m[pW] != 0:
            return _base_trig_poly(m[pW], m[wW])

        # -------------------------------------------------
        #  4) Moltiplicazione per potenza di t: t^n * trig
        #     -> (base_poly)^{n+1}
        # -------------------------------------------------
        m = term.match((t_sym**nW) * sp.cos(wW*t_sym))
        if m and _is_nonneg_int(m[nW]) and not m[wW].has(t_sym):
            return (_base_trig_poly(sp.Integer(1), m[wW]))**(int(m[nW]) + 1)

        m = term.match((t_sym**nW) * sp.sin(wW*t_sym))
        if m and _is_nonneg_int(m[nW]) and not m[wW].has(t_sym):
            return (_base_trig_poly(sp.Integer(1), m[wW]))**(int(m[nW]) + 1)

        m = term.match((t_sym**nW) * (pW**t_sym) * sp.cos(wW*t_sym))
        if m and _is_nonneg_int(m[nW]) and not m[wW].has(t_sym) and not m[pW].has(t_sym) and m[pW] != 0:
            return (_base_trig_poly(m[pW], m[wW]))**(int(m[nW]) + 1)

        m = term.match((t_sym**nW) * (pW**t_sym) * sp.sin(wW*t_sym))
        if m and _is_nonneg_int(m[nW]) and not m[wW].has(t_sym) and not m[pW].has(t_sym) and m[pW] != 0:
            return (_base_trig_poly(m[pW], m[wW]))**(int(m[nW]) + 1)

        return None

    factors = []
    for addend in rhs_expr.as_ordered_terms():
        A_i = ann_term(addend)
        if A_i is None:
            return None, r"\text{Non determinato automaticamente}"
        factors.append(A_i)

    A_total = sp.Mul(*factors, evaluate=False)
    A_show = sp.simplify(A_total)
    latex_in_Delta = sp.latex(A_show).replace(str(d_sym), r"\Delta")
    return A_total, latex_in_Delta


# ------------------------------------------------------------
#  Solver (shift in avanti): P(Δ) y(t) = u(t),  t>=0
# ------------------------------------------------------------

def solve_difference_equation_shift(equation_str: str, condizioni_iniziali=None):
    """Endpoint solver per equazioni alle differenze con shift in avanti.

    Questa versione è pensata per backend Flask:
      - normalizza input LaTeX-like
      - parse LHS/RHS in SymPy
      - estrae P(d) (polinomio nell'operatore shift)
      - semplifica forzanti impulsive usando t>=0
      - calcola annichilatore discreto (lookup) per impulsi

    Nota: qui NON usiamo rsolve/solve (che spesso falliscono con KroneckerDelta).
    Il risultato restituisce gli step LaTeX e i pezzi simbolici utili.
    """

    if condizioni_iniziali is None:
        condizioni_iniziali = []

    latex_steps = []

    def build_homogeneous_solution(P_d_expr: sp.Expr, t_sym: sp.Symbol, d_sym: sp.Symbol) -> sp.Expr:
        """Costruisce y_o(t) per P(d) y = 0 con d = shift in avanti.

        - Per il fattore d^m: genera impulsi c_0*KD(t,0)+...+c_{m-1}*KD(t,m-1)
          (coerente con dominio t>=0: d^m y=0 => y(t)=0 per t>=m, ma i primi m campioni sono liberi).
        - Per radici non nulle λ con molteplicità r: somma_{j=0}^{r-1} c * t^j * λ^t
        """
        P_poly = sp.Poly(sp.expand(P_d_expr), d_sym)
        if P_poly.is_zero:
            return sp.Integer(0)

        # Molteplicità del fattore d
        m0 = 0
        tmp = P_poly
        while tmp.degree() > 0:
            rem = tmp.rem(sp.Poly(d_sym, d_sym))
            if rem.is_zero:
                m0 += 1
                tmp = sp.Poly(tmp.as_expr() / d_sym, d_sym)
            else:
                break

        terms = []
        c_index = 0

        # Parte impulsiva (d^m0)
        for k in range(m0):
            c = sp.Symbol(f"c_{c_index}")
            c_index += 1
            terms.append(c * sp.KroneckerDelta(t_sym, k, evaluate=False))

        # Parte esponenziale dalle radici non nulle
        Q_expr = sp.expand(P_d_expr / (d_sym ** m0)) if m0 > 0 else sp.expand(P_d_expr)
        Q_poly = sp.Poly(Q_expr, d_sym)
        if Q_poly.is_zero:
            return sp.Add(*terms) if terms else sp.Integer(0)

        # SymPy compatibility: in some versions Poly has no .roots(); use sympy.roots on the expression.
        roots_dict = sp.roots(Q_poly.as_expr(), d_sym)  # dict: root -> multiplicity

        # Fallback: if symbolic roots are not available, try algebraic roots list.
        if not roots_dict:
            try:
                all_roots = Q_poly.all_roots()  # may return RootOf objects
                roots_dict = {}
                for r in all_roots:
                    roots_dict[r] = roots_dict.get(r, 0) + 1
            except Exception:
                roots_dict = {}

        # --------------------------------------------------
        #  Forma reale trigonometrica per coppie complesse coniugate
        #    se lam = ρ e^{iω} => ρ^t cos(ω t), ρ^t sin(ω t)
        #  (solo estetica; soluzione equivalente in R)
        # --------------------------------------------------
        unused = dict(roots_dict)  # root -> multiplicity

        def _is_real(x: sp.Expr) -> bool:
            # prova veloce: immaginaria identicamente 0
            try:
                return sp.simplify(sp.im(x)) == 0
            except Exception:
                return False

        while unused:
            lam, mult = next(iter(unused.items()))
            del unused[lam]
            mult = int(mult)

            # radice reale: usa lam^t come prima
            if _is_real(lam):
                for j in range(mult):
                    c = sp.Symbol(f"c_{c_index}")
                    c_index += 1
                    terms.append(c * (t_sym ** j) * (lam ** t_sym))
                continue

            # radice complessa: cerca il coniugato
            lam_conj = sp.conjugate(lam)
            mult_conj = unused.get(lam_conj, None)

            # Se non troviamo il coniugato (o molteplicità diversa), fallback alla forma complessa
            if mult_conj is None or int(mult_conj) != mult:
                for j in range(mult):
                    c = sp.Symbol(f"c_{c_index}")
                    c_index += 1
                    terms.append(c * (t_sym ** j) * (lam ** t_sym))
                continue

            # consuma anche il coniugato
            del unused[lam_conj]

            rho = sp.simplify(sp.Abs(lam))
            omega = sp.simplify(sp.arg(lam))

            # genera 2*mult termini reali: t^j rho^t cos(omega t) e t^j rho^t sin(omega t)
            for j in range(mult):
                c_cos = sp.Symbol(f"c_{c_index}")
                c_index += 1
                c_sin = sp.Symbol(f"c_{c_index}")
                c_index += 1

                base = (t_sym ** j) * (rho ** t_sym)
                terms.append(c_cos * base * sp.cos(omega * t_sym))
                terms.append(c_sin * base * sp.sin(omega * t_sym))

        # (opzionale) semplifica i termini per compattezza
        terms = [sp.simplify(sp.expand_mul(tt)) for tt in terms]
        return sp.Add(*terms) if terms else sp.Integer(0)

    def latex_with_discrete_delta(expr: sp.Expr) -> str:
        r"""Renderizza KroneckerDelta(t,k) come \delta_0(t-k) (notazione corso).

        Implementazione robusta: sostituisce KD(t,k) / KD(k,t) con placeholder simbolici,
        genera LaTeX, poi rimpiazza i placeholder con \delta_0(t-k).
        """
        expr = sp.expand(expr)

        repl = {}
        # crea placeholder unici per ogni k trovato
        for kd in expr.atoms(sp.KroneckerDelta):
            a, b = kd.args
            k = None
            if a == t and b.is_Integer:
                k = int(b)
            elif b == t and a.is_Integer:
                k = int(a)
            if k is not None:
                repl[kd] = sp.Symbol(f"__KD_{k}__")

        tmp = expr.xreplace(repl) if repl else expr
        s = sp.latex(tmp)

        # rimpiazza i placeholder con delta_0(t-k) (notazione corso)
        for kd_obj, sym in repl.items():
            name = sym.name
            m = re.match(r"__KD_([\-0-9]+)__", name)
            if not m:
                continue
            k = m.group(1)
            # Notazione corso: KD(t,k) = \delta_0(t-k)
            kk = int(k)
            if kk == 0:
                repl_tex = r"\delta_0(t)"
            elif kk > 0:
                repl_tex = rf"\delta_0(t-{kk})"
            else:
                repl_tex = rf"\delta_0(t+{-kk})"

            s = s.replace(sp.latex(sym), repl_tex)

        # Cosmetic: add thin space between numeric coefficients and deltas (for MathJax readability)
        s = re.sub(r"(\d)(\\delta_0)", r"\1\\,\\delta_0", s)
        s = re.sub(r"\)(\\delta_0)", r")\\,\\delta_0", s)
        return s

    def build_particular_impulse_solution(P_d_expr: sp.Expr, u_expr: sp.Expr, t_sym: sp.Symbol, d_sym: sp.Symbol) -> sp.Expr | None:
        """Costruisce una soluzione particolare 'causale a supporto finito' quando u(t) è impulsiva.

        Metodo robusto (stile corso, t>=0):
        1) Fattorizza P(d) = d^m Q(d) con Q(0) != 0.
        2) Definisce z(t) = y(t+m)  =>  Q(d) z(t) = u(t).
        3) Impone condizione di coda z(t)=0 per t > T (supporto finito) e risolve all'indietro:
              q0 z(t) + q1 z(t+1) + ... + qn z(t+n) = u(t)
           quindi:
              z(t) = (u(t) - sum_{i>=1} qi z(t+i)) / q0
        4) Ricostruisce y(t) = z(t-m) => in termini di impulsi KD(t, k+m).

        Nota: esistono infinite particolari; questa scelta riproduce la particolare 'a impulsi' tipica del corso.
        """

        P_expr = sp.expand(P_d_expr)
        P_poly = sp.Poly(P_expr, d_sym)
        if P_poly.is_zero:
            return None

        # u(t) deve essere combinazione di impulsi
        u_s = simplify_delta_t_nonneg(sp.expand(u_expr), t_sym)

        impulses = []  # list of (k, coeff)
        for term in u_s.as_ordered_terms():
            coeff, core = term.as_coeff_Mul()
            if isinstance(core, sp.KroneckerDelta):
                a, b = core.args
                if a == t_sym and b.is_Integer:
                    impulses.append((int(b), coeff))
                elif b == t_sym and a.is_Integer:
                    impulses.append((int(a), coeff))
            elif isinstance(term, sp.KroneckerDelta):
                a, b = term.args
                if a == t_sym and b.is_Integer:
                    impulses.append((int(b), sp.Integer(1)))
                elif b == t_sym and a.is_Integer:
                    impulses.append((int(a), sp.Integer(1)))

        if not impulses:
            return None

        k_max = max(k for k, _ in impulses)

        # m = ordine del fattore d (valutazione a 0): P(d)=d^m Q(d), Q(0)!=0
        m = 0
        tmp = P_poly
        while tmp.degree() > 0:
            rem = tmp.rem(sp.Poly(d_sym, d_sym))
            if rem.is_zero:
                m += 1
                tmp = sp.Poly(tmp.as_expr() / d_sym, d_sym)
            else:
                break

        Q_expr = sp.expand(P_expr / (d_sym ** m)) if m > 0 else P_expr
        Q_poly = sp.Poly(Q_expr, d_sym)
        if Q_poly.is_zero:
            return None

        # Coefficienti q_i di Q(d)=sum_{i=0..n} q_i d^i
        n = int(Q_poly.degree())
        q = [sp.Integer(0)] * (n + 1)
        for i in range(n + 1):
            q[i] = sp.simplify(Q_poly.nth(i))

        q0 = q[0]
        if q0 == 0:
            # Non dovrebbe accadere se abbiamo rimosso tutti i fattori d
            return None

        # definisci u_val(t) per t intero >=0
        u_map = {}
        for k, c in impulses:
            u_map[k] = sp.simplify(u_map.get(k, 0) + c)

        def u_at(tt: int) -> sp.Expr:
            return u_map.get(tt, sp.Integer(0))

        # cutoff: oltre T imponiamo z(t)=0
        T = k_max + n + 2

        z_vals = {tt: sp.Integer(0) for tt in range(T + n + 5)}

        # risoluzione all'indietro per z(t) da T a 0
        for tt in range(T, -1, -1):
            tail = sp.Integer(0)
            for i in range(1, n + 1):
                tail += q[i] * z_vals.get(tt + i, sp.Integer(0))
            z_vals[tt] = sp.simplify((u_at(tt) - tail) / q0)

        # costruisci y_p(t) = z(t-m) -> impulsi a indice (tt+m)
        terms = []
        for tt in range(0, T + 1):
            val = sp.simplify(z_vals.get(tt, 0))
            if val == 0:
                continue
            idx = tt + m
            terms.append(val * sp.KroneckerDelta(t_sym, idx, evaluate=False))

        if not terms:
            return sp.Integer(0)

        return sp.Add(*terms)

    # simboli locali (IMPORTANTE: non usare globali in un server)
    t = sp.Symbol("t", integer=True, nonnegative=True)
    d = sp.Symbol("d")  # operatore shift in avanti

    transformations = standard_transformations + (implicit_multiplication_application,)

    if not isinstance(equation_str, str) or not equation_str.strip() or "=" not in equation_str:
        return {"success": False, "error": "Equazione non valida o mancante '='."}

    print("[DEBUG][diff] Raw equation:", equation_str)

    lhs_str, rhs_str = equation_str.split("=", 1)

    lhs_norm = normalize_latex_like_discrete(lhs_str)
    rhs_norm = normalize_latex_like_discrete(rhs_str)

    print("[DEBUG][diff] lhs_norm =", lhs_norm)
    print("[DEBUG][diff] rhs_norm =", rhs_norm)

    local_dict = {
        "t": t,
        "d": d,
        "Delta": d,      # tollera eventuale 'Delta'
        "KroneckerDelta": sp.KroneckerDelta,
        "pi": sp.pi,
        "sin": sp.sin,
        "cos": sp.cos,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
    }

    try:
        lhs_expr = parse_expr(lhs_norm, transformations=transformations, local_dict=local_dict)
        rhs_expr = parse_expr(rhs_norm, transformations=transformations, local_dict=local_dict)
    except Exception as e:
        return {
            "success": False,
            "error": f"Parse error: {str(e)}",
            "debug": {"lhs_norm": lhs_norm, "rhs_norm": rhs_norm},
        }

    print("[DEBUG][diff] LHS parsed:", lhs_expr)
    print("[DEBUG][diff] RHS parsed (raw):", rhs_expr)

    rhs_expr = simplify_delta_t_nonneg(rhs_expr, t)
    rhs_expr = simplify_delta_t_nonneg(rhs_expr, t)  # normalizza anche ordine KD
    print("[DEBUG][diff] RHS parsed (simplified t>=0):", rhs_expr)

    # Estrai P(d) dal LHS rimuovendo y (se presente come simbolo letterale)
    # In questa route discreta ci aspettiamo che il frontend invii già P(d)*y oppure solo P(d).
    y_sym = sp.Symbol("y")
    P_d = lhs_expr
    P_d = P_d.replace(lambda e: e == y_sym, lambda e: 1)
    P_d = sp.expand(P_d)
    print("[DEBUG][diff] P(d) extracted:", P_d)
    # Ordine dell'equazione alle differenze: massimo shift (grado in d)
    ordine = int(sp.Poly(P_d, d).degree()) if not sp.Poly(P_d, d).is_zero else 0

    # Se l'utente ha scritto qualcosa tipo (d-1)(d+1)y, allora P_d conterrà ancora y (se non era simbolo)
    # Difensivo: se resta una AppliedUndef o Function, segnala.
    if any(isinstance(a, AppliedUndef) for a in P_d.atoms(AppliedUndef)):
        return {"success": False, "error": "Per ora usa y come simbolo (non y(t))."}

    # --------------------------------------------------
    # Soluzione dell'equazione omogenea: P(Δ) y(t) = 0
    # --------------------------------------------------
    y_o = build_homogeneous_solution(P_d, t, d)
    print("[DEBUG][diff] y_o(t) =", y_o)

    latex_steps.append({
        "title": "Soluzione dell'equazione omogenea:",
        "content": rf"y_{{o}}(t) = {latex_with_discrete_delta(y_o)}"
    })

    # Annichilatore discreto (lookup) per impulsi
    A_d, A_latex = get_annichilatore_discrete(rhs_expr, t, d)
    print("[DEBUG][diff] A(d) =", A_d)
    if A_d is None:
        latex_steps.append({
            "title": r"Annichilatore \(A(\Delta)\):",
            "content": r"\text{Non determinato automaticamente (lookup impulsi)}",
        })
    else:
        latex_steps.append({
            "title": r"Annichilatore \(A(\Delta)\) tale che \(A(\Delta)u(t)=0\) su \(t\ge 0\):",
            "content": rf"A(\Delta) = {A_latex}",
        })

        full_poly = sp.expand(A_d * P_d)
        # Mostra sempre fattorizzata (su R): (d+1)^2*(d^2+4) ecc.
        full_poly_fact = sp.factor(full_poly)
        full_poly_latex = sp.latex(full_poly_fact).replace("d", r"\Delta")
        latex_steps.append({
            "title": "Equazione omogenea estesa:",
            "content": rf"\left({full_poly_latex}\right) y(t)=0",
        })
        # --------------------------------------------------
        # Soluzione dell'omogenea estesa: (A(Δ)P(Δ)) y(t) = 0
        # --------------------------------------------------
        y_o_e = build_homogeneous_solution(full_poly, t, d)
        print("[DEBUG][diff] y_o,e(t) =", y_o_e)
        latex_steps.append({
            "title": "Soluzione dell'equazione omogenea estesa:",
            "content": rf"y_{{o,e}}(t) = {latex_with_discrete_delta(y_o_e)}"
        })

    def build_particular_from_extended(y_o_expr: sp.Expr, y_oe_expr: sp.Expr):
        """Per u(t) non impulsiva: y_p sta nello span dei modi aggiunti da A(Δ).

        Ritorna una terna (y_p_ansatz, alpha_syms, cores) dove:
          - y_p_ansatz = sum alpha_i * core_i
          - cores_i sono i modi nuovi (senza coefficienti)
        """
        y_o_expr = sp.expand(y_o_expr)
        y_oe_expr = sp.expand(y_oe_expr)

        def basis_cores(expr: sp.Expr):
            out = []
            for term in sp.Add.make_args(expr):
                syms = [s for s in term.free_symbols if s.name.startswith('c_')]
                if syms:
                    c = sorted(syms, key=lambda s: s.name)[0]
                    core = sp.simplify(term / c)
                else:
                    core = sp.simplify(term)
                out.append(core)
            return out

        B_o = basis_cores(y_o_expr)
        B_oe = basis_cores(y_oe_expr)

        def in_basis(x, basis):
            for b in basis:
                try:
                    if sp.simplify(x - b) == 0:
                        return True
                except Exception:
                    pass
            return False

        cores = [core for core in B_oe if not in_basis(core, B_o)]
        if not cores:
            return sp.Integer(0), [], []

        alpha_syms = list(sp.symbols('alpha0:' + str(len(cores))))
        y_ansatz = sp.Add(*[alpha_syms[i] * cores[i] for i in range(len(cores))])
        return y_ansatz, alpha_syms, cores

    def solve_particular_coeffs_by_sampling(y_ansatz: sp.Expr, alpha_syms, rhs_u: sp.Expr, P_poly: sp.Expr, t_sym: sp.Symbol, d_sym: sp.Symbol):
        """Risolve i coefficienti dell'ansatz imponendo P(d) y_p = u(t) su alcuni campioni t=0..K.

        Funziona bene per forzanti di tipo tabellare (a^t, trig, polinomi*trig, ecc.).
        """
        if not alpha_syms:
            return {}, y_ansatz

        # calcola residuo simbolico
        lhs = apply_forward_shift_poly_to(y_ansatz, P_poly, t_sym, d_sym)
        lhs = simplify_delta_t_nonneg(sp.expand(lhs), t_sym)
        rhs_u = simplify_delta_t_nonneg(sp.expand(rhs_u), t_sym)
        resid = sp.simplify(lhs - rhs_u)

        # genera sistema lineare su alpha valutando t=0..K
        m = len(alpha_syms)
        K = max(8, m + 4)
        eqs = []
        for k in range(K):
            eqs.append(sp.Eq(sp.simplify(resid.subs(t_sym, k)), 0))

        try:
            A_mat, b_vec = sp.linear_eq_to_matrix([eq.lhs for eq in eqs], alpha_syms)
            sol_set = sp.linsolve((A_mat, b_vec), *alpha_syms)
            if not sol_set:
                return None, None
            sol_tuple = list(sol_set)[0]
            sol = {alpha_syms[i]: sp.simplify(sol_tuple[i]) for i in range(m)}
            return sol, sp.simplify(y_ansatz.subs(sol))
        except Exception:
            # fallback: direct solve
            sol = sp.solve([eq.lhs for eq in eqs], alpha_syms, dict=True)
            if not sol:
                return None, None
            sol0 = sol[0]
            return sol0, sp.simplify(y_ansatz.subs(sol0))

    # --------------------------------------------------
    # Soluzione particolare (impulsiva o extended-homogeneous)
    # --------------------------------------------------
    y_p = build_particular_impulse_solution(P_d, rhs_expr, t, d)

    # Se non è impulsiva e abbiamo A(Δ), ricava i modi nuovi e risolvi i coefficienti (nessun alpha in output)
    if y_p is None and A_d is not None:
        try:
            y_ansatz, alpha_syms, cores = build_particular_from_extended(y_o, y_o_e)
            sol, y_p_solved = solve_particular_coeffs_by_sampling(y_ansatz, alpha_syms, rhs_expr, P_d, t, d)
            y_p = y_p_solved
        except Exception:
            y_p = None

    print("[DEBUG][diff] y_p(t) =", y_p)
    if y_p is None:
        latex_steps.append({
            "title": "Soluzione particolare:",
            "content": r"\text{Non determinabile automaticamente}" 
        })
    else:
        latex_steps.append({
            "title": "Soluzione particolare:",
            "content": rf"y_p(t) = {latex_with_discrete_delta(sp.simplify(y_p))}"
        })
    # --------------------------------------------------
    # Soluzione generale (senza CI): y_g(t) = y_o(t) + y_p(t)
    # --------------------------------------------------
    y_g = sp.simplify(sp.expand(y_o + (y_p if y_p is not None else 0)))
    print("[DEBUG][diff] y_g(t) =", y_g)
    latex_steps.append({
        "title": "Soluzione generale (senza condizioni iniziali):",
        "content": rf"y_{{g}}(t) = {latex_with_discrete_delta(y_g)}"
    })

    # Debug utile: applica P(d) alla particolare candidata se il client la manda
    # (opzionale)  payload: {"candidate": "-3*KroneckerDelta(t,2)"}

    # --------------------------------------------------
    # APPLICAZIONE CONDIZIONI INIZIALI (se presenti)
    # --------------------------------------------------
    if condizioni_iniziali:
        if ordine == 0:
            return {"success": False, "error": "Equazione di ordine 0: nessuna CI necessaria."}

        if len(condizioni_iniziali) != ordine:
            return {
                "success": False,
                "error": f"Servono {ordine} condizioni iniziali (y(0)..y({ordine-1})), ne sono state fornite {len(condizioni_iniziali)}.",
            }

        # Costanti libere c_i presenti nella soluzione generale
        c_syms = sorted([s for s in y_g.free_symbols if s.name.startswith('c_')], key=lambda s: s.name)

        # Valori delle CI: possono essere numerici o simbolici.
        # Regola: se una CI è una stringa ma rappresenta un numero/espressione numerica (es. "1", "-1/2", "sqrt(2)"),
        # la trattiamo come NUMERICA. Solo stringhe con simboli (es. "y_0") attivano la modalità canonica.
        ci_vals = []
        ci_syms = []
        any_symbolic = False

        def _parse_ci(ci_raw):
            if isinstance(ci_raw, str):
                try:
                    v = sp.sympify(ci_raw)
                    # se non contiene simboli liberi, è numerica
                    if len(v.free_symbols) == 0:
                        return v, False
                    return v, True
                except Exception:
                    # fallback: considerala simbolica
                    return sp.Symbol(ci_raw), True
            else:
                v = sp.sympify(ci_raw)
                return v, (len(v.free_symbols) != 0)

        for ci in condizioni_iniziali:
            v, is_symb = _parse_ci(ci)
            if is_symb:
                any_symbolic = True
                # registra solo i simboli 'canonici' delle CI
                for s in sorted(v.free_symbols, key=lambda s: s.name):
                    if s not in ci_syms:
                        ci_syms.append(s)
            ci_vals.append(v)

        # Sistema: y_g(i) = CI_i per i=0..ordine-1
        eqs = []
        for i in range(ordine):
            eqs.append(sp.Eq(sp.simplify(y_g.subs(t, i)), ci_vals[i]))

        sol_list = sp.solve([eq.lhs - eq.rhs for eq in eqs], c_syms, dict=True)
        if not sol_list:
            return {
                "success": False,
                "error": "Impossibile determinare le costanti dalle condizioni iniziali fornite.",
            }

        sol_c = sol_list[0]

        # Costruisci la soluzione finale
        y_ci = sp.simplify(sp.expand(y_g.subs(sol_c)))

        # Step: sistema (non serve ridire le CI; mostra solo le equazioni in c_i)
        # Riscrivi le equazioni sostituendo i valori numerici/simbolici a destra
        latex_steps.append({
            "title": "Sistema per determinare le costanti:",
            "content": r"\begin{cases}" + r"\\ ".join(latex(eq) for eq in eqs) + r"\end{cases}",
        })

        if any_symbolic:
            # --------------------------------------------------
            # Decomposizione corretta (stile corso):
            #   - risposta libera  = termini che dipendono dalle CI (y_0, y_1, ...)
            #   - risposta forzata = tutti gli altri termini (dipendono solo da t e dai parametri del sistema)
            # Nota: NON coincide in generale con y_p.
            # --------------------------------------------------
            y_ci_exp = sp.expand(y_ci)
            terms_ci = sp.Add.make_args(y_ci_exp)

            parte_libera = sp.Integer(0)
            for term in terms_ci:
                if any(sym in term.free_symbols for sym in ci_syms):
                    parte_libera += term

            parte_forzata = sp.simplify(y_ci_exp - parte_libera)
            parte_libera = sp.simplify(parte_libera)

            latex_steps.append({
                "title": "Risposta forzata:",
                "content": rf"y_{{f}}(t) = {latex_with_discrete_delta(parte_forzata)}",
            })
            latex_steps.append({
                "title": "Risposta libera:",
                "content": rf"y_{{l}}(t) = {latex_with_discrete_delta(parte_libera)}",
            })
        else:
            latex_steps.append({
                "title": "Soluzione con condizioni iniziali:",
                "content": rf"y(t) = {latex_with_discrete_delta(y_ci)}",
            })

    return {
        "success": True,
        "latex": latex_steps,
        "debug": {
            "lhs_norm": lhs_norm,
            "rhs_norm": rhs_norm,
            "P_d": str(P_d),
            "rhs_simplified": str(rhs_expr),
        },
    }


# ------------------------------------------------------------
#  Route Flask
# ------------------------------------------------------------
@equazioni_differenze_bp.route("/api/equazioni_differenze", methods=["POST"])
def api_equazioni_differenze():
    data = request.get_json() or {}
    equation = data.get("equazione", "")
    condizioni = data.get("condizioniIniziali", [])

    result = solve_difference_equation_shift(equation, condizioni_iniziali=condizioni)
    return jsonify(result)
