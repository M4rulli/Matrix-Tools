from flask import Blueprint, request, jsonify
import sympy as sp
from ast import literal_eval
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application, parse_expr
from sympy.parsing.sympy_parser import convert_xor

# Optional dependency: Lcapy (better Z-transform support than SymPy alone)
try:
    from lcapy import expr as lcapy_expr
    _LCAPY_AVAILABLE = True
except Exception:
    lcapy_expr = None
    _LCAPY_AVAILABLE = False

def normalize_latex_like(expr: str) -> str:
    r"""Normalize a controlled LaTeX-like input into a SymPy/Lcapy-friendly string.

    Supports: \sin, \cos, \tan, \exp, \ln/\log, \pi (and π), \cdot, ^,
    \left/\right, { }, \frac{a}{b}.

    Also fixes common implicit multiplication (e.g., 2t -> 2*t, 2k -> 2*k).
    """
    import re

    if not isinstance(expr, str):
        return expr

    s = expr

    # Basic function names (LaTeX -> SymPy)
    replacements = {
        r"\sin": "sin",
        r"\cos": "cos",
        r"\tan": "tan",
        r"\exp": "exp",
        r"\ln": "log",
        r"\log": "log",
        r"\pi": "pi",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)

    # LaTeX fractions: \frac{a}{b} -> (a)/(b)
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)

    # Compact fractions: \frac12 -> 1/2
    s = re.sub(r"\\frac\s*([0-9]+)\s*([0-9]+)", r"\1/\2", s)

    # Unicode pi
    s = s.replace("π", "pi")

    # Operators
    s = s.replace(r"\cdot", "*")
    s = s.replace("^", "**")

    # Remove LaTeX sizing parentheses
    s = s.replace(r"\left", "")
    s = s.replace(r"\right", "")

    # Convert braces to parentheses
    s = s.replace("{", "(").replace("}", ")")

    # Remove spaces
    s = s.replace(" ", "")

    # Ensure pi multiplies variables: pi t -> pi*t, pi(t) -> pi*(t)
    s = re.sub(r"\bpi(?=[a-zA-Z(])", "pi*", s)

    # implicit multiplication: number-letter (2t -> 2*t, 3pi -> 3*pi)
    s = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", s)

    # implicit multiplication: )( -> )*(
    s = re.sub(r"\)\(", r")*(", s)

    # implicit multiplication: )x -> )*x
    s = re.sub(r"\)([a-zA-Z])", r")*\1", s)

    return s

sistemi_dinamici_bp = Blueprint("sistemi_dinamici", __name__)

@sistemi_dinamici_bp.route("/api/forced-output", methods=["POST"])
@sistemi_dinamici_bp.route("/api/uscita_forzata", methods=["POST"])
def compute_output_y():
    t, s = sp.symbols("t s")
    A = sp.Matrix(request.json["A"])
    B = sp.Matrix(request.json["B"])
    C = sp.Matrix(request.json["C"]).reshape(1, A.shape[0])
    D = sp.sympify(request.json["D"])
    u_expr_raw = request.json.get("u", "0")
    u_expr_raw = str(u_expr_raw) if u_expr_raw is not None else "0"
    u_expr_raw = u_expr_raw.strip() if u_expr_raw.strip() else "0"

    print("[DEBUG][laplace-y] Payload keys:", list(request.json.keys()))
    print("[DEBUG][laplace-y] u_expr raw:", repr(u_expr_raw))

    u_expr = normalize_latex_like(u_expr_raw)
    print("[DEBUG][laplace-y] u_expr normalized:", repr(u_expr))

    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    try:
        u = parse_expr(
            u_expr,
            local_dict={"t": t, "pi": sp.pi, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log},
            transformations=transformations
        )
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Errore di parsing dell'ingresso u(t): {str(e)}",
            "debug": {"u_raw": u_expr_raw, "u_normalized": u_expr}
        }), 400

    # Step 1: U(s) = Laplace{u(t)}
    U_s = sp.laplace_transform(u, t, s, noconds=True)
    if isinstance(U_s, tuple):
        U_s = U_s[0]

    # Step 2: H(s) = C (sI - A)^(-1)
    I = sp.eye(A.shape[0])
    H_s = (C * (s * I - A).inv()).reshape(1, A.shape[0])

    # Step 3: W(s) = H(s)B + D
    W_s = H_s * B + sp.Matrix([[D]])

    # Step 4: y(s) = H(s)x0 + W(s)U(s) — only W(s)U(s) part for now, x0 will be zero
    y_s = (W_s * U_s)[0].expand()

    latex_steps = [
        {
            "title": "Trasformata dell'ingresso:",
            "content": f"U(s) = \\mathcal{{L}}\\{{u(t)\\}} = {sp.latex(U_s)}"
        },
        {
            "title": "Calcolo della funzione di trasferimento:",
            "content": f"H(s) =C (sI - A)^{{-1}} = {sp.latex(H_s.applyfunc(sp.factor))}"
        },
        {
            "title": "Calcolo della funzione d'ingresso moltiplicata:",
            "content": f"W(s) = H(s) B + D = {sp.latex(sp.factor((H_s * B)[0]))} + {sp.latex(D)}"
        },
        {
            "title": "Calcolo dell'uscita nel dominio di Laplace:",
            "content": f"\\text{{Poiché }} x_0 = 0, \\quad y(s) = W(s) U(s) = {' + '.join([sp.latex(sp.factor(term)) for term in y_s.as_ordered_terms()])}"
        },
        {
            "title": "Inversione della trasformata di Laplace:",
            "content": f"y(t) = \\mathcal{{L}}^{{-1}}\\{{y(s)\\}} = {sp.latex(sp.inverse_laplace_transform(y_s, s, t).subs(sp.Heaviside(t), 1).subs(sp.Heaviside(t - 0), 1).expand())}"
        }
    ]
    return jsonify({"latex": latex_steps})


# New route per state forced
@sistemi_dinamici_bp.route("/api/forced-state", methods=["POST"])
@sistemi_dinamici_bp.route("/api/stato_forzato", methods=["POST"])
def compute_state_x():
    t, s = sp.symbols("t s")

    A = sp.Matrix(request.json["A"])
    B = sp.Matrix(request.json["B"])

    # u(t)
    u_expr_raw = request.json.get("u", "0")
    u_expr_raw = str(u_expr_raw) if u_expr_raw is not None else "0"
    u_expr_raw = u_expr_raw.strip() if u_expr_raw.strip() else "0"

    print("[DEBUG][laplace-x] Payload keys:", list(request.json.keys()))
    print("[DEBUG][laplace-x] u_expr raw:", repr(u_expr_raw))

    u_expr = normalize_latex_like(u_expr_raw)
    print("[DEBUG][laplace-x] u_expr normalized:", repr(u_expr))

    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    try:
        u = parse_expr(
            u_expr,
            local_dict={"t": t, "pi": sp.pi, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log},
            transformations=transformations
        )
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Errore di parsing dell'ingresso u(t): {str(e)}",
            "debug": {"u_raw": u_expr_raw, "u_normalized": u_expr}
        }), 400

    n = A.shape[0]

    # x0 (optional)
    x0_raw = request.json.get("x0", None)
    if x0_raw is None or (isinstance(x0_raw, str) and not x0_raw.strip()):
        x0 = sp.zeros(n, 1)
    else:
        try:
            if isinstance(x0_raw, str):
                try:
                    x0_val = literal_eval(x0_raw)
                    x0 = sp.Matrix(x0_val)
                except Exception:
                    x0 = sp.Matrix(sp.sympify(x0_raw))
            else:
                x0 = sp.Matrix(x0_raw)

            # allow row vector input
            if x0.shape == (1, n):
                x0 = x0.T
            # allow flat list [a,b,c]
            if x0.shape == (n,):
                x0 = sp.Matrix(list(x0)).reshape(n, 1)

            if x0.shape != (n, 1):
                return jsonify({
                    "success": False,
                    "error": f"x0 deve essere un vettore colonna di dimensione {n} (ricevuto {x0.shape})."
                }), 400
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"x0 non valido: {str(e)}"
            }), 400

    # Step 1: U(s) = Laplace{u(t)}
    U_s = sp.laplace_transform(u, t, s, noconds=True)
    if isinstance(U_s, tuple):
        U_s = U_s[0]

    # Step 2: Phi(s) = (sI - A)^(-1)
    I = sp.eye(n)
    Phi_s = (s * I - A).inv()

    # Step 3: X(s) = (sI-A)^(-1)x0 + (sI-A)^(-1)B U(s)
    X_free_s = (Phi_s * x0)
    X_forced_s = (Phi_s * B) * U_s
    X_s = (X_free_s + X_forced_s)

    def inv_laplace_scalar(expr):
        return sp.inverse_laplace_transform(expr, s, t, noconds=True)\
            .subs(sp.Heaviside(t), 1).subs(sp.Heaviside(t - 0), 1).expand()

    x_t = sp.Matrix([inv_laplace_scalar(X_s[i, 0]) for i in range(n)])

    latex_steps = [
        {
            "title": "Trasformata dell'ingresso:",
            "content": f"U(s) = \\mathcal{{L}}\\{{u(t)\\}} = {sp.latex(U_s)}"
        },
        {
            "title": "Calcolo della matrice risolvente:",
            "content": f"(sI - A)^{{-1}} = {sp.latex(Phi_s.applyfunc(sp.factor))}"
        },
        {
            "title": "Calcolo dello stato nel dominio di Laplace:",
            "content": (
                "X(s) = (sI-A)^{-1}x_0 + (sI-A)^{-1}B\\,U(s) = "
                f"{sp.latex(X_free_s.applyfunc(sp.factor))} + {sp.latex(X_forced_s.applyfunc(sp.factor))}"
            )
        },
        {
            "title": "Inversione della trasformata di Laplace:",
            "content": f"x(t) = \\mathcal{{L}}^{{-1}}\\{{X(s)\\}} = {sp.latex(x_t)}"
        }
    ]

    return jsonify({"latex": latex_steps})


# New route: output nel domain Z (only y(z), no inverse transform)
@sistemi_dinamici_bp.route("/api/forced-output-z", methods=["POST"])
@sistemi_dinamici_bp.route("/api/uscita_forzata_z", methods=["POST"])
def compute_output_y_z():
    k, z = sp.symbols("k z", integer=True, nonnegative=True)

    A = sp.Matrix(request.json["A"])
    B = sp.Matrix(request.json["B"])
    C = sp.Matrix(request.json["C"]).reshape(1, A.shape[0])
    D = sp.sympify(request.json.get("D", 0))

    # u(k)
    u_expr_raw = request.json.get("u", "0")
    u_expr_raw = str(u_expr_raw) if u_expr_raw is not None else "0"
    u_expr_raw = u_expr_raw.strip() if u_expr_raw.strip() else "0"

    print("[DEBUG][z] Payload keys:", list(request.json.keys()))
    print("[DEBUG][z] u_expr raw:", repr(u_expr_raw))

    u_expr = normalize_latex_like(u_expr_raw)
    print("[DEBUG][z] u_expr normalized:", repr(u_expr))

    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    try:
        u = parse_expr(
            u_expr,
            local_dict={"k": k, "t": k, "pi": sp.pi, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log},
            transformations=transformations
        )
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Errore di parsing dell'ingresso u(k): {str(e)}",
            "debug": {
                "u_raw": u_expr_raw,
                "u_normalized": u_expr
            }
        }), 400

    n = A.shape[0]

    # x0 (optional)
    x0_raw = request.json.get("x0", None)
    if x0_raw is None or (isinstance(x0_raw, str) and not x0_raw.strip()):
        x0 = sp.zeros(n, 1)
    else:
        try:
            if isinstance(x0_raw, str):
                try:
                    x0_val = literal_eval(x0_raw)
                    x0 = sp.Matrix(x0_val)
                except Exception:
                    x0 = sp.Matrix(sp.sympify(x0_raw))
            else:
                x0 = sp.Matrix(x0_raw)

            if x0.shape == (1, n):
                x0 = x0.T
            if x0.shape == (n,):
                x0 = sp.Matrix(list(x0)).reshape(n, 1)

            if x0.shape != (n, 1):
                return jsonify({
                    "success": False,
                    "error": f"x0 deve essere un vettore colonna di dimensione {n} (ricevuto {x0.shape})."
                }), 400
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"x0 non valido: {str(e)}"
            }), 400

    # Step 1: U(z) = Z{u(k)} (unilaterale, k>=0) — usiamo Lcapy
    if not _LCAPY_AVAILABLE:
        return jsonify({
            "success": False,
            "error": "Lcapy non è disponibile. Installa il pacchetto 'lcapy' per usare la trasformata Z."
        }), 400

    try:
        # Lcapy uses 'n' as the discrete index.
        u_str = str(u_expr)
        u_str = u_str.replace("k", "n").replace("t", "n")

        U_l = lcapy_expr(u_str).ZT()  # unilateral Z-transform

        # Converti in SymPy per continuare i calcoli matriciali e LaTeX
        if hasattr(U_l, "sympy"):
            U_z = sp.simplify(U_l.sympy)
        else:
            U_z = sp.simplify(sp.sympify(str(U_l)))
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Impossibile calcolare automaticamente U(z) con Lcapy: {str(e)}"
        }), 400

    # Step 2: M(z) = z (zI - A)^(-1)
    I = sp.eye(n)
    M_z = z * (z * I - A).inv()

    # Step 4: H(z) = C * M(z)
    H_z = (C * M_z).reshape(1, n)

    # Step 5: W(z) = H(z)/z * B + D
    W_z = (H_z / z) * B + sp.Matrix([[D]])

    # Step 6: y_l(z) = H(z) x0,  y_f(z) = W(z) U(z),  y(z) = y_l + y_f
    y_l_z = (H_z * x0)[0]
    y_f_z = (W_z * U_z)[0]
    y_z = sp.simplify(sp.expand(y_l_z + y_f_z))

    latex_steps = [
        {
            "title": "Trasformata Z dell'ingresso:",
            "content": f"U(z) = \\mathcal{{Z}}\\{{u[k]\\}} = {sp.latex(U_z)}"
        },
        {
            "title": "Calcolo di \\(z(zI-A)^{-1}\\):",
            "content": f"z(zI-A)^{{-1}} = {sp.latex(M_z.applyfunc(sp.factor))}"
        },
        {
            "title": "Calcolo della funzione di trasferimento in Z:",
            "content": f"H(z) = C\\,[z(zI-A)^{{-1}}] = {sp.latex(H_z.applyfunc(sp.factor))}"
        },
        {
            "title": "Calcolo di W(z):",
            "content": f"W(z) = \\frac{{H(z)}}{{z}}B + D = {sp.latex(sp.simplify(((H_z / z) * B)[0]))} + {sp.latex(D)}"
        },
        {
            "title": "Risposta libera nel dominio Z:",
            "content": f"y_l(z) = H(z) x_0 = {sp.latex(sp.factor(y_l_z))}"
        },
        {
            "title": "Risposta forzata nel dominio Z:",
            "content": f"y_f(z) = W(z) U(z) = {sp.latex(sp.factor(y_f_z))}"
        },
        {
            "title": "Risposta completa nel dominio Z:",
            "content": f"y(z) = y_l(z) + y_f(z) = {sp.latex(sp.factor(y_z))}"
        }
    ]

    return jsonify({"success": True, "latex": latex_steps})