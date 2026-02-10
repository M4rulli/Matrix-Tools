def latex_to_sympy_str(expr):
    import re
    # Rimozione \left e \right usati per parentesi grandi in LaTeX
    expr = re.sub(r'\\left', '', expr)
    expr = re.sub(r'\\right', '', expr)
    # Gestione \cdot
    expr = re.sub(r'\\cdot', '*', expr)
    # Gestione frazioni
    expr = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', expr)
    # Sostituzione pedici: x_1 -> x1
    expr = re.sub(r'x_(\d+)', r'x\1', expr)
    expr = re.sub(r'x_\{(\d+)\}', r'x\1', expr)
    # Gestione potenze: x^3 → x**3, x^{3} → x**3
    expr = re.sub(r'\^(\{[^{}]+\}|[a-zA-Z0-9\(])', lambda m: '**' + m.group(1).strip('{}'), expr)
    # Moltiplicazione implicita: inserisce * tra numeri, variabili e parentesi adiacenti
    expr = re.sub(r'(?<=[0-9A-Za-z\)])(?=[A-Za-z\(])', '*', expr)
    # Rimuove eventuali backslash residui
    expr = expr.replace('\\', '')
    return expr


from flask import Blueprint, request, jsonify
import sympy as sp
import re
from sympy import Matrix, symbols, simplify, sympify, Eq, solve
from ast import literal_eval
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application,
    convert_xor, function_exponentiation
)



transformations = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
    function_exponentiation
)


def sostituisci_pedici(equation_str):
    """
    Converte 'x_1', 'x_{2}', ecc. in 'x1', 'x2', ...
    """
    def replacer(match):
        index = match.group(1) or match.group(2)
        return f"x{index}"
    equation_str = re.sub(r"x_(?:\{(\d+)\}|\s*(\d+))", replacer, equation_str)
    return equation_str

linearizzazione_bp = Blueprint("linearizzazione", __name__)

@linearizzazione_bp.route('/api/linearizzazione', methods=['POST'])
def linearizzazione():
    try:
        # Estrai i dati dal JSON
        data = request.get_json()
        equazioni = data.get('equazioni', [])
        equazione_uscita = data.get('equazioneUscita', '')
        valore_ingresso_str = data.get('valoreIngresso', '0')
        tipo_dominio = data.get('dominio', '')

        # Normalizza eventuali doppie backslash da input LaTeX
        equazioni = [eq.replace('\\\\', '\\') for eq in equazioni]
        equazione_uscita = equazione_uscita.replace('\\\\', '\\')
        valore_ingresso_str = valore_ingresso_str.replace('\\\\', '\\')
        # Rimuovi \left e \right anche da valore_ingresso_str
        valore_ingresso_str = valore_ingresso_str.replace('\\left', '').replace('\\right', '')
        # Normalizza shorthand LaTeX \frac12  → \frac{1}{2}
        valore_ingresso_str = re.sub(r"\\frac\s*([0-9]+)\s*([0-9]+)", r"\\frac{\1}{\2}", valore_ingresso_str)

        # Normalizza \cdot come moltiplicazione
        equazioni = [eq.replace(r'\cdot', '*') for eq in equazioni]
        equazione_uscita = equazione_uscita.replace(r'\cdot', '*')



        latex_expr = latex_to_sympy_str(valore_ingresso_str)
        valore_ingresso = parse_expr(latex_expr, transformations=transformations)

        numero_equazioni = len(equazioni)

        # Definisci le variabili simboliche x1, x2, ..., xn
        x = symbols(f'x1:{numero_equazioni+1}')  # Genera x1, x2, ..., xn
        from sympy.abc import u

        # Pre-elabora le equazioni
        eqs_sostituite = []
        f_exprs = []
        for eq in equazioni:
            try:
                local_dict = {f'x{i+1}': x[i] for i in range(numero_equazioni)}
                local_dict['u'] = u
                parsed_expr = parse_expr(latex_to_sympy_str(eq), transformations=transformations, local_dict=local_dict)
                f_exprs.append(parsed_expr)  # Salva espressione simbolica con 'u'
                expr_at_input = parsed_expr.subs(u, valore_ingresso)
                eqs_sostituite.append(expr_at_input)
            except Exception as e:
                return jsonify({
                    "success": False,
                    "errore": f"Errore nell'elaborazione dell'equazione LaTeX '{eq}': {str(e)}"
                })


        # Costruisci le equazioni per il punto di equilibrio
        if tipo_dominio == 'typeContinuo':
            eqs_punto_di_equilibrio = [Eq(eq, 0) for eq in eqs_sostituite]
        elif tipo_dominio == 'typeDiscreto':
            eqs_punto_di_equilibrio = [Eq(x[i], eq) for i, eq in enumerate(eqs_sostituite)]
        else:
            return jsonify({"success": False, "errore": "Selezionare un tipo di dominio valido."})

     

        # DEBUG: stampa il sistema di equazioni in forma Python e LaTeX
        # Rimuovi solo equazioni tautologiche 0=0, ma conserva Eq(x1,0)
        eqs_da_risolvere = []
        for eq in eqs_punto_di_equilibrio:
            if isinstance(eq, Eq) and eq.lhs.is_number and eq.rhs.is_number and eq.lhs == eq.rhs:
                continue
            eqs_da_risolvere.append(eq)

        # Manual solve: support multiple roots for polynomial equations
        soluzioni = []
        if len(eqs_da_risolvere) == 1:
            # Single equation: possibly multiple solutions for one variable
            eq = eqs_da_risolvere[0]
            syms = list(eq.free_symbols)
            if syms:
                var = syms[0]
                try:
                    roots = solve(eq, var)
                except Exception:
                    roots = []
                # For each root, build a solution dict
                for r in roots:
                    if r.is_real is False:
                        continue  # scarta radici complesse
                    if r.is_real is None:
                        # usa il test simbolico
                        re_part, im_part = sp.re(r), sp.im(r)
                        if im_part.simplify() != 0:
                            continue
                    sol = {var: r}
                    # assign parameters to other variables
                    params = iter(symbols(f'c1:{len(x)+1}'))
                    for v in x:
                        if v not in sol:
                            sol[v] = next(params)
                    soluzioni.append(sol)
        else:
            # Multiple equations: solve system normally
            sols = solve(eqs_da_risolvere, x, dict=True)
            for sol in sols:
                # sympify and convert to simple dict
                s = {}
                for v in x:
                    s[v] = sol.get(v, None)
                soluzioni.append(s)

        # --- FILTRO: mantieni solo soluzioni reali ---
        def is_real_expr(expr):
            """
            Restituisce True se l'espressione è numericamente reale
            (parte immaginaria simbolicamente zero).
            """
            if expr is None:
                return False
            # usa proprietà is_real quando disponibile
            if expr.is_real is True:
                return True
            if expr.is_real is False:
                return False
            # fallback: controlla parte immaginaria
            re_part, im_part = sp.re(expr), sp.im(expr)
            return im_part.simplify() == 0

        soluzioni = [
            sol for sol in soluzioni
            if all(is_real_expr(val) for val in sol.values())
        ]

        # Se dopo tutto non ci sono soluzioni, creane una sottodeterminata generica
        if not soluzioni:
            # fallback generico: tutti parametri liberi
            params = iter(symbols(f'c1:{len(x)+1}'))
            sol = {v: next(params) for v in x}
            soluzioni = [sol]

        # Nuovo blocco: accetta anche soluzioni simboliche, mantenendo consistenza
        soluzioni_reali = []
        for sol in soluzioni:
            sol_dict = {}
            for i, var in enumerate(x):
                chiave = f"x_{i+1}"
                valore = sol.get(var, None)
                if valore is None:
                    # parametro libero: creiamo un Symbol c_{i+1}
                    sol_dict[chiave] = sp.Symbol(f"c_{i+1}")
                else: 
                    # semplifichiamo e manteniamo oggetti SymPy (Symbol, Rational, etc.)
                    sol_dict[chiave] = sp.nsimplify(valore, rational=True)
            soluzioni_reali.append(sol_dict)

        # Sostituisci simboli con espressioni semplici se sono riferimenti ad altri simboli o costanti
        for sol in soluzioni_reali:
            changed = True
            while changed:
                changed = False
                for k, v in sol.items():
                    if v in sol and sol[v] != v:
                        sol[k] = sol[v]
                        changed = True

        # Step 1: Punto di equilibrio
        # Build LaTeX for equilibrium equations and solution
        eqs_latex = " \\\\ ".join([sp.latex(eq) for eq in eqs_punto_di_equilibrio])
        latex_steps = []
        for idx, sol in enumerate(soluzioni_reali):
            # semplifica espressioni tra variabili nel punto di lavoro
            changed = True
            while changed:
                changed = False
                for k, v in sol.items():
                    if v in sol and sol[v] != v:
                        sol[k] = sol[v]
                        changed = True
            if tipo_dominio == "typeContinuo":
                descrizione_dominio = r"\mathbb{T} = \mathbb{R}\implies f(\mathbf{x}_e, u_e) = 0 \implies "
            else:
                descrizione_dominio = r"\mathbb{T} = \mathbb{Z}\implies f(\mathbf{x}_e, u_e) = \mathbf{x}_e \implies "

            # costruisci dizionario di sostituzioni per x_i → c_i
            subs_dict = { x[j]: sp.Symbol(f"c_{j+1}") for j in range(numero_equazioni) }
            descrizione_latex = (
                descrizione_dominio
                + r"\mathbf{x}_e = \left("
                + ", ".join(sp.latex(v.subs(subs_dict)) for v in sol.values())
                + r"\right)"
            )

            # Rileva tutti i simboli c_i anche dentro espressioni complesse
            const_set = {c for v in sol.values() for c in v.free_symbols if c.name.startswith("c_")}
            if const_set:
                const_latex = ", ".join(sp.latex(c) for c in sorted(const_set, key=lambda s: s.name))
                descrizione_latex += " \\ \\operatorname{con}\\," + const_latex + " \\in \\mathbb{R}"

            latex_steps.append({
                "title": f"Punto di equilibrio {idx + 1}:",
                "content": descrizione_latex
            })

        # Verifica se ci sono soluzioni reali
        if not soluzioni_reali:
            return jsonify({
                "success": False,
                "errore": "Nessuna soluzione reale trovata per il punto di equilibrio.",
                "suggerimento": "Verifica che: 1) Le equazioni siano corrette 2) Il valore di ingresso sia compatibile 3) Il sistema ammetta soluzioni reali",
                "debug_soluzioni_raw": [str(s) for s in soluzioni]
            })

        soluzione_latex = ",\\quad ".join([f"{k} = {v}" for k, v in soluzioni_reali[0].items()])


        from sympy import diff, Matrix
        def matrix_to_latex(mat):
            mat = sp.nsimplify(mat, rational=True)
            rows = [" & ".join(sp.latex(el) for el in row) for row in mat.tolist()]
            return "\\begin{bmatrix}" + " \\\\ ".join(rows) + "\\end{bmatrix}"

        for idx, sol in enumerate(soluzioni_reali):
            punto_eq_subs = {}
            for i in range(numero_equazioni):
                chiave = f"x_{i+1}"
                valore = sol[chiave]
                try:
                    punto_eq_subs[x[i]] = sp.nsimplify(valore, rational=True)
                except Exception as e:
                    punto_eq_subs[x[i]] = sp.sympify(valore)

            # Ricrea subs_dict per sostituire x_i → c_i
            subs_dict = { x[j]: sp.Symbol(f"c_{j+1}") for j in range(numero_equazioni) }

            try:
                raw_exprs = equazioni
                f_exprs_symbolic = []
                f_exprs_evaluated = []
                for i, expr in enumerate(equazioni):
                    tex_str = latex_to_sympy_str(expr)
                    local_dict = {f'x{j+1}': x[j] for j in range(numero_equazioni)}
                    local_dict['u'] = u
                    parsed_raw = parse_expr(tex_str, transformations=transformations, local_dict=local_dict)
                    f_exprs_symbolic.append(parsed_raw)
                    f_exprs_evaluated.append(parsed_raw.subs(u, valore_ingresso))

                A_sym = Matrix([[diff(expr, x_var) for x_var in x] for expr in f_exprs_evaluated])
                # Se il punto ha parametri c_i, sostituisci x_i→c_i senza evalf, altrimenti usa evalf
                if any(isinstance(v, sp.Symbol) and v.name.startswith("c_") for v in punto_eq_subs.values()):
                    A = A_sym.subs(punto_eq_subs)
                else:
                    A = A_sym.subs(punto_eq_subs).evalf()

                B_symbolic = Matrix([[diff(expr, u)] for expr in f_exprs_symbolic])
                if any(isinstance(v, sp.Symbol) and v.name.startswith("c_") for v in punto_eq_subs.values()):
                    B = B_symbolic.subs(punto_eq_subs).subs(u, valore_ingresso)
                else:
                    B = B_symbolic.subs(punto_eq_subs).subs(u, valore_ingresso).evalf()


                # Coerente con il parsing delle equazioni
                h_tex = latex_to_sympy_str(equazione_uscita)
                local_dict = {f'x{i+1}': x[i] for i in range(numero_equazioni)}
                local_dict['u'] = u
                h_orig = parse_expr(h_tex, transformations=transformations, local_dict=local_dict)
                h_uscita = h_orig.subs(u, valore_ingresso)

                C_sym = Matrix([[diff(h_uscita, x_var) for x_var in x]])
                if any(isinstance(v, sp.Symbol) and v.name.startswith("c_") for v in punto_eq_subs.values()):
                    C = C_sym.subs(punto_eq_subs)
                else:
                    C = C_sym.subs(punto_eq_subs).evalf()

                D_expr = diff(h_orig, u)
                D_sym = D_expr.subs(u, valore_ingresso)
                if any(isinstance(v, sp.Symbol) and v.name.startswith("c_") for v in punto_eq_subs.values()):
                    D = D_sym.subs(punto_eq_subs)
                else:
                    D = D_sym.subs(punto_eq_subs).evalf()
                D = Matrix([[D]])

                contenuto_matrici = (
                    r"A = " + matrix_to_latex(A) + r",\quad B = " + matrix_to_latex(B) + r" \\ " + r",\quad C = " + matrix_to_latex(C) + r",\quad D = " + matrix_to_latex(D)
                )

                # Usa lo stesso subs_dict di sopra per sostituire x_i → c_i
                xe_latex = ", ".join(sp.latex(v.subs(subs_dict)) for v in sol.values())
                title = f"Linearizzazione relativa a \\( \\mathbf{{x}}_e = \\left({xe_latex}\\right) \\):"
                latex_steps.append({
                    "title": title,
                    "content": contenuto_matrici
                })

            except Exception as e:
                return jsonify({
                    "success": False,
                    "errore": f"Errore durante la linearizzazione del punto {idx+1}: {str(e)}"
                })

        return jsonify({
            "success": True,
            "latex": latex_steps,
            "messaggio": "Punto di equilibrio calcolato correttamente.",
            "punto_equilibrio": [{k: str(v) for k, v in sol.items()} for sol in soluzioni_reali],
            "punto_equilibrio_tex": soluzione_latex,
            "valore_ingresso": str(valore_ingresso),
            "equazioni": equazioni,
            "tipo_dominio": tipo_dominio,
            "numero_equazioni": numero_equazioni
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "errore": f"Errore durante il calcolo: {str(e)}"
        })