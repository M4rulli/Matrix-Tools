from flask import Blueprint, request, jsonify
import sympy as sp

# Tutti i messaggi d’error contengono LaTeX racchiuso tra \( … \) così
# il frontend può renderizzarli with MathJax.

teorema_cinese_resto_bp = Blueprint("teorema_cinese_resto", __name__)


def _latex_eq(a: int, b: int, n: int) -> str:
    """Return LaTeX string for  ax ≡ b (mod n)."""
    return fr"{a}x \equiv {b} \pmod{{{n}}}"


@teorema_cinese_resto_bp.route("/api/chinese-remainder-theorem", methods=["POST"])
@teorema_cinese_resto_bp.route("/api/teorema-cinese-resto", methods=["POST"])
def teorema_cinese_resto():
    """
    Step 1  Classi inverse (allineate)
    Step 2  Sistema semplificato + verifica forma cinese
    """
    try:
        data = request.get_json(force=True)
        if not isinstance(data, dict) or "equations" not in data:
            raise ValueError("'equations' mancante nel JSON.")
        equations = data["equations"]
        if not isinstance(equations, list) or not equations:
            raise ValueError(
                r"Parametro JSON \texttt{'equations'} assente "
                r"o non valido \((\text{lista non vuota})\)."
            )

        simplified_info = []
        align_rows = []
        mod_list = []
        all_invertible = True

        for eq in equations:
            if not isinstance(eq, dict):
                raise ValueError(
                    r"Ogni equazione deve essere un dizionario con chiavi "
                    r"\(\,a, b, n\,\)."
                )
            a = int(eq.get("a"))
            b = int(eq.get("b"))
            n = int(eq.get("n"))
            if n <= 0:
                raise ValueError(r"I moduli devono essere interi positivi \((n > 0)\).")

            # Semplificazione by gcd(a,n)
            d = sp.gcd(a, n)
            if b % d != 0:
                raise ValueError(
                    fr"\({_latex_eq(a, b, n)}\)"
                    fr"è impossibile perché \(\gcd({a},{n}) = {d}\) "
                    fr"non divide \({b}\)."
                )
            a1, b1, n1 = int(a//d), int(b//d), int(n//d)

            g = sp.gcd(a1, n1)
            if g == 1:
                inv = int(sp.invert(a1, n1))
                k = (inv * b1) % n1
                align_rows.append(fr"{_latex_eq(a1,b1,n1)} &\;\Longrightarrow x \equiv {inv}\cdot {b1} \equiv {k} \pmod{{{n1}}}\\")
                simplified_info.append({"a": a1, "b": b1, "n": n1, "inverse": inv, "x": k})
                mod_list.append(n1)
            else:
                align_rows.append(fr"{_latex_eq(a1,b1,n1)} &\;\Longrightarrow \text{{non esiste inversa (gcd={g})}}\\")
                simplified_info.append({"a": a1, "b": b1, "n": n1, "inverse": None, "x": None})
                all_invertible = False

        # Step 1 latex
        step1 = {
            "title": "Classi inverse:",
            "content": r"\begin{align*}" + "\n".join(align_rows) + r"\end{align*}"
        }

        # Step 2: system semplificato + forma cinese
        chinese = all_invertible and all(sp.gcd(m1, m2) == 1 for i, m1 in enumerate(mod_list) for m2 in mod_list[i+1:])
        if chinese:
            sys_rows = [ fr"x \equiv {info['x']} \pmod{{{info['n']}}}" for info in simplified_info ]
        else:
            # Manteniamo only quelle with inversa per mostrare eventuale sottosistema
            sys_rows = [ fr"x \equiv {info['x']} \pmod{{{info['n']}}}" 
                         for info in simplified_info if info['inverse'] is not None ]

        sistema = r"\left\{\begin{array}{l}" + r"\\[2pt]".join(sys_rows) + r"\end{array}\right."

        step2 = {
            "title": "Sistema semplificato e verifica forma cinese:",
            "content": fr"{sistema}\qquad\text{{({'Il sistema è in forma cinese' if chinese else 'Il sistema NON è in forma cinese'})}}"
        }

        # --- Step 3: prodotto dei moduli e valori N_i ------------------------
        extra_steps = []
        if chinese:
            # N = prodotto dei moduli
            bigN = 1
            for m in mod_list:
                bigN *= m

            # Compute N_i = N / n_i
            Ni_values = [bigN // m for m in mod_list]

            # Step 3: show N and each N_i
            align_N = [fr"N = " + " \\cdot ".join(map(str, mod_list)) + fr" = {bigN}\\\\[2pt]"]
            for idx, (Ni, mi) in enumerate(zip(Ni_values, mod_list), 1):
                align_N.append(
                    fr"N_{{{idx}}} \;=\; \frac{{N}}{{n_{{{idx}}}}} \;=\; "
                    fr"\frac{{{bigN}}}{{{mi}}} \;=\; {Ni}\\"
                )

            step3 = {
                "title": "Prodotto dei moduli $N$ e valori $N_i$:",
                "content": r"\begin{align*}" + "\n".join(align_N) + r"\end{align*}"
            }
            extra_steps.append(step3)

            # --- Step 4: solve le congruenze N_i y_i ≡ 1 (mod n_i) ----------
            align_y = []
            yi_list = []
            for idx, (Ni, mi) in enumerate(zip(Ni_values, mod_list), 1):
                inv = int(sp.invert(Ni, mi))
                yi_list.append(inv)
                align_y.append(
                    fr"{Ni}\,y_{{{idx}}} \equiv 1 \pmod{{{mi}}} &\;\Longrightarrow\; y_{{{idx}}} \equiv {inv} \pmod{{{mi}}}\\"
                )

            step4 = {
                "title": "Soluzione delle congruenze $N_i y_i \\equiv 1 \\pmod{n_i}$:",
                "content": r"\begin{align*}" + "\n".join(align_y) + r"\end{align*}"
            }
            extra_steps.append(step4)

            # --- Step 5: combinazione lineare per una solution particular ---
            #   x_0 ≡ Σ a_i N_i y_i  (mod N)
            terms_sym = " + ".join(
                [fr"a_{{{i+1}}}N_{{{i+1}}}y_{{{i+1}}}" for i in range(len(mod_list))]
            )
            line_formula = fr"x_0 \equiv {terms_sym} \pmod{{N}}"

            numeric_terms = []
            total_value = 0
            for info, Ni, yi in zip(simplified_info, Ni_values, yi_list):
                ai = info['x']  # resto semplificato
                numeric_terms.append(fr"{ai}\cdot {Ni}\cdot {yi}")
                total_value += ai * Ni * yi

            x0_val = total_value % bigN

            line_numeric = (
                fr"x_0 \equiv " + " + ".join(numeric_terms) + fr" \pmod{{{bigN}}}"
            )
            line_result = fr"\equiv {total_value} \equiv {x0_val} \pmod{{{bigN}}}"

            step5 = {
                "title": "Combinazione lineare ($x_0$):",
                "content": r"\begin{align*}"
                + line_formula
                + r"\\"
                + line_numeric
                + r"\\"
                + line_result
                + r"\end{align*}",
            }
            extra_steps.append(step5)

            step6 = {
                "title": "Soluzione particolare e forma generale:",
                "content": r"\begin{align*}"
                    + fr"x_0 &\equiv {x0_val} \pmod{{{bigN}}}\\[4pt]"
                    + r"\text{Insieme di tutte le soluzioni:}\\"
                    + r"x &= x_0 + N z \\"
                    + fr"&= {x0_val} + {bigN}k,\quad \forall z \in \mathbb{{Z}}"
                    + r"\end{align*}"
            }
            extra_steps.append(step6)

            # Aggiungiamo gli step extra alla lista latex
            latex_steps = [step1, step2] + extra_steps
        else:
            latex_steps = [step1, step2]

        return jsonify({"success": True, "simplified": simplified_info, "latex": latex_steps})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
