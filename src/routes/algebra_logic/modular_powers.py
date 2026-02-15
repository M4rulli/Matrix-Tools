from flask import Blueprint, request, jsonify
import sympy as sp

potenze_modulari_bp = Blueprint("potenze_modulari", __name__)

@potenze_modulari_bp.route("/api/modular-powers", methods=["POST"])
@potenze_modulari_bp.route("/api/potenze-modulari", methods=["POST"])
def potenze_modulari():
    try:
        data = request.get_json()
        a = int(data.get("base"))
        e = int(data.get("exponent"))
        n = int(data.get("modulo"))

        if n <= 0:
            raise ValueError("Il modulo deve essere maggiore di 0.")
        
        # Step list da restituire al frontend
        latex_steps = []
        
        # Step 1: riduzione modulo n
        a_mod = a % n
        latex_steps.append({
            "title": "Tentativo di riduzione della base:",
            "content": fr"{a} \bmod {n} = {a_mod}"
        })
        if a_mod != a:
            latex_steps.append({
                "title": "Studio della nuova potenza:",
                "content": fr"\Rightarrow\ {a_mod}^{{{e}}} \bmod {n}"
            })

        # Step 2: controlla invertibilità (gcd)
        d = sp.gcd(a_mod, n)
        latex_steps.append({
            "title": "Controllo invertibilità:",
            "content": fr"\gcd({a_mod}, {n}) = {d}"
        })

        # Compute la function φ(n) usando le proprietà:
        # φ(p) = p−1 if p è primo
        # φ(p^l) = p^l − p^{l−1}
        # φ(rs) = φ(r)·φ(s) if MCD(r,s) = 1 (già implicito nel prodotto dei fattori)
        def phi_custom(n):
            if n == 1:
                return 1
            factors = sp.factorint(n)
            result = 1
            for p, l in factors.items():
                result *= p**l - p**(l-1)
            return result

        # Caso 1: a invertibile → applico Eulero
        if d == 1:
            phi = phi_custom(n)
            q = e // phi
            r = e % phi

            latex_steps.append({
                "title": "Poiché è invertibile, applichiamo il Teorema di Eulero:",
                "content": fr"""
\varphi({n}) = {phi} \Rightarrow {a_mod}^{{{phi}}} \equiv 1 \mod {n}
\]
\[
{a_mod}^{{{e}}} = {a_mod}^{{{{ {phi} \cdot {q} + {r} }}}} = \left({a_mod}^{{{phi}}}\right)^{{{q}}} \cdot {a_mod}^{{{r}}} = {a_mod}^{{{r}}} \mod {n}
"""
            })

            result = pow(int(a_mod), int(r), int(n))
            latex_steps.append({
                "title": "Risultato finale:",
                "content": fr"{a_mod}^{{{r}}} \bmod {n} = {result}"
            })

        # Caso 2: a non invertibile → cerco ciclo di potenze
        else:
            potenze = []
            seen = {}
            cycle_found = False
            for k in range(1, n * 2):  # limite generoso
                pot = pow(a_mod, k, n)
                if pot in seen:
                    e_prime = seen[pot]
                    e_double_prime = k
                    latex_steps.append({
                        "title": "Non è invertibile, cerchiamo un ciclo di potenze:",
                        "content": fr"{a_mod}^{{{e_prime}}} \equiv {a_mod}^{{{e_double_prime}}} \mod {n}"
                    })
                    cycle_found = True
                    break
                seen[pot] = k
                potenze.append(pot)

            if not cycle_found:
                raise Exception("Nessun ciclo trovato in tempo utile.")

            # Riduzione esponente via divisione e - e' = (e'' - e') q + r
            delta = e_double_prime - e_prime
            r = (e - e_prime) % delta
            latex_steps.append({
                "title": "Riduzione esponente:",
                "content": fr"""
{e} - {e_prime} = (e'' - e') \cdot q + r
\]
\[
= ({e_double_prime} - {e_prime}) \cdot q + {r} \Rightarrow {a_mod}^{{{e}}} = {a_mod}^{{{e_prime}}} \cdot {a_mod}^{{{r}}}
"""
            })

            part1 = pow(int(a_mod), int(e_prime), int(n))
            part2 = pow(int(a_mod), int(r), int(n))
            result = (part1 * part2) % n

            latex_steps.append({
                "title": "Risultato finale:",
                "content": fr"{part1} \cdot {part2} \mod {n} = {result}"
            })

        return jsonify({
            "success": True,
            "result": result,
            "latex": latex_steps
        })

    except (TypeError, ValueError) as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "Errore imprevisto: " + str(e)
        })