from flask import Blueprint, request, jsonify

def divisors(n: int) -> list[int]:
    """Restituisce tutti i divisori di n in ordine crescente."""
    result = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            result.append(i)
            if i != n // i:
                result.append(n // i)
    return sorted(result)

def compute_hasse_edges(S: list[int]) -> list[tuple[int,int]]:
    """Calcola le coppie (a, b) di coperture immediate in S per la relazione di divisibilità."""
    edges = []
    for i, a in enumerate(S):
        for b in S[i+1:]:
            if b % a != 0:
                continue
            # controlla if esiste un c intermedio
            if not any(c != a and c != b and c % a == 0 and b % c == 0 for c in S):
                edges.append((a, b))
    return edges

# --- AGGIUNTA: verifica_reticolo ---
def verifica_reticolo(S: list[int]) -> tuple[bool, list[tuple[int, int, str]]]:
    """
    Verifica se l'insieme S (con relazione di divisibilità) è un reticolo.
    Restituisce una tupla:
      - bool: True se è reticolo, False altrimenti
      - lista delle coppie (a, b, errore) che mancano sup o inf
    """
    problemi = []
    for i, a in enumerate(S):
        for b in S[i:]:
            # Compute sup (minimo tra i multipli comuni in S)
            multipli = [z for z in S if z % a == 0 and z % b == 0]
            sup = min(multipli) if multipli else None

            # Compute inf (massimo tra i divisori comuni in S)
            divisori = [z for z in S if a % z == 0 and b % z == 0]
            inf = max(divisori) if divisori else None

            if sup not in S:
                problemi.append((a, b, "sup \\notin E"))
            if inf not in S:
                problemi.append((a, b, "inf \\notin E"))

    return len(problemi) == 0, problemi

def genera_hasse_tikz(nodes: list[int], edges: list[tuple[int,int]]) -> str:
    """
    Genera un semplice codice TikZ con posizionamento a livelli basato su profondità.
    Ogni livello y = profondità nel DAG; x equidistante a seconda dell'indice.
    """
    # compute profondità di ciascun nodo (numero di coperture minori)
    depth = {nodes[0]: 0}
    for node in nodes[1:]:
        parents = [a for (a, b) in edges if b == node]
        depth[node] = 1 + max(depth[p] for p in parents) if parents else 0

    # raggruppa per livello
    levels: dict[int, list[int]] = {}
    for node, d in depth.items():
        levels.setdefault(d, []).append(node)

    # ordina i nodi per livello per ridurre intersezioni (barycenter heuristic)
    for d in sorted(levels.keys()):
        if d == 0:
            continue  # il primo livello non ha genitori
        prev_level = levels[d - 1]
        positions = {node: idx for idx, node in enumerate(prev_level)}
        # compute baricentri
        levels[d].sort(key=lambda node: (
            sum(positions.get(p, 0) for p, b in edges if b == node) /
            max(1, sum(1 for p, b in edges if b == node))
        ))

    # genera nodi TikZ
    tikz = ["\\begin{tikzpicture}[scale=3, every node/.style={font=\\small}]"]
    for d, level_nodes in sorted(levels.items()):
        count = len(level_nodes)
        for idx, node in enumerate(level_nodes):
            x = idx - (count-1)/2
            y = d
            tikz.append(f"  \\node ({node}) at ({x},{y}) {{{node}}};")

    # genera archi
    for a, b in edges:
        tikz.append(f"  \\draw ({a}) -- ({b});")

    tikz.append("\\end{tikzpicture}")
    return "\n".join(tikz)

# Blueprint
diagramma_hasse_bp = Blueprint("diagramma_hasse", __name__)

@diagramma_hasse_bp.route("/api/hasse-diagram", methods=["POST"])
@diagramma_hasse_bp.route("/api/diagramma-hasse", methods=["POST"])
def diagramma_hasse():
    """
    API che riceve JSON con:
      - { type: "n", n: <int> }
      - { type: "set", set: [<int>, ...] }
    Restituisce JSON:
      success: bool
      tikz: codice LaTeX (se success)
      error: messaggio (se not success)
    """
    data = request.get_json(force=True)
    if not data or "type" not in data:
        return jsonify(success=False, error="Payload JSON non valido"), 400

    try:
        if data["type"] == "n":
            n = int(data["n"])
            nodes = divisors(n)
        elif data["type"] == "set":
            nodes = sorted(int(x) for x in data["set"])
        else:
            raise ValueError("Tipo non riconosciuto")
        edges = compute_hasse_edges(nodes)
        # --- AGGIUNTA: verifica reticolo ---
        is_lattice, problemi = verifica_reticolo(nodes)
        tikz_code = genera_hasse_tikz(nodes, edges)
        latex_steps = []

        # Step 0: Insieme inserito
        if data["type"] == "n":
            # Notation D_n for the set of divisors of n
            ins_latex = f"D_{{{n}}} = \\{{{', '.join(map(str, nodes))}\\}}"
        else:
            # Insieme arbitrario E = { ... }
            ins_latex = "E = \\{" + ", ".join(map(str, nodes)) + "\\}"

        latex_steps.append({
            "title": "Insieme inserito:",
            "content": f"{ins_latex}"
        })

        # Step 2: Analisi del diagramma di Hasse
        # Compute elementi minimali e massimali
        minimals = [n for n in nodes if not any(b == n for (a, b) in edges)]
        maximals = [n for n in nodes if not any(a == n for (a, b) in edges)]
        # Compute bound inferiori e superiori tramite gcd e lcm
        from math import gcd
        from functools import reduce
        g = reduce(gcd, nodes)
        lcm = reduce(lambda x, y: x * y // gcd(x, y), nodes)
        bounded_below = g in nodes
        bounded_above = lcm in nodes

        # Compute atomi (elementi coperti dal minimo if esiste un minimo)
        atoms = []
        if minimals:
            min_el = min(minimals)
            # Atomi: elementi che coprono il minimo
            atoms = [b for (a, b) in edges if a == min_el]
        # Compute ∨-irriducibili (join-irreducibili) per divisibilità (lcm)
        from math import gcd
        from functools import reduce
        def lcm_func(x, y): return x * y // gcd(x, y)
        join_irreds = []
        for a in nodes:
            reducible = False
            for b in nodes:
                for c in nodes:
                    if b < a and c < a and lcm_func(b, c) == a:
                        reducible = True
                        break
                if reducible:
                    break
            if not reducible:
                join_irreds.append(a)

        # Build contenuto LaTeX
        varnothing = '\\varnothing'
        analysis = fr"""
  \text{{Elementi minimali di }} E: {', '.join(str(x) for x in minimals)}
\]
\[
  \text{{Elementi massimali di }} E: {', '.join(str(x) for x in maximals)}
\]
\[
  \text{{Minimo di }}E: {min(minimals) if minimals else varnothing}
\]
\[
  \text{{Massimo di }}E: {lcm if bounded_above else varnothing}
\]
\[
  \text{{Atomi:}}\, {', '.join(str(x) for x in atoms) or varnothing}
\]
\[
  \text{{Elementi \( \vee \)-irriducibili:}}\, {', '.join(str(x) for x in join_irreds) or varnothing}
\]
\[
  E \text{{ è limitato inferiormente: }} {'Sì' if bounded_below else 'No'}
\]
\[
  E \text{{ è limitato superiormente: }} {'Sì' if bounded_above else 'No'}
\]
\[
  E \text{{ è limitato: }} {'Sì' if bounded_below and bounded_above else 'No'}
"""
        latex_steps.append({
            "title": "Analisi del diagramma di Hasse:",
            "content": analysis
        })

        # --- AGGIUNTA: tabella inf e sup sempre ---
        tabella = "\\begin{array}{c|" + "c" * len(nodes) + "} \\text{} & " + " & ".join(str(x) for x in nodes) + " \\\\ \\hline\n"
        for a in nodes:
            row = [str(a)]
            for b in nodes:
                multipli = [z for z in nodes if z % a == 0 and z % b == 0]
                sup = min(multipli) if multipli else "-"
                divisori = [z for z in nodes if a % z == 0 and b % z == 0]
                inf = max(divisori) if divisori else "-"
                row.append(f"\\tiny\\begin{{matrix}}\\sup={sup}\\\\\\inf={inf}\\end{{matrix}}")
            tabella += " & ".join(row) + " \\\\\n"
        tabella += "\\end{array}"
        latex_steps.append({
            "title": r"Tabella \(\inf\) e \(\sup\):",
            "content": f"{tabella}"
        })

        # Step conclusivo: verifica reticolo
        if is_lattice:
            latex_steps.append({
                "title": "Verifica proprietà di reticolo:",
                "content": "\\text{L'insieme è un reticolo: ogni coppia ha un inf e un sup in } E."
            })
        else:
            # Step 2: elenco coppie problematiche in MathJax with virgole e spazi
            notin = '\\notin'
            text_cmd = '\\text'
            coppie_str = ', '.join(
                f"{text_cmd}{{{motivo.split(' ',1)[0]}}}({a},{b}) {notin} E"
                for a, b, motivo in problemi
            )
            contenuto = fr"""
\text{{L'insieme non è un reticolo. Ecco le coppie problematiche:}}
\]
\[
{coppie_str}
"""
            latex_steps.append({
                "title": "Verifica proprietà di reticolo:",
                "content": contenuto
            })
        cy_nodes = [{"data": {"id": str(n), "label": str(n)}} for n in nodes]
        cy_edges = [{"data": {"source": str(a), "target": str(b)}} for (a, b) in edges]
        return jsonify(
            success=True,
            tikz=tikz_code,
            latex=latex_steps,
            nodes=cy_nodes,
            edges=cy_edges
        )
    except Exception as e:
        return jsonify(success=False, error=str(e)), 400