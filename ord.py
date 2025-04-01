import os
import json
import random
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Levenshtein import editops
from ordpy import complexity_entropy

def get_edit_operations(lemma, form):
    ops = editops(lemma, form)
    return [1 if op == 'insert' else 2 if op == 'delete' else 3 for op, _, _ in ops] or [0]

def build_transition_graph(seq):
    G = nx.DiGraph()
    for i in range(len(seq) - 1):
        G.add_edge(seq[i], seq[i + 1])
    return G

def random_walk(G, length):
    if not G.nodes:
        return []
    walk = [random.choice(list(G.nodes))]
    for _ in range(length - 1):
        neighbors = list(G.successors(walk[-1]))
        if not neighbors:
            break
        walk.append(random.choice(neighbors))
    return walk

def process_file(path, lang, dx=6, walk_length=50, n_walks=20):
    df = pd.read_csv(path, sep='\t', header=None, names=["lemma", "form", "tags"])
    results = {"lang": lang, "V": {"series": [], "graph": []}, "N": {"series": [], "graph": []}}

    for pos in ["V", "N"]:
        sub = df[df['tags'].str.startswith(pos)]
        all_ops = []
        for _, row in sub.iterrows():
            ops = get_edit_operations(str(row["lemma"]), str(row["form"]))
            all_ops.extend(ops)

        if len(all_ops) >= dx:
            H, C = complexity_entropy(all_ops, dx=dx)
            results[pos]["series"].append({"H": H, "C": C})

            G = build_transition_graph(all_ops)
            for _ in range(n_walks):
                walk = random_walk(G, walk_length)
                if len(walk) >= dx:
                    H_w, C_w = complexity_entropy(walk, dx=dx)
                    results[pos]["graph"].append({"H": H_w, "C": C_w})

    return results if any(results[pos]["series"] or results[pos]["graph"] for pos in ["V", "N"]) else None

def run_all(folder="datos", dx=6, output_json="complejidad_entropia_lenguas.json"):
    all_results = []
    for fname in os.listdir(folder):
        if fname.endswith(".tsv"):
            lang = fname.replace(".tsv", "")
            path = os.path.join(folder, fname)
            result = process_file(path, lang, dx=dx)
            if result:
                all_results.append(result)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"✅ Resultados guardados en {output_json}")
    return all_results

def get_lang_colors(results):
    unique_langs = sorted(set(res["lang"] for res in results))
    color_map = cm.get_cmap('tab20', len(unique_langs))
    return {lang: color_map(i) for i, lang in enumerate(unique_langs)}

def plot_series_only(results, pos_tag="V", title="Serie completa", savepath=None):
    plt.figure(figsize=(8, 6))
    lang_to_color = get_lang_colors(results)
    for lang_result in results:
        lang = lang_result["lang"]
        color = lang_to_color[lang]
        for point in lang_result[pos_tag]["series"]:
            plt.scatter(point["H"], point["C"], marker="o", color=color, label=lang)
            plt.text(point["H"] + 0.002, point["C"] + 0.002, lang, fontsize=8)
    plt.xlabel("Entropía (H)")
    plt.ylabel("Complejidad (C)")
    plt.title(f"{title} – POS: {pos_tag}")
    plt.grid(True)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=8)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()

def plot_graph_only(results, pos_tag="V", title="Caminatas en grafo", savepath=None):
    plt.figure(figsize=(8, 6))
    lang_to_color = get_lang_colors(results)
    for lang_result in results:
        lang = lang_result["lang"]
        color = lang_to_color[lang]
        points = lang_result[pos_tag]["graph"]
        H_vals = [p["H"] for p in points]
        C_vals = [p["C"] for p in points]
        plt.scatter(H_vals, C_vals, alpha=0.6, color=color, marker="x", label=lang)
    plt.xlabel("Entropía (H)")
    plt.ylabel("Complejidad (C)")
    plt.title(f"{title} – POS: {pos_tag}")
    plt.grid(True)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=8)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()

# === USO ===
if __name__ == "__main__":
    results = run_all()

    # Gráficos para Verbos
    plot_series_only(results, pos_tag="V", title="Figura 4 – Método Serie", savepath="serie_verbos.png")
    plot_graph_only(results, pos_tag="V", title="Figura 4 – Método Grafo", savepath="grafo_verbos.png")

    # Gráficos para Sustantivos
    plot_series_only(results, pos_tag="N", title="Figura 4 – Método Serie", savepath="serie_sustantivos.png")
    plot_graph_only(results, pos_tag="N", title="Figura 4 – Método Grafo", savepath="grafo_sustantivos.png")
