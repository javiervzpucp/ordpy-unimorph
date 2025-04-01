# grafo_visual.py (con opciones de node2vec-like bias)
import os
import json
import random
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from Levenshtein import editops
from ordpy import complexity_entropy

OP_LABELS = {1: 'insert', 2: 'delete', 3: 'replace'}
REVERSE_OP_LABELS = {'insert': 1, 'delete': 2, 'replace': 3}

def get_edit_operations(lemma, form):
    ops = editops(lemma, form)
    return [REVERSE_OP_LABELS[op] for op, _, _ in ops] or [0]

def build_transition_graph(seq):
    G = nx.DiGraph()
    for i in range(len(seq) - 1):
        u, v = seq[i], seq[i + 1]
        if G.has_edge(u, v):
            G[u][v]['weight'] += 1
        else:
            G.add_edge(u, v, weight=1)
    return G

def build_trigram_graph(forms):
    G = nx.DiGraph()
    for form in forms:
        padded = f"#{form}$"
        trigrams = [padded[i:i+3] for i in range(len(padded) - 2)]
        for i in range(len(trigrams) - 1):
            u, v = trigrams[i], trigrams[i + 1]
            if G.has_edge(u, v):
                G[u][v]['weight'] += 1
            else:
                G.add_edge(u, v, weight=1)
    return G

def weighted_random_choice(neighbors):
    total = sum(weight for _, weight in neighbors)
    r = random.uniform(0, total)
    upto = 0
    for node, weight in neighbors:
        if upto + weight >= r:
            return node
        upto += weight
    return neighbors[-1][0]  # fallback

def weighted_random_walk(G, length, avoid_cycles=False, restart_prob=0.0):
    if not G.nodes:
        return []
    walk = []
    start = random.choice(list(G.nodes))
    current = start
    walk.append(current)
    for _ in range(length - 1):
        if restart_prob > 0 and random.random() < restart_prob:
            current = start
        neighbors = [(nbr, G[current][nbr]['weight']) for nbr in G.successors(current)]
        if avoid_cycles:
            neighbors = [(n, w) for n, w in neighbors if n not in walk]
        if not neighbors:
            break
        current = weighted_random_choice(neighbors)
        walk.append(current)
    return walk

def node2vec_random_walk(G, length, p=1.0, q=1.0):
    if not G.nodes:
        return []
    walk = []
    start = random.choice(list(G.nodes))
    walk.append(start)

    neighbors = list(G.successors(start))
    if not neighbors:
        return walk
    current = weighted_random_choice([(nbr, G[start][nbr]['weight']) for nbr in neighbors])
    walk.append(current)

    for _ in range(length - 2):
        prev = walk[-2]
        curr = walk[-1]
        neighbors = list(G.successors(curr))
        if not neighbors:
            break
        weights = []
        for nbr in neighbors:
            if nbr == prev:
                weight = G[curr][nbr]['weight'] / p
            elif G.has_edge(prev, nbr):
                weight = G[curr][nbr]['weight']
            else:
                weight = G[curr][nbr]['weight'] / q
            weights.append((nbr, weight))
        current = weighted_random_choice(weights)
        walk.append(current)
    return walk

def process_file_graph(path, lang, pos_tag="V", dx=6, walk_length=50, n_walks=50, avoid_cycles=False, restart_prob=0.0, use_node2vec=False, p=1.0, q=1.0):
    df = pd.read_csv(path, sep='\t', header=None, names=["lemma", "form", "tags"])
    sub = df[df['tags'].str.startswith(pos_tag)]
    edit_seqs = [get_edit_operations(str(row["lemma"]), str(row["form"])) for _, row in sub.iterrows()]
    full_seq = [op for seq in edit_seqs for op in seq]
    forms = [str(row["form"]) for _, row in sub.iterrows()]

    results = {"weighted": [], "trigram": []}

    G1 = build_transition_graph(full_seq)
    for _ in range(n_walks):
        walk = node2vec_random_walk(G1, walk_length, p, q) if use_node2vec else weighted_random_walk(G1, walk_length, avoid_cycles=avoid_cycles, restart_prob=restart_prob)
        if len(walk) >= dx:
            H, C = complexity_entropy(walk, dx=dx)
            results["weighted"].append((H, C))

    G2 = build_trigram_graph(forms)
    for _ in range(n_walks):
        walk = node2vec_random_walk(G2, walk_length, p, q) if use_node2vec else weighted_random_walk(G2, walk_length, avoid_cycles=avoid_cycles, restart_prob=restart_prob)
        flat_walk = [ord(c) for trigram in walk for c in trigram] if walk else []
        if len(flat_walk) >= dx:
            H, C = complexity_entropy(flat_walk, dx=dx)
            results["trigram"].append((H, C))

    return results, G1, G2

if __name__ == "__main__":
    folder = "datos"
    all_results = {"weighted": {}, "trigram": {}}
    comparison_data = {"weighted": {}, "trigram": {}}

    for fname in os.listdir(folder):
        if fname.endswith(".tsv"):
            lang = fname.replace(".tsv", "")
            path = os.path.join(folder, fname)
            hc_vals, G1, G2 = process_file_graph(path, lang, avoid_cycles=True, restart_prob=0.1, use_node2vec=True, p=1.0, q=2.0)
            all_results["weighted"][lang] = hc_vals["weighted"]
            all_results["trigram"][lang] = hc_vals["trigram"]
            comparison_data["weighted"][lang] = hc_vals["weighted"]
            comparison_data["trigram"][lang] = hc_vals["trigram"]

            for g, name in zip([G1, G2], ["transiciones", "trigramas"]):
                if g:
                    plt.figure(figsize=(10, 8))
                    pos = nx.kamada_kawai_layout(g)
                    edge_labels = nx.get_edge_attributes(g, 'weight')
                    nx.draw_networkx_nodes(g, pos, node_color='skyblue', node_size=1200, edgecolors='black')
                    nx.draw_networkx_edges(g, pos, edge_color='gray', arrows=True)
                    nx.draw_networkx_labels(g, pos, font_size=12)
                    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_color='black')
                    plt.title(f"Grafo {name} – {lang}")
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(f"{lang}_grafo_{name}.png", dpi=300)
                    plt.close()

    for method in ["weighted", "trigram"]:
        plt.figure(figsize=(10, 8))
        for lang, values in comparison_data[method].items():
            if values:
                Hs, Cs = zip(*values)
                mean_H, mean_C = np.mean(Hs), np.mean(Cs)
                std_H, std_C = np.std(Hs), np.std(Cs)
                plt.errorbar(mean_H, mean_C, xerr=std_H, yerr=std_C, fmt='o', label=lang, capsize=5)
        plt.xlabel("Entropía (H)")
        plt.ylabel("Complejidad (C)")
        plt.title(f"Promedio y desviación – Método {method}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"promedio_desviacion_lenguas_{method}.png", dpi=300)
        plt.close()

    with open("grafo_resultados_completos.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)