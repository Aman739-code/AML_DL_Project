"""
Stage 06: DAG Construction
============================
Convert extracted triples into a NetworkX directed graph, reject self-loops,
detect cycles, and resolve them by removing the lowest-confidence edge in
each cycle. Visualise the graph before and after cycle removal.
"""

import os
import sys

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import SEED, setSeed, ensureDir

setSeed(SEED)

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPHS_DIR = ensureDir(os.path.join(STAGE_DIR, "graphs"))
TRIPLES_PATH = os.path.join(
    os.path.dirname(__file__), "..", 
    "05_relationExtractionBaseline", "results", "extractedTriples.csv"
)


def buildGraphFromTriples(triplesDf: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed graph from extracted triples.
    
    - Self-loops are rejected
    - Each edge carries: relation, confidence, sentenceId, docId
    - Multiple edges between same nodes are collapsed, keeping highest confidence
    """
    G = nx.DiGraph()
    
    selfLoopsRejected = 0
    edgesAdded = 0
    
    for _, row in triplesDf.iterrows():
        subj = row["subject"]
        obj = row["object"]
        
        # Reject self-loops
        if subj == obj:
            selfLoopsRejected += 1
            continue
        
        # Skip invalid entities
        if subj == "?" or obj == "?":
            continue
        
        # Add or update edge
        if G.has_edge(subj, obj):
            # Keep the edge with higher confidence
            existing = G[subj][obj].get("confidence", 0)
            if row.get("confidence", 1.0) > existing:
                G[subj][obj]["confidence"] = row.get("confidence", 1.0)
                G[subj][obj]["relation"] = row["relation"]
                G[subj][obj]["sentenceId"] = row["sentenceId"]
                G[subj][obj]["docId"] = row["docId"]
        else:
            G.add_edge(
                subj, obj,
                relation=row["relation"],
                confidence=row.get("confidence", 1.0),
                sentenceId=row["sentenceId"],
                docId=row["docId"],
            )
            edgesAdded += 1
    
    print(f"  Self-loops rejected: {selfLoopsRejected}")
    print(f"  Edges added: {edgesAdded}")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    
    return G


def detectAndResolveCycles(G: nx.DiGraph) -> tuple:
    """
    Detect cycles in the directed graph and resolve them by removing
    the lowest-confidence edge in each cycle.
    
    Returns:
        - dagGraph: the acyclic graph
        - removedEdges: list of edges removed
        - cyclesFound: number of cycles detected
    """
    removedEdges = []
    cyclesFound = 0
    
    while True:
        try:
            cycle = nx.find_cycle(G, orientation="original")
            cyclesFound += 1
            
            # Find the edge with lowest confidence in this cycle
            minConfidence = float("inf")
            minEdge = None
            
            for u, v, direction in cycle:
                edgeConf = G[u][v].get("confidence", 1.0)
                if edgeConf < minConfidence:
                    minConfidence = edgeConf
                    minEdge = (u, v)
            
            # If all edges have equal confidence, remove the last edge in the cycle
            if minEdge is None:
                minEdge = (cycle[-1][0], cycle[-1][1])
            
            # Record the removed edge
            edgeData = dict(G[minEdge[0]][minEdge[1]])
            removedEdges.append({
                "source": minEdge[0],
                "target": minEdge[1],
                "confidence": minConfidence,
                "relation": edgeData.get("relation", ""),
                "reason": f"Cycle {cyclesFound}: lowest confidence edge",
            })
            
            G.remove_edge(*minEdge)
            print(f"  Cycle {cyclesFound}: removed edge {minEdge[0]} → {minEdge[1]} "
                  f"(confidence={minConfidence:.3f})")
            
        except nx.NetworkXNoCycle:
            break
    
    # Verify DAG
    assert nx.is_directed_acyclic_graph(G), "Graph is still not a DAG!"
    
    print(f"\n  Total cycles found: {cyclesFound}")
    print(f"  Edges removed: {len(removedEdges)}")
    print(f"  Final DAG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G, removedEdges, cyclesFound


def visualiseGraph(G: nx.DiGraph, title: str, outputPath: str, 
                   highlightEdges=None):
    """
    Visualise the graph with clear, readable labels.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No nodes to display", ha="center", va="center",
                fontsize=16, transform=ax.transAxes)
        ax.set_title(title, fontsize=14, fontweight="bold")
        fig.savefig(outputPath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    
    # Layout
    if G.number_of_nodes() <= 20:
        pos = nx.spring_layout(G, seed=SEED, k=2.0)
    else:
        pos = nx.kamada_kawai_layout(G)
    
    # Draw nodes
    nodeColors = ["#3498db" for _ in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=nodeColors,
                          node_size=800, alpha=0.9, edgecolors="black", linewidths=1.5)
    
    # Draw edges
    edgeColors = []
    edgeWidths = []
    for u, v in G.edges():
        if highlightEdges and (u, v) in highlightEdges:
            edgeColors.append("#e74c3c")
            edgeWidths.append(2.5)
        else:
            edgeColors.append("#2c3e50")
            edgeWidths.append(1.5)
    
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edgeColors,
                          width=edgeWidths, alpha=0.7, 
                          arrows=True, arrowsize=20,
                          connectionstyle="arc3,rad=0.1")
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_weight="bold",
                           font_color="white")
    
    # Edge labels (relation type)
    edgeLabels = {}
    for u, v, data in G.edges(data=True):
        rel = data.get("relation", "")
        if rel:
            edgeLabels[(u, v)] = rel.replace("_", "\n")
    
    if len(edgeLabels) <= 30:  # Only show labels if not too many
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edgeLabels,
                                     font_size=7, ax=ax, alpha=0.8)
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    
    plt.tight_layout()
    fig.savefig(outputPath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved {outputPath}")


def writeCycleResolutionReport(cyclesFound: int, removedEdges: list, 
                                graphBefore: dict, graphAfter: dict):
    """Write the cycle resolution report."""
    path = os.path.join(STAGE_DIR, "cycleResolutionReport.md")
    
    with open(path, "w") as f:
        f.write("# Cycle Resolution Report\n\n")
        
        f.write("## Graph Statistics\n\n")
        f.write("| Metric | Before | After |\n")
        f.write("|--------|--------|-------|\n")
        f.write(f"| Nodes | {graphBefore['nodes']} | {graphAfter['nodes']} |\n")
        f.write(f"| Edges | {graphBefore['edges']} | {graphAfter['edges']} |\n")
        f.write(f"| Is DAG | {'No' if graphBefore.get('hasCycles') else 'Yes'} | Yes |\n\n")
        
        f.write(f"## Cycles Found: {cyclesFound}\n\n")
        f.write(f"## Edges Removed: {len(removedEdges)}\n\n")
        
        if removedEdges:
            f.write("| # | Source | Target | Confidence | Relation | Reason |\n")
            f.write("|---|--------|--------|-----------|----------|--------|\n")
            for i, edge in enumerate(removedEdges):
                f.write(f"| {i+1} | {edge['source']} | {edge['target']} | "
                        f"{edge['confidence']:.3f} | {edge['relation']} | {edge['reason']} |\n")
        else:
            f.write("No cycles were found — the extracted graph was already a DAG.\n\n")
            f.write("This is expected when the number of extracted triples is small ")
            f.write("or when the extraction patterns naturally produce acyclic structures.\n")
        
        f.write("\n## Resolution Strategy\n\n")
        f.write("Cycles were resolved by iteratively finding cycles and removing the ")
        f.write("edge with the **lowest confidence score** in each cycle. This is a ")
        f.write("principled approach because lower-confidence edges are more likely to ")
        f.write("be extraction errors. If all edges in a cycle have equal confidence, ")
        f.write("the last edge discovered is removed.\n\n")
        
        f.write("## Gap Analysis\n\n")
        f.write("This stage highlights the central gap of the project: text extraction ")
        f.write("produces edges independently, without enforcing global structural ")
        f.write("constraints like acyclicity. The fact that cycle resolution may remove ")
        f.write("edges demonstrates that pure text extraction cannot guarantee a valid ")
        f.write("causal DAG — this is precisely what the Bayesian network stage addresses.\n")
    
    print(f"  → Saved {path}")


def main():
    """Main entry point for Stage 06."""
    print("=" * 60)
    print("Stage 06: DAG Construction")
    print("=" * 60)
    
    # Load extracted triples
    if not os.path.exists(TRIPLES_PATH):
        print(f"  ⚠️ Triples file not found at {TRIPLES_PATH}")
        print("  Creating synthetic triples from test data for demonstration...")
        triplesDf = createSyntheticTriples()
    else:
        triplesDf = pd.read_csv(TRIPLES_PATH)
    
    print(f"\n  Loaded {len(triplesDf)} triples")
    
    # Build graph
    print("\n🔨 Building directed graph...")
    G = buildGraphFromTriples(triplesDf)
    
    # Record before stats
    graphBefore = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "hasCycles": not nx.is_directed_acyclic_graph(G),
    }
    
    # Visualise before cycle removal
    visualiseGraph(
        G.copy(), 
        "Causal Graph — Before Cycle Removal",
        os.path.join(GRAPHS_DIR, "causalGraphBeforeCycleRemoval.png")
    )
    
    # Detect and resolve cycles
    print("\n🔄 Detecting and resolving cycles...")
    G, removedEdges, cyclesFound = detectAndResolveCycles(G)
    
    graphAfter = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
    }
    
    # Visualise final DAG
    visualiseGraph(
        G, 
        "Causal DAG — After Cycle Resolution",
        os.path.join(GRAPHS_DIR, "causalDAG.png")
    )
    
    # Save graph
    graphmlPath = os.path.join(GRAPHS_DIR, "dag.graphml")
    nx.write_graphml(G, graphmlPath)
    print(f"  → DAG saved to {graphmlPath}")
    
    # Write report
    writeCycleResolutionReport(cyclesFound, removedEdges, graphBefore, graphAfter)
    
    print("\n✅ Stage 06 complete.")
    return G


def createSyntheticTriples():
    """
    Fallback: create triples from Corr2Cause data directly for demonstration.
    Used only if Stage 05 hasn't been run yet.
    """
    import re
    processedDir = os.path.join(os.path.dirname(__file__), "..", "02_dataAcquisition", "processed")
    testPath = os.path.join(processedDir, "test.csv")
    
    if not os.path.exists(testPath):
        # Create minimal triples
        return pd.DataFrame({
            "subject": ["A", "B", "C", "A"],
            "relation": ["direct_cause", "direct_cause", "confounder", "collider"],
            "object": ["B", "C", "A", "C"],
            "confidence": [0.9, 0.8, 0.7, 0.6],
            "sentenceId": ["s1", "s2", "s3", "s4"],
            "docId": ["d1", "d2", "d3", "d4"],
            "model": ["synthetic"] * 4,
        })
    
    df = pd.read_csv(testPath)
    triples = []
    
    for idx, row in df.iterrows():
        if row.get("label", row.get("label", 0)) == 1:
            inputText = str(row.get("input", row.get(list(row.index)[0], "")))
            if "Hypothesis:" in inputText:
                hypothesis = inputText.split("Hypothesis:")[1].strip()
                hypoVars = re.findall(r'\b([A-F])\b', hypothesis)
                if len(hypoVars) >= 2:
                    hypoLower = hypothesis.lower()
                    if "directly causes" in hypoLower:
                        rel = "direct_cause"
                    elif "collider" in hypoLower:
                        rel = "collider"
                    elif "confounder" in hypoLower:
                        rel = "confounder"
                    elif "ancestor" in hypoLower:
                        rel = "indirect_cause"
                    else:
                        rel = "causal_relation"
                    
                    triples.append({
                        "subject": hypoVars[0],
                        "relation": rel,
                        "object": hypoVars[1],
                        "confidence": 1.0,
                        "sentenceId": f"test_{idx:04d}_hypo",
                        "docId": f"test_{idx}",
                        "model": "groundTruth",
                    })
    
    return pd.DataFrame(triples) if triples else pd.DataFrame({
        "subject": ["A"], "relation": ["direct_cause"], "object": ["B"],
        "confidence": [1.0], "sentenceId": ["s1"], "docId": ["d1"],
        "model": ["synthetic"],
    })


if __name__ == "__main__":
    main()
