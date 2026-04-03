import streamlit as st
import pulp
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# --- Page Config ---
st.set_page_config(page_title="Optical RWA Simulator", layout="wide")

st.title("🌐 Optical Network RWA Management System")
st.markdown("This system simulates Routing and Wavelength Assignment (RWA) based on ILP formulations.")

# --- Sidebar Controls ---
st.sidebar.header("Network Settings")
num_waves = st.sidebar.slider("Available Wavelengths", 1, 5, 3)
st.sidebar.subheader("Fault Management")
simulate_fail = st.sidebar.checkbox("Simulate Link Failure")
failed_link_choice = st.sidebar.selectbox("Select Link to Fail", [(1,4), (2,3), (5,6), (7,1)])

# --- 1. Define Network Data ---
nodes = [1, 2, 3, 4, 5, 6, 7]
all_links = [(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,1), (2,7), (3,6), (1,4)]
# Make it bidirectional for the solver
links = all_links + [(v, u) for u, v in all_links]
sd_pairs = [(1,4), (2,5), (7,3), (6,2)]
wavelengths = list(range(1, num_waves + 1))

# --- 2. RWA Solver Function ---
def solve_rwa(link_to_fail=None):
    # Remove failed link if requested
    active_links = [l for l in links if l != link_to_fail and (l[1], l[0]) != link_to_fail]
    
    prob = pulp.LpProblem("RWA_Optimization", pulp.LpMinimize)
    
    # Variables
    f = pulp.LpVariable.dicts("flow", (sd_pairs, active_links, wavelengths), 0, 1, pulp.LpInteger)
    ff = pulp.LpVariable.dicts("wave", (sd_pairs, wavelengths), 0, 1, pulp.LpInteger)

    # Objective: Minimize total wavelength-links used
    prob += pulp.lpSum(f[s][l][w] for s in sd_pairs for l in active_links for w in wavelengths)

    # Constraints
    for s in sd_pairs:
        prob += pulp.lpSum(ff[s][w] for w in wavelengths) == 1
        for n in nodes:
            for w in wavelengths:
                in_f = pulp.lpSum(f[s][l][w] for l in active_links if l[1] == n)
                out_f = pulp.lpSum(f[s][l][w] for l in active_links if l[0] == n)
                if n == s[0]: prob += out_f - in_f == ff[s][w]
                elif n == s[1]: prob += in_f - out_f == ff[s][w]
                else: prob += in_f - out_f == 0

    for l in active_links:
        for w in wavelengths:
            prob += pulp.lpSum(f[s][l][w] for s in sd_pairs) <= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    output = []
    if pulp.LpStatus[prob.status] == 'Optimal':
        for s in sd_pairs:
            for w in wavelengths:
                path = [l for l in active_links if pulp.value(f[s][l][w]) == 1]
                if path:
                    output.append({"Demand": f"{s[0]} \u2192 {s[1]}", "Wavelength": f"λ{w}", "Path": path})
    return output, pulp.value(prob.objective)

# --- 3. Execute and Show Results ---
target_fail = failed_link_choice if simulate_fail else None
results, total_cost = solve_rwa(target_fail)

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Results Summary")
    if results:
        st.success(f"Optimal Solution Found!")
        st.metric("Total Link Usage Cost", int(total_cost))
        st.write("#### Lightpath Assignments")
        st.table(pd.DataFrame(results)[["Demand", "Wavelength", "Path"]])
    else:
        st.error("No Solution Found! Try increasing wavelengths.")

with col2:
    st.subheader("Network Topology Map")
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(all_links)
    
    pos = nx.kamada_kawai_layout(G)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw base network
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='#1f77b4', ax=ax)
    nx.draw_networkx_labels(G, pos, font_color='white', font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=all_links, edge_color='lightgray', style='dashed', alpha=0.5, ax=ax)
    
    # Highlight Failed Link
    if simulate_fail:
        nx.draw_networkx_edges(G, pos, edgelist=[failed_link_choice], edge_color='black', width=5, style='dotted', ax=ax)
        st.warning(f"Note: Link {failed_link_choice} is currently DOWN.")

    # Draw active lightpaths
    path_colors = ['red', 'green', 'blue', 'orange', 'purple']
    for i, res in enumerate(results):
        nx.draw_networkx_edges(G, pos, edgelist=res['Path'], edge_color=path_colors[i % len(path_colors)], width=3.5, ax=ax)

    ax.axis('off')
    st.pyplot(fig)