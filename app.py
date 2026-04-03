import streamlit as st
import pulp
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# --- Configuration ---
st.set_page_config(page_title="Ultra Optical RWA Suite", layout="wide")
st.title("🚀 Ultra Optical Network Management & RWA Suite")

# --- Sidebar: Dynamic Inputs ---
st.sidebar.header("1. Network Resources")
num_waves = st.sidebar.slider("Wavelength Capacity per Link", 1, 8, 4)

st.sidebar.header("2. Traffic Management")
src = st.sidebar.selectbox("Source Node", [1, 2, 3, 4, 5, 6, 7], index=0)
dst = st.sidebar.selectbox("Destination Node", [1, 2, 3, 4, 5, 6, 7], index=3)
if st.sidebar.button("Add New Demand"):
    if 'custom_demands' not in st.session_state:
        st.session_state.custom_demands = [(1,4), (2,5), (7,3)]
    if (src, dst) not in st.session_state.custom_demands and src != dst:
        st.session_state.custom_demands.append((src, dst))
        st.sidebar.success(f"Added: {src} -> {dst}")

st.sidebar.header("3. Fault Simulation")
simulate_fail = st.sidebar.checkbox("Activate Link Failure")
fail_choice = st.sidebar.selectbox("Select Failed Link", [(1,2), (2,3), (1,4), (6,7)])

# --- Data Setup ---
nodes = [1, 2, 3, 4, 5, 6, 7]
base_links = [(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,1), (2,7), (3,6), (1,4)]
links = base_links + [(v, u) for u, v in base_links]
sd_pairs = st.session_state.get('custom_demands', [(1,4), (2,5), (7,3)])
wavelengths = list(range(1, num_waves + 1))

# --- Solver Engine ---
def solve_advanced_rwa(fail_l=None):
    active_links = [l for l in links if l != fail_l and (l[1], l[0]) != fail_l]
    prob = pulp.LpProblem("Ultra_RWA", pulp.LpMinimize)
    f = pulp.LpVariable.dicts("flow", (sd_pairs, active_links, wavelengths), 0, 1, pulp.LpInteger)
    ff = pulp.LpVariable.dicts("wave", (sd_pairs, wavelengths), 0, 1, pulp.LpInteger)

    prob += pulp.lpSum(f[s][l][w] for s in sd_pairs for l in active_links for w in wavelengths)

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
    
    res = []
    utilization = {l: 0 for l in base_links}
    if pulp.LpStatus[prob.status] == 'Optimal':
        for s in sd_pairs:
            for w in wavelengths:
                path = [l for l in active_links if pulp.value(f[s][l][w]) == 1]
                if path:
                    res.append({"Demand": f"{s[0]} \u2192 {s[1]}", "Wave": f"λ{w}", "Path": path})
                    for lp in path:
                        norm_l = tuple(sorted(lp))
                        if norm_l in utilization: utilization[norm_l] += 1
    return res, utilization, pulp.value(prob.objective)

# --- Execute & Layout ---
results, usage, cost = solve_advanced_rwa(fail_choice if simulate_fail else None)

m_col1, m_col2 = st.columns([1, 1.5])

with m_col1:
    st.subheader("📊 Network Analytics")
    st.metric("Total Link-Wavelength Resources Used", int(cost) if results else 0)
    
    if results:
        st.write("#### Active Lightpaths")
        st.dataframe(pd.DataFrame(results), use_container_width=True)
        
        st.write("#### Link Utilization (Wavelengths per Link)")
        usage_df = pd.DataFrame([{"Link": k, "Load": v} for k, v in usage.items()])
        st.bar_chart(usage_df.set_index("Link"))
    else:
        st.error("Network Congested: Cannot route all demands. Increase Wavelengths!")

with m_col2:
    st.subheader("🗺️ Live Topology Map")
    G = nx.Graph()
    G.add_edges_from(base_links)
    pos = nx.kamada_kawai_layout(G)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    nx.draw_networkx_nodes(G, pos, node_size=900, node_color='#0E1117', edgecolors='#00FFAA', ax=ax)
    nx.draw_networkx_labels(G, pos, font_color='white', font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=base_links, edge_color='#555555', style='dashed', alpha=0.3, ax=ax)
    
    if simulate_fail:
        nx.draw_networkx_edges(G, pos, edgelist=[fail_choice], edge_color='red', width=6, ax=ax)
        st.warning(f"⚠️ ALERT: Fiber Cut detected at Link {fail_choice}!")

    colors = ['#FF4B4B', '#1C83E1', '#00FF00', '#F63366', '#FFAA00', '#7D3C98']
    for i, r in enumerate(results):
        nx.draw_networkx_edges(G, pos, edgelist=r['Path'], edge_color=colors[i % len(colors)], width=4, ax=ax, label=r['Demand'])
    
    ax.set_facecolor('#0E1117')
    fig.patch.set_facecolor('#0E1117')
    plt.legend(facecolor='#0E1117', labelcolor='white')
    st.pyplot(fig)
