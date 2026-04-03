import streamlit as st
import pulp
import nx_helper # Ignore this, just standard networkx
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# --- Page Configuration ---
st.set_page_config(page_title="Optical Network Master", layout="wide")
st.title("🌐 Optical Network RWA & QoS Simulator")
st.markdown("This system optimizes routing based on **Latency**, **Priority**, and **Wavelength Continuity**.")

# --- 1. Topology & Physics Data ---
# Light in fiber: ~200 km/ms
SPEED_OF_LIGHT = 200 

nodes = [1, 2, 3, 4, 5, 6, 7]
# Link distances in km
link_distances = {
    (1,2): 120, (2,3): 150, (3,4): 100, (4,5): 200, 
    (5,6): 130, (6,7): 90, (7,1): 180, (2,7): 110, 
    (3,6): 140, (1,4): 300
}
base_links = list(link_distances.keys())
all_links = base_links + [(v, u) for u, v in base_links]
full_dist = {**link_distances, **{(v, u): d for (u, v), d in link_distances.items()}}

# --- 2. Sidebar Management ---
st.sidebar.header("⚙️ Network Resources")
waves_count = st.sidebar.slider("Wavelengths per Fiber", 1, 10, 4)

if 'demands' not in st.session_state:
    st.session_state.demands = [
        {"src": 1, "dst": 4, "priority": "High"},
        {"src": 2, "dst": 5, "priority": "Normal"}
    ]

with st.sidebar.expander("➕ Add New Demand"):
    s_node = st.selectbox("Source", nodes, index=0)
    d_node = st.selectbox("Destination", nodes, index=3)
    prio_val = st.radio("Priority", ["Normal", "High"])
    if st.sidebar.button("Add Demand"):
        if s_node != d_node:
            st.session_state.demands.append({"src": s_node, "dst": d_node, "priority": prio_val})

if st.sidebar.button("Reset All Traffic"):
    st.session_state.demands = []

st.sidebar.header("⚠️ Fault Management")
is_fail = st.sidebar.checkbox("Simulate Fiber Cut")
cut_link = st.sidebar.selectbox("Select Link to Cut", base_links)

# --- 3. Optimization Engine ---
def run_optimization(fail_link=None):
    active = [l for l in all_links if l != fail_link and (l[1], l[0]) != fail_link]
    current_d = st.session_state.demands
    w_list = list(range(1, waves_count + 1))
    
    prob = pulp.LpProblem("RWA_System", pulp.LpMinimize)
    
    # Decision Variables
    f = pulp.LpVariable.dicts("flow", (range(len(current_d)), active, w_list), 0, 1, pulp.LpInteger)
    ff = pulp.LpVariable.dicts("wave", (range(len(current_d)), w_list), 0, 1, pulp.LpInteger)

    # Objective: Minimize Latency (High Priority gets 10x importance)
    obj = []
    for i, d in enumerate(current_d):
        prio_weight = 10 if d['priority'] == "High" else 1
        for l in active:
            d_km = full_dist[l]
            for w in w_list:
                obj.append(f[i][l][w] * d_km * prio_weight)
    prob += pulp.lpSum(obj)

    # Routing & Wavelength Continuity Constraints
    for i in range(len(current_d)):
        prob += pulp.lpSum(ff[i][w] for w in w_list) == 1
        s, dest = current_d[i]['src'], current_d[i]['dst']
        for n in nodes:
            for w in w_list:
                in_f = pulp.lpSum(f[i][l][w] for l in active if l[1] == n)
                out_f = pulp.lpSum(f[i][l][w] for l in active if l[0] == n)
                if n == s: prob += out_f - in_f == ff[i][w]
                elif n == dest: prob += in_f - out_f == ff[i][w]
                else: prob += in_f - out_f == 0

    # Capacity Constraint
    for l in active:
        for w in w_list:
            prob += pulp.lpSum(f[i][l][w] for i in range(len(current_d))) <= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    results_data = []
    if pulp.LpStatus[prob.status] == 'Optimal':
        for i, d in enumerate(current_d):
            for w in w_list:
                path = [l for l in active if pulp.value(f[i][l][w]) == 1]
                if path:
                    d_total = sum(full_dist[p] for p in path)
                    lat = (d_total / SPEED_OF_LIGHT)
                    results_data.append({
                        "Service": f"{d['src']}→{d['dst']}",
                        "Priority": d['priority'],
                        "Wave": f"λ{w}",
                        "Distance": f"{d_total} km",
                        "Latency": f"{round(lat, 2)} ms",
                        "Path": path,
                        "raw_lat": lat
                    })
    return results_data

# --- 4. Dashboard Visualization ---
final_results = run_optimization(cut_link if is_fail else None)

c1, c2 = st.columns([1, 1.2])

with c1:
    st.subheader("📊 Performance Analytics")
    if final_results:
        res_df = pd.DataFrame(final_results)
        st.table(res_df[["Service", "Priority", "Wave", "Distance", "Latency"]])
        
        st.write("#### Latency Distribution")
        st.bar_chart(res_df.set_index("Service")["raw_lat"])
    else:
        st.warning("No feasible routing! Try increasing wavelength capacity.")

with c2:
    st.subheader("🗺️ Dynamic Topology Map")
    G = nx.Graph()
    G.add_edges_from(base_links)
    pos = nx.kamada_kawai_layout(G)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, node_size=900, node_color='#111', edgecolors='#00defa', ax=ax)
    nx.draw_networkx_labels(G, pos, font_color='white', font_weight='bold', ax=ax)
    
    # Draw Base Links and Labels (FIXED SCRIPT)
    for u, v in base_links:
        dist_val = link_distances[(u,v)]
        # This was the error line - now fixed with xy=pos
        mid_point = (pos[u] + pos[v]) / 2
        ax.annotate(f"{dist_val}km", xy=mid_point, alpha=0.6, color='gray', ha='center', fontsize=9)
    
    nx.draw_networkx_edges(G, pos, edgelist=base_links, edge_color='#444', style='dashed', alpha=0.4, ax=ax)
    
    if is_fail:
        nx.draw_networkx_edges(G, pos, edgelist=[cut_link], edge_color='red', width=6, ax=ax)
        st.error(f"FIBER CUT DETECTED: Link {cut_link}")

    # Draw Active Lightpaths
    colors = ['#FF4B4B', '#00FF00', '#1C83E1', '#F1C40F', '#9B59B6']
    for i, r in enumerate(final_results):
        p_width = 5 if r['Priority'] == "High" else 2.5
        nx.draw_networkx_edges(G, pos, edgelist=r['Path'], edge_color=colors[i % len(colors)], width=p_width, ax=ax)

    ax.set_facecolor('#0E1117')
    fig.patch.set_facecolor('#0E1117')
    plt.axis('off')
    st.pyplot(fig)
