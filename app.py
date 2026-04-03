import streamlit as st
import pulp
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# --- Configuration ---
st.set_page_config(page_title="Master Optical Planner", layout="wide")
st.title("🛡️ Enterprise Optical Network & QoS Management")

# --- 1. Network Constants ---
# Physics: Light in fiber travels at ~200,000 km/s (5 microseconds per km)
SPEED_OF_LIGHT_FIBER = 200000 

nodes = [1, 2, 3, 4, 5, 6, 7]
# Define links with physical distances (km)
link_data = {
    (1,2): 120, (2,3): 150, (3,4): 100, (4,5): 200, 
    (5,6): 130, (6,7): 90, (7,1): 180, (2,7): 110, 
    (3,6): 140, (1,4): 300
}
base_links = list(link_data.keys())
links = base_links + [(v, u) for u, v in base_links]
full_link_distances = {**link_data, **{(v, u): d for (u, v), d in link_data.items()}}

# --- 2. Sidebar Management ---
st.sidebar.header("📡 Resource & Traffic")
num_waves = st.sidebar.slider("Wavelengths per Link", 1, 10, 4)

if 'demands' not in st.session_state:
    st.session_state.demands = [
        {"src": 1, "dst": 4, "priority": "High"},
        {"src": 2, "dst": 5, "priority": "Normal"}
    ]

with st.sidebar.expander("➕ Add New Traffic Demand"):
    new_src = st.selectbox("From Node", nodes, index=0)
    new_dst = st.selectbox("To Node", nodes, index=5)
    new_prio = st.radio("Priority Level", ["Normal", "High"])
    if st.sidebar.button("Provision Lightpath"):
        if new_src != new_dst:
            st.session_state.demands.append({"src": new_src, "dst": new_dst, "priority": new_prio})

if st.sidebar.button("Clear All Demands"):
    st.session_state.demands = []

st.sidebar.header("🛠️ Fault Simulation")
fail_on = st.sidebar.checkbox("Simulate Fiber Cut")
failed_l = st.sidebar.selectbox("Select Cut Link", base_links)

# --- 3. RWA Engine with QoS & Latency ---
def solve_master_rwa(fail_link=None):
    active_links = [l for l in links if l != fail_link and (l[1], l[0]) != fail_link]
    current_demands = st.session_state.demands
    wavelengths = list(range(1, num_waves + 1))
    
    prob = pulp.LpProblem("Master_RWA", pulp.LpMinimize)
    
    # Variables
    f = pulp.LpVariable.dicts("flow", (range(len(current_demands)), active_links, wavelengths), 0, 1, pulp.LpInteger)
    ff = pulp.LpVariable.dicts("wave", (range(len(current_demands)), wavelengths), 0, 1, pulp.LpInteger)

    # Objective: Minimize Latency for High Priority, and Link Usage for Normal
    obj = []
    for i, d in enumerate(current_demands):
        multiplier = 10 if d['priority'] == "High" else 1
        for l in active_links:
            dist = full_link_distances[l]
            for w in wavelengths:
                obj.append(f[i][l][w] * dist * multiplier)
    prob += pulp.lpSum(obj)

    # Constraints
    for i in range(len(current_demands)):
        prob += pulp.lpSum(ff[i][w] for w in wavelengths) == 1
        s, dest = current_demands[i]['src'], current_demands[i]['dst']
        for n in nodes:
            for w in wavelengths:
                in_f = pulp.lpSum(f[i][l][w] for l in active_links if l[1] == n)
                out_f = pulp.lpSum(f[i][l][w] for l in active_links if l[0] == n)
                if n == s: prob += out_f - in_f == ff[i][w]
                elif n == dest: prob += in_f - out_f == ff[i][w]
                else: prob += in_f - out_f == 0

    for l in active_links:
        for w in wavelengths:
            prob += pulp.lpSum(f[i][l][w] for i in range(len(current_demands))) <= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    res = []
    if pulp.LpStatus[prob.status] == 'Optimal':
        for i, d in enumerate(current_demands):
            for w in wavelengths:
                path = [l for l in active_links if pulp.value(f[i][l][w]) == 1]
                if path:
                    total_dist = sum(full_link_distances[lp] for lp in path)
                    latency = (total_dist / SPEED_OF_LIGHT_FIBER) * 1000 # Convert to ms
                    res.append({
                        "Demand": f"{d['src']}→{d['dst']}",
                        "Priority": d['priority'],
                        "Wave": f"λ{w}",
                        "Distance (km)": total_dist,
                        "Latency (ms)": round(latency, 3),
                        "Path": path
                    })
    return res

# --- 4. UI Layout ---
results = solve_master_rwa(failed_l if fail_on else None)

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📊 Path Analysis & QoS Report")
    if results:
        df = pd.DataFrame(results)
        st.dataframe(df.drop(columns=['Path']), use_container_width=True)
        
        # Latency Comparison Chart
        st.write("#### Latency per Service (ms)")
        st.bar_chart(df.set_index("Demand")["Latency (ms)"])
    else:
        st.error("🚨 Network Overload! No feasible paths for current demands.")

with col2:
    st.subheader("🗺️ Fiber Topology Map")
    G = nx.Graph()
    G.add_edges_from(base_links)
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='#111', edgecolors='#00defa', ax=ax)
    nx.draw_networkx_labels(G, pos, font_color='white', font_weight='bold', ax=ax)
    
    # Draw all fiber links
    for u, v in base_links:
        dist = link_data[(u,v)]
        ax.annotate(f"{dist}km", pos=(pos[u]+pos[v])/2, alpha=0.5, color='gray')
    
    nx.draw_networkx_edges(G, pos, edgelist=base_links, edge_color='#333', style='dashed', ax=ax)
    
    if fail_on:
        nx.draw_networkx_edges(G, pos, edgelist=[failed_l], edge_color='red', width=5, ax=ax)

    # Draw active lightpaths with priority-based thickness
    colors = ['#ff4b4b', '#00ff00', '#0099ff', '#f1c40f', '#9b59b6']
    for i, r in enumerate(results):
        width = 5 if r['Priority'] == "High" else 2
        nx.draw_networkx_edges(G, pos, edgelist=r['Path'], edge_color=colors[i % len(colors)], width=width, ax=ax, label=f"D{i+1}")

    ax.set_facecolor('#0E1117')
    fig.patch.set_facecolor('#0E1117')
    st.pyplot(fig)
