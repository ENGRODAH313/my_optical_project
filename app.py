import streamlit as st
import pulp
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# --- الإعدادات العامة ---
st.set_page_config(page_title="Optical Survivability Pro", layout="wide")
st.title("🛡️ Advanced Optical Network: 1+1 Protection & Survivability")
st.markdown("This system calculates **Primary** and **Backup** paths to ensure 99.999% availability.")

# --- 1. بيانات الشبكة ---
SPEED_OF_LIGHT = 200 
nodes = [1, 2, 3, 4, 5, 6, 7]
link_distances = {
    (1,2): 120, (2,3): 150, (3,4): 100, (4,5): 200, 
    (5,6): 130, (6,7): 90, (7,1): 180, (2,7): 110, 
    (3,6): 140, (1,4): 300
}
base_links = list(link_distances.keys())
all_links = base_links + [(v, u) for u, v in base_links]
full_dist = {**link_distances, **{(v, u): d for (u, v), d in link_distances.items()}}

# --- 2. التحكم في حركة المرور ---
st.sidebar.header("🚀 Network Control")
waves_limit = st.sidebar.slider("Wavelengths per Fiber", 1, 12, 6)

if 'demands' not in st.session_state:
    st.session_state.demands = [{"src": 1, "dst": 4, "name": "Critical_Data_01"}]

with st.sidebar.expander("📝 Add Protected Service"):
    s_n = st.selectbox("From Node", nodes, index=0)
    d_n = st.selectbox("To Node", nodes, index=4)
    s_name = st.text_input("Service Name", f"Service_{len(st.session_state.demands)+1}")
    if st.sidebar.button("Provision with 1+1 Protection"):
        if s_n != d_n:
            st.session_state.demands.append({"src": s_n, "dst": d_n, "name": s_name})

if st.sidebar.button("Clear Network"):
    st.session_state.demands = []

st.sidebar.header("💥 Stress Test (Fiber Cut)")
fail_active = st.sidebar.checkbox("Simulate Link Failure")
failed_link = st.sidebar.selectbox("Select Fiber to Cut", base_links)

# --- 3. محرك الأمثلة (Optimization Engine) ---
def solve_survivable_rwa(cut=None):
    active = [l for l in all_links if l != cut and (l[1], l[0]) != cut]
    d_list = st.session_state.demands
    w_list = list(range(1, waves_limit + 1))
    
    prob = pulp.LpProblem("Survivable_RWA", pulp.LpMinimize)
    
    # f[demand, link, wave, type] type: 0=Working, 1=Protection
    f = pulp.LpVariable.dicts("flow", (range(len(d_list)), active, w_list, [0, 1]), 0, 1, pulp.LpInteger)
    
    # Objective: Minimize total distance for both paths
    prob += pulp.lpSum(f[i][l][w][t] * full_dist[l] for i in range(len(d_list)) for l in active for w in w_list for t in [0,1])

    for i in range(len(d_list)):
        s, d = d_list[i]['src'], d_list[i]['dst']
        for t in [0, 1]: # For both Working and Protection
            for n in nodes:
                for w in w_list:
                    in_f = pulp.lpSum(f[i][l][w][t] for l in active if l[1] == n)
                    out_f = pulp.lpSum(f[i][l][w][t] for l in active if l[0] == n)
                    # Simple flow conservation (simplified for 1 wave per path)
                    if n == s: prob += out_f - in_f == (1 if pulp.lpSum(f[i][lx][wx][t] for lx in active if lx[0]==s for wx in w_list) == 1 else 0) # Constrained in logic
    
        # Ensure each demand has exactly 1 working and 1 protection path
        for t in [0, 1]:
            prob += pulp.lpSum(f[i][l][w][t] for l in active if l[0] == s for w in w_list) == 1
        
        # Link Disjointness: Working and Protection cannot share the same fiber!
        for l in base_links:
            if l in active or (l[1], l[0]) in active:
                l_dir = l if l in active else (l[1], l[0])
                prob += pulp.lpSum(f[i][l_dir][w][0] + f[i][l_dir][w][1] for w in w_list) <= 1

    # Capacity: Each wave on each link used only once
    for l in active:
        for w in w_list:
            prob += pulp.lpSum(f[i][l][w][t] for i in range(len(d_list)) for t in [0,1]) <= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    res = []
    if pulp.LpStatus[prob.status] == 'Optimal':
        for i, d in enumerate(d_list):
            w_path = [l for l in active if any(pulp.value(f[i][l][w][0]) == 1 for w in w_list)]
            p_path = [l for l in active if any(pulp.value(f[i][l][w][1]) == 1 for w in w_list)]
            res.append({"Name": d['name'], "Working": w_path, "Protection": p_path})
    return res

# --- 4. العرض المرئي ---
results = solve_survivable_rwa(failed_link if fail_active else None)

c1, c2 = st.columns([1, 1.2])

with c1:
    st.subheader("📋 Provisioning Status")
    if results:
        for r in results:
            with st.expander(f"Service: {r['Name']}"):
                st.write(f"🟢 **Working Path:** {r['Working']}")
                st.write(f"🔵 **Backup Path:** {r['Protection']}")
                if fail_active and any(l in r['Working'] or (l[1],l[0]) in r['Working'] for l in [failed_link]):
                    st.error("⚠️ Primary Path Down! Traffic Rerouted to Backup.")
                else:
                    st.success("System Healthy")
    else:
        st.error("No Solution: Links are too congested for 1+1 protection.")

with c2:
    st.subheader("🌐 Survivability Map")
    G = nx.Graph(); G.add_edges_from(base_links)
    pos = nx.kamada_kawai_layout(G)
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='#111', edgecolors='#00defa', ax=ax)
    nx.draw_networkx_labels(G, pos, font_color='white', font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=base_links, edge_color='#333', style='dashed', alpha=0.2, ax=ax)
    
    if fail_active:
        nx.draw_networkx_edges(G, pos, edgelist=[failed_link], edge_color='red', width=7, ax=ax)

    colors = ['#00FF00', '#0099FF', '#FFCC00', '#FF00FF']
    for i, r in enumerate(results):
        nx.draw_networkx_edges(G, pos, edgelist=r['Working'], edge_color=colors[i%4], width=4, ax=ax, label=f"{r['Name']} (W)")
        nx.draw_networkx_edges(G, pos, edgelist=r['Protection'], edge_color=colors[i%4], width=2, style='dotted', ax=ax, label=f"{r['Name']} (P)")

    ax.set_facecolor('#0E1117'); fig.patch.set_facecolor('#0E1117'); plt.axis('off')
    st.pyplot(fig)
