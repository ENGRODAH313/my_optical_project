import streamlit as st
import pulp
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# --- إعدادات الصفحة ---
st.set_page_config(page_title="Optical survivability Suite", layout="wide")
st.title(" Enterprise Optical Network: Survivability & RWA")
st.markdown("This system manages **Working** and **Backup** lightpaths with high reliability.")

# --- 1. بيانات الشبكة والفيزياء ---
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

# --- 2. إدارة البيانات في الجلسة ---
if 'demands' not in st.session_state:
    st.session_state.demands = [{"src": 1, "dst": 4, "name": "Critical_Link_01"}]

# --- 3. القائمة الجانبية ---
st.sidebar.header(" Network Control")
waves_limit = st.sidebar.slider("Wavelengths per Fiber", 1, 20, 12)

with st.sidebar.expander(" Add Protected Service"):
    s_n = st.selectbox("From Node", nodes, index=0)
    d_n = st.selectbox("To Node", nodes, index=4)
    s_name = st.text_input("Service Name", f"Svc_{len(st.session_state.demands)+1}")
    if st.sidebar.button("Provision Lightpath"):
        if s_n != d_n:
            st.session_state.demands.append({"src": s_n, "dst": d_n, "name": s_name})

if st.sidebar.button("Clear Network"):
    st.session_state.demands = []

st.sidebar.header(" Stress Test")
fail_active = st.sidebar.checkbox("Simulate Fiber Cut")
failed_link = st.sidebar.selectbox("Select Fiber to Cut", base_links)

# --- 4. محرك الحل الرياضي (ILP Solver) ---
def solve_rwa_pro(cut=None):
    active = [l for l in all_links if l != cut and (l[1], l[0]) != cut]
    d_list = st.session_state.demands
    w_list = list(range(1, waves_limit + 1))
    
    if not d_list: return []

    prob = pulp.LpProblem("RWA_System", pulp.LpMinimize)
    
    # f[demand_idx, link, wave, type] type: 0=Working, 1=Protection
    f = pulp.LpVariable.dicts("flow", (range(len(d_list)), active, w_list, [0, 1]), 0, 1, pulp.LpInteger)

    # Objective: Minimize total latency and encourage path diversity
    # We add a small penalty to the protection path to make it different if possible
    prob += pulp.lpSum(f[i][l][w][0] * full_dist[l] for i in range(len(d_list)) for l in active for w in w_list) + \
            pulp.lpSum(f[i][l][w][1] * full_dist[l] * 1.2 for i in range(len(d_list)) for l in active for w in w_list)

    for i in range(len(d_list)):
        s, d = d_list[i]['src'], d_list[i]['dst']
        for t in [0, 1]:
            for n in nodes:
                # Flow Conservation
                in_f = pulp.lpSum(f[i][l][w][t] for l in active if l[1] == n for w in w_list)
                out_f = pulp.lpSum(f[i][l][w][t] for l in active if l[0] == n for w in w_list)
                if n == s: prob += out_f - in_f == 1
                elif n == d: prob += in_f - out_f == 1
                else: prob += in_f - out_f == 0
            
            # Wavelength Continuity: Each path must use exactly one wavelength
            prob += pulp.lpSum(f[i][l][w][t] for l in active if l[0] == s for w in w_list) == 1

        # Link Diversity Constraint: Try not to share links unless necessary (Relaxed)
        for l in active:
            prob += pulp.lpSum(f[i][l][w][0] + f[i][l][w][1] for w in w_list) <= 1

    # Capacity Constraint: Fiber can't exceed wavelength limit
    for l in active:
        for w in w_list:
            prob += pulp.lpSum(f[i][l][w][0] + f[i][l][w][1] for i in range(len(d_list))) <= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    res = []
    if pulp.LpStatus[prob.status] == 'Optimal':
        for i, d in enumerate(d_list):
            w_path = [l for l in active if any(pulp.value(f[i][l][w][0]) == 1 for w in w_list)]
            p_path = [l for l in active if any(pulp.value(f[i][l][w][1]) == 1 for w in w_list)]
            res.append({"Name": d['name'], "Working": w_path, "Protection": p_path})
    return res

# --- 5. الواجهة الرسومية والنتائج ---
results = solve_rwa_pro(failed_link if fail_active else None)

col1, col2 = st.columns([1, 1.3])

with col1:
    st.subheader("📋 Lightpath Services")
    if results:
        for r in results:
            with st.expander(f"🔹 {r['Name']}", expanded=True):
                st.info(f"**Primary:** {r['Working']}")
                st.warning(f"**Backup:** {r['Protection']}")
                # التحقق من حالة المسار عند القطع
                if fail_active:
                    is_hit = any(failed_link == l or (failed_link[1], failed_link[0]) == l for l in r['Working'])
                    if is_hit:
                        st.error("⚠️ Primary Path Cut! Traffic switched to Backup.")
                    else:
                        st.success("✅ Primary Path is still active.")
    else:
        st.error("🚨 Configuration Error: No valid paths found. Adjust Wavelengths or Source/Dest.")

with col2:
    st.subheader("🌐 Network Intelligence Map")
    G = nx.Graph(); G.add_edges_from(base_links)
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='#111', edgecolors='#00defa', ax=ax)
    nx.draw_networkx_labels(G, pos, font_color='white', font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=base_links, edge_color='#333', style='dashed', alpha=0.3, ax=ax)
    
    if fail_active:
        nx.draw_networkx_edges(G, pos, edgelist=[failed_link], edge_color='red', width=7, ax=ax)

    colors = ['#00FF00', '#0099FF', '#FFCC00', '#FF00FF']
    for i, r in enumerate(results):
        c = colors[i % len(colors)]
        nx.draw_networkx_edges(G, pos, edgelist=r['Working'], edge_color=c, width=4.5, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=r['Protection'], edge_color=c, width=2, style='dotted', ax=ax)

    ax.set_facecolor('#0E1117'); fig.patch.set_facecolor('#0E1117'); plt.axis('off')
    st.pyplot(fig)
