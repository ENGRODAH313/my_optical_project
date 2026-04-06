import streamlit as st
import pulp
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. إعدادات الهوية وصفحة الترحيب ---
st.set_page_config(page_title="Optical Network System", layout="wide")

if 'entered' not in st.session_state:
    st.session_state.entered = False

if not st.session_state.entered:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_t1, col_t2, col_t3 = st.columns([1, 2, 1])
    with col_t2:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
        st.title("نظام إدارة وتخطيط الشبكات الضوئية")
        st.subheader("Technological University")
        st.write("### كلية: هندسة الاتصالات ضوئية")
        st.write("### الاشراف: د. احمد جياد كاظم")
        st.markdown("---")
        st.info(f"إعداد الطالب: رضا صلاح الدين  مصطفى صلاح")
        if st.button("🚀 الدخول إلى النظام (Enter Dashboard)"):
            st.session_state.entered = True
            st.rerun()
    st.stop()

# --- 2. المحاكاة والبيانات ---
nodes = [1, 2, 3, 4, 5, 6, 7]
link_distances = {(1,2): 120, (2,3): 150, (3,4): 100, (4,5): 200, (5,6): 130, (6,7): 90, (7,1): 180, (2,7): 110, (3,6): 140, (1,4): 300}
all_links = list(link_distances.keys()) + [(v, u) for u, v in link_distances.keys()]
full_dist = {**link_distances, **{(v, u): d for (u, v), d in link_distances.items()}}
SPEED_OF_LIGHT = 200 # km/ms

if 'demands' not in st.session_state:
    st.session_state.demands = [{"src": 1, "dst": 4, "name": "Backbone_01"}]

# --- 3. Sidebar (التحكم) ---
st.sidebar.header("⚙️ Control Panel")
waves_limit = st.sidebar.slider("Wavelengths Capacity", 1, 40, 20)

with st.sidebar.expander("➕ Add New Service"):
    s_node = st.selectbox("Source", nodes, index=0)
    d_node = st.selectbox("Destination", nodes, index=3)
    s_name = st.text_input("Name", f"Svc_{len(st.session_state.demands)+1}")
    if st.sidebar.button("Add"):
        st.session_state.demands.append({"src": s_node, "dst": d_node, "name": s_name})

st.sidebar.header("⚠️ Failure Simulation")
fail_on = st.sidebar.checkbox("Cut Fiber Link")
cut_link = st.sidebar.selectbox("Select Link", list(link_distances.keys()))

# --- 4. RWA Engine ---
def solve_rwa(fail=None):
    active = [l for l in all_links if l != fail and (l[1], l[0]) != fail]
    d_list = st.session_state.demands
    w_list = list(range(1, waves_limit + 1))
    if not d_list: return []
    prob = pulp.LpProblem("RWA", pulp.LpMinimize)
    f = pulp.LpVariable.dicts("flow", (range(len(d_list)), active, w_list, [0, 1]), 0, 1, pulp.LpInteger)
    prob += pulp.lpSum(f[i][l][w][t] * full_dist[l] for i in range(len(d_list)) for l in active for w in w_list for t in [0,1])
    for i in range(len(d_list)):
        s, d = d_list[i]['src'], d_list[i]['dst']
        for t in [0,1]:
            for n in nodes:
                in_f = pulp.lpSum(f[i][l][w][t] for l in active if l[1]==n for w in w_list)
                out_f = pulp.lpSum(f[i][l][w][t] for l in active if l[0]==n for w in w_list)
                if n == s: prob += out_f - in_f == 1
                elif n == d: prob += in_f - out_f == 1
                else: prob += in_f - out_f == 0
        for l in active: prob += pulp.lpSum(f[i][l][w][0] + f[i][l][w][1] for w in w_list) <= 1
    for l in active:
        for w in w_list: prob += pulp.lpSum(f[i][l][w][t] for i in range(len(d_list)) for t in [0,1]) <= 1
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    res = []
    if pulp.LpStatus[prob.status] == 'Optimal':
        for i, d in enumerate(d_list):
            w_p = [l for l in active if any(pulp.value(f[i][l][w][0])==1 for w in w_list)]
            p_p = [l for l in active if any(pulp.value(f[i][l][w][1])==1 for w in w_list)]
            dist = sum(full_dist[lx] for lx in w_p)
            res.append({"Name": d['name'], "Working": w_p, "Protection": p_p, "Latency": round(dist/SPEED_OF_LIGHT, 2)})
    return res

results = solve_rwa(cut_link if fail_on else None)

# --- 5. Main Dashboard View ---
# Top Metrics
m1, m2, m3 = st.columns(3)
m1.metric("Total Services", len(st.session_state.demands))
m2.metric("Network Status", "Stable" if not fail_on else "DEGRADED", delta="- Fiber Cut" if fail_on else None)
m3.metric("Wavelength Capacity", f"{waves_limit} λ")

st.divider()

c1, c2 = st.columns([1, 1.3])
with c1:
    st.subheader("📋 Analytics Report")
    if results:
        df = pd.DataFrame(results)
        st.dataframe(df[["Name", "Latency"]], use_container_width=True)
        # Download Button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export CSV Report", data=csv, file_name='network_report.csv', mime='text/csv')
    else:
        st.error("Capacity Reached!")

with c2:
    st.subheader("🌐 Real-time Topology")
    fig, ax = plt.subplots(figsize=(10, 7))
    G = nx.Graph(); G.add_edges_from(list(link_distances.keys()))
    pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='#111', edgecolors='#00defa', ax=ax)
    nx.draw_networkx_labels(G, pos, font_color='white', font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=list(link_distances.keys()), edge_color='#333', style='dashed', alpha=0.3, ax=ax)
    if fail_on: nx.draw_networkx_edges(G, pos, edgelist=[cut_link], edge_color='red', width=6, ax=ax)
    
    colors = ['#00FF00', '#0099FF', '#FFCC00', '#FF00FF']
    for i, r in enumerate(results):
        c = colors[i % len(colors)]
        nx.draw_networkx_edges(G, pos, edgelist=r['Working'], edge_color=c, width=4, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=r['Protection'], edge_color=c, width=2, style='dotted', ax=ax)
    
    ax.set_facecolor('#0E1117'); fig.patch.set_facecolor('#0E1117'); plt.axis('off')
    st.pyplot(fig)
