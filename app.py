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
    # واجهة الترحيب
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_t1, col_t2, col_t3 = st.columns([1, 2, 1])
    
    with col_t2:
        # إضافة شعار بسيط
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
        
        # المعلومات التي طلبتها بالضبط
        st.title("نظام إدارة وتخطيط الشبكات الضوئية")
        st.subheader("Technological University")
        st.write("### كلية: هندسة الاتصالات ضوئية")
        st.write("### الاشراف: د. احمد جياد كاظم")
        
        st.markdown("---")
        st.info(f"إعداد الطالب: رضا صلاح الدين , مصطفى صلاح")
        
        if st.button("🚀 الدخول إلى النظام (Enter Dashboard)"):
            st.session_state.entered = True
            st.rerun()
    st.stop()

# --- 2. الكود الأساسي للمحاكاة (يظهر بعد الدخول) ---
st.sidebar.success("✅ تم تسجيل الدخول - الجامعة التكنولوجية")

# بيانات الشبكة
nodes = [1, 2, 3, 4, 5, 6, 7]
link_distances = {
    (1,2): 120, (2,3): 150, (3,4): 100, (4,5): 200, 
    (5,6): 130, (6,7): 90, (7,1): 180, (2,7): 110, 
    (3,6): 140, (1,4): 300
}
base_links = list(link_distances.keys())
all_links = base_links + [(v, u) for u, v in base_links]
full_dist = {**link_distances, **{(v, u): d for (u, v), d in link_distances.items()}}

if 'demands' not in st.session_state:
    st.session_state.demands = [{"src": 1, "dst": 4, "name": "Main_Trunk_01"}]

# القائمة الجانبية
st.sidebar.header("⚙️ إعدادات الموارد")
waves_limit = st.sidebar.slider("عدد الأطوال الموجية (Wavelengths)", 1, 30, 15)

with st.sidebar.expander("➕ إضافة خدمة مرور جديدة"):
    s_node = st.selectbox("من العقدة", nodes, index=0)
    d_node = st.selectbox("إلى العقدة", nodes, index=4)
    s_name = st.text_input("اسم الخدمة", f"Svc_{len(st.session_state.demands)+1}")
    if st.sidebar.button("تفعيل الخدمة"):
        if s_node != d_node:
            st.session_state.demands.append({"src": s_node, "dst": d_node, "name": s_name})

if st.sidebar.button("🗑️ مسح كافة البيانات"):
    st.session_state.demands = []

st.sidebar.header("💥 اختبار الصمود (Stress Test)")
fail_on = st.sidebar.checkbox("محاكاة قطع ليف ضوئي")
cut_link = st.sidebar.selectbox("اختر الوصلة المقطوعة", base_links)

# محرك الحل الرياضي (RWA Engine)
def solve_final_rwa(fail=None):
    active = [l for l in all_links if l != fail and (l[1], l[0]) != fail]
    d_list = st.session_state.demands
    w_list = list(range(1, waves_limit + 1))
    if not d_list: return []

    prob = pulp.LpProblem("RWA_Final", pulp.LpMinimize)
    f = pulp.LpVariable.dicts("flow", (range(len(d_list)), active, w_list, [0, 1]), 0, 1, pulp.LpInteger)

    # توزيع المسارات وتقليل التأخير
    prob += pulp.lpSum(f[i][l][w][0] * full_dist[l] for i in range(len(d_list)) for l in active for w in w_list) + \
            pulp.lpSum(f[i][l][w][1] * full_dist[l] * 1.5 for i in range(len(d_list)) for l in active for w in w_list)

    for i in range(len(d_list)):
        s, d = d_list[i]['src'], d_list[i]['dst']
        for t in [0, 1]:
            for n in nodes:
                in_f = pulp.lpSum(f[i][l][w][t] for l in active if l[1] == n for w in w_list)
                out_f = pulp.lpSum(f[i][l][w][t] for l in active if l[0] == n for w in w_list)
                if n == s: prob += out_f - in_f == 1
                elif n == d: prob += in_f - out_f == 1
                else: prob += in_f - out_f == 0
            prob += pulp.lpSum(f[i][l][w][t] for l in active if l[0] == s for w in w_list) == 1
        
        for l in active:
            prob += pulp.lpSum(f[i][l][w][0] + f[i][l][w][1] for w in w_list) <= 1

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

# العرض والنتائج
results = solve_final_rwa(cut_link if fail_on else None)
c1, c2 = st.columns([1, 1.4])

with c1:
    st.subheader(" حالة المسارات")
    if results:
        for r in results:
            with st.expander(f"🔹 Service: {r['Name']}", expanded=True):
                st.write(f"**Primary Path:** {r['Working']}")
                st.write(f"**Backup Path:** {r['Protection']}")
                if fail_on and any(cut_link == l or (cut_link[1], cut_link[0]) == l for l in r['Working']):
                    st.error(" فشل المسار الأساسي! تم التحويل للاحتياطي.")
    else:
        st.error(" لا توجد موارد كافية. يرجى زيادة الأطوال الموجية.")

with c2:
    st.subheader("🌐 خريطة الشبكة")
    G = nx.Graph(); G.add_edges_from(base_links)
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='#111', edgecolors='#00defa', ax=ax)
    nx.draw_networkx_labels(G, pos, font_color='white', font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=base_links, edge_color='#333', style='dashed', alpha=0.3, ax=ax)
    
    if fail_on:
        nx.draw_networkx_edges(G, pos, edgelist=[cut_link], edge_color='red', width=7, ax=ax)

    colors = ['#00FF00', '#0099FF', '#FFCC00', '#FF00FF']
    for i, r in enumerate(results):
        c = colors[i % len(colors)]
        nx.draw_networkx_edges(G, pos, edgelist=r['Working'], edge_color=c, width=4.5, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=r['Protection'], edge_color=c, width=2.5, style='dotted', ax=ax)

    ax.set_facecolor('#0E1117'); fig.patch.set_facecolor('#0E1117'); plt.axis('off')
    st.pyplot(fig)
