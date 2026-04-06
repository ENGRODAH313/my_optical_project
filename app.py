import streamlit as st
import pulp
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. إعدادات الصفحة والترحيب ---
st.set_page_config(page_title="Optical survivability Suite", layout="wide")

# استخدام Session State للتحكم في ظهور صفحة الترحيب
if 'entered' not in st.session_state:
    st.session_state.entered = False

if not st.session_state.entered:
    # --- واجهة الترحيب (Landing Page) ---
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.balloons() # تأثير احتفالي بسيط عند الفتح
    
    col_t1, col_t2, col_t3 = st.columns([1, 2, 1])
    with col_t2:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100) # أيقونة تعبر عن الألياف
        st.title(" مشروع نظام إدارة الشبكات الضوئية")
        st.header("إعداد الطالب: رضا صلاح الدين ,مصطفى صلاح ")
        st.subheader("Technological University
        كلية: هندسة الاتصالات ضوئية 
        الاشراف: د. احمد جياد كاظم")
        st.markdown("""
        ---
        **وصف المشروع:**
        نظام محاكاة ذكي لتخطيط المسارات وتوزيع الأطوال الموجية (RWA) 
        مع تفعيل خاصية الحماية الذاتية (1+1 Protection) لضمان استمرارية الخدمة.
        """)
        
        if st.button("(Enter Dashboard)"):
            st.session_state.entered = True
            st.rerun()
    st.stop() # إيقاف التنفيذ هنا حتى يضغط المستخدم على الزر

# --- 2. محتوى المشروع الأساسي (يظهر بعد الضغط على الزر) ---
st.title(" لوحة تحكم الشبكة الضوئية الذكية")
st.sidebar.success("تم تسجيل الدخول بنجاح")

# --- بيانات الشبكة والفيزياء ---
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

if 'demands' not in st.session_state:
    st.session_state.demands = [{"src": 1, "dst": 4, "name": "Main_Trunk_01"}]

# --- القائمة الجانبية ---
st.sidebar.header("⚙️ التحكم بالموارد")
waves_limit = st.sidebar.slider("عدد الأطوال الموجية لكل ليف", 1, 25, 15)

with st.sidebar.expander("➕ إضافة خدمة محمية"):
    s_n = st.selectbox("من العقدة", nodes, index=0)
    d_n = st.selectbox("إلى العقدة", nodes, index=4)
    s_name = st.text_input("اسم الخدمة", f"Service_{len(st.session_state.demands)+1}")
    if st.sidebar.button("تفعيل المسار"):
        if s_n != d_n:
            st.session_state.demands.append({"src": s_n, "dst": d_n, "name": s_name})

if st.sidebar.button("تفريغ الشبكة"):
    st.session_state.demands = []

st.sidebar.header(" محاكاة قطع الألياف")
fail_active = st.sidebar.checkbox("تفعيل قطع في الوصلة")
failed_link = st.sidebar.selectbox("اختر الوصلة المقطوعة", base_links)

# --- محرك الحل الرياضي ---
def solve_rwa_pro(cut=None):
    active = [l for l in all_links if l != cut and (l[1], l[0]) != cut]
    d_list = st.session_state.demands
    w_list = list(range(1, waves_limit + 1))
    if not d_list: return []

    prob = pulp.LpProblem("RWA_Final", pulp.LpMinimize)
    f = pulp.LpVariable.dicts("flow", (range(len(d_list)), active, w_list, [0, 1]), 0, 1, pulp.LpInteger)

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

# --- العرض المرئي ---
results = solve_rwa_pro(failed_link if fail_active else None)
col1, col2 = st.columns([1, 1.3])

with col1:
    st.subheader("📋 حالة الخدمات")
    if results:
        for r in results:
            with st.expander(f"🔹 {r['Name']}", expanded=True):
                st.write(f"✅ الأساسي: {r['Working']}")
                st.write(f"🛡️ الاحتياطي: {r['Protection']}")
                if fail_active:
                    if any(failed_link == l or (failed_link[1], failed_link[0]) == l for l in r['Working']):
                        st.error("تم القطع! التحويل للاحتياطي جاري...")
    else:
        st.error("السعة غير كافية، يرجى زيادة الأطوال الموجية")

with col2:
    st.subheader("🌐 خريطة الشبكة الذكية")
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
        nx.draw_networkx_edges(G, pos, edgelist=r['Working'], edge_color=c, width=4, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=r['Protection'], edge_color=c, width=2, style='dotted', ax=ax)

    ax.set_facecolor('#0E1117'); fig.patch.set_facecolor('#0E1117'); plt.axis('off')
    st.pyplot(fig)
