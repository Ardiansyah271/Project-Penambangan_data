"""ClusterMhs v2 - Website Clustering Mahasiswa Tanjungpinang (K-Means).
Jalankan:  pip install streamlit pandas numpy altair   lalu   streamlit run app_web.py
Seluruh parameter model (centroid, scaler, titik data) sudah tertanam di file ini.
"""
import html
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# ======================= PARAMETER MODEL (hasil Colab) =======================
M = {'akt': [[19.0, 7.6, 3.4, 3.3, 11.0], [17.7, 8.2, 3.8, 3.2, 10.9], [20.6, 8.7, 4.2, 3.6, 10.5]],
 'cap': 592500.0,
 'cent': [[-0.444, 0.841], [-0.178, -0.925], [1.67, 0.149]],
 'ipk': [3.75, 3.24, 3.55],
 'komp': [[0.028, 0.139, 0.048, 0.013, 0.475, 0.133, 0.106, 0.042],
          [0.012, 0.126, 0.066, 0.009, 0.422, 0.129, 0.173, 0.064],
          [0.039, 0.136, 0.048, 0.016, 0.336, 0.107, 0.175, 0.143]],
 'mean': [257023.03, 3.51],
 'n': [65, 63, 24],
 'names': {0: 'Hemat & Berprestasi', 1: 'Hemat, Perlu Pendampingan', 2: 'Pengeluaran Tinggi & Berprestasi'},
 'peng': [199000, 234000, 555000],
 'pts': [[295, 3.35, 1], [455, 3.61, 2], [359, 3.44, 2], [169, 2.89, 1], [333, 3.85, 0], [392, 3.32, 2],
         [202, 3.27, 1], [92, 3.2, 1], [190, 3.93, 0], [350, 3.32, 1], [65, 3.62, 0], [338, 3.16, 1],
         [436, 3.71, 2], [258, 3.7, 0], [330, 4.0, 0], [373, 3.58, 2], [128, 3.45, 1], [499, 4.0, 2],
         [86, 3.63, 0], [131, 3.77, 0], [124, 3.3, 1], [299, 3.3, 1], [315, 3.75, 0], [268, 3.41, 1],
         [127, 3.74, 0], [277, 3.36, 1], [74, 3.61, 0], [518, 3.34, 2], [94, 3.66, 0], [277, 3.41, 1],
         [130, 3.7, 0], [139, 3.57, 0], [478, 3.41, 2], [76, 3.72, 0], [245, 3.44, 1], [338, 3.29, 1],
         [204, 3.33, 1], [202, 3.7, 0], [276, 3.4, 1], [163, 3.9, 0], [237, 3.63, 0], [455, 3.59, 2],
         [374, 2.84, 1], [394, 3.5, 2], [146, 3.35, 1], [306, 3.79, 0], [289, 3.44, 1], [291, 3.39, 1],
         [138, 3.8, 0], [97, 3.33, 1], [219, 3.39, 1], [101, 3.38, 1], [294, 3.1, 1], [259, 2.96, 1],
         [214, 3.2, 1], [129, 2.92, 1], [440, 3.57, 2], [249, 3.23, 1], [228, 3.3, 1], [320, 3.47, 1],
         [151, 2.96, 1], [324, 3.42, 1], [388, 2.84, 1], [592, 3.0, 2], [378, 3.42, 2], [148, 3.7, 0],
         [270, 3.8, 0], [288, 3.39, 1], [95, 3.55, 0], [277, 2.98, 1], [277, 3.68, 0], [193, 3.65, 0],
         [159, 3.12, 1], [229, 3.64, 0], [342, 3.61, 2], [315, 3.25, 1], [422, 3.52, 2], [512, 3.53, 2],
         [118, 3.88, 0], [166, 3.9, 0], [185, 3.57, 0], [324, 3.48, 1], [230, 3.14, 1], [592, 3.8, 2],
         [135, 3.65, 0], [22, 3.82, 0], [245, 3.41, 1], [215, 3.11, 1], [0, 4.0, 0], [335, 3.75, 0],
         [70, 3.8, 0], [70, 4.0, 0], [200, 3.35, 1], [157, 3.65, 0], [131, 4.0, 0], [340, 3.85, 0],
         [259, 3.93, 0], [290, 3.56, 0], [270, 3.0, 1], [210, 2.8, 1], [25, 3.97, 0], [592, 3.8, 2],
         [380, 3.75, 2], [241, 3.38, 1], [143, 3.88, 0], [15, 3.83, 0], [151, 3.4, 1], [170, 3.96, 0],
         [212, 3.13, 1], [338, 3.06, 1], [592, 3.8, 2], [331, 3.95, 0], [220, 3.4, 1], [121, 3.45, 1],
         [111, 3.6, 0], [139, 3.34, 1], [113, 3.06, 1], [592, 3.42, 2], [108, 3.25, 1], [91, 3.5, 0],
         [434, 2.94, 1], [133, 3.2, 1], [96, 3.69, 0], [117, 3.3, 1], [228, 3.42, 1], [270, 3.32, 1],
         [163, 3.52, 0], [143, 3.25, 1], [265, 3.8, 0], [235, 3.2, 1], [290, 3.8, 0], [295, 3.7, 0],
         [339, 3.8, 0], [298, 3.6, 0], [592, 3.54, 2], [592, 3.57, 2], [247, 3.63, 0], [334, 3.67, 0],
         [255, 3.75, 0], [300, 3.89, 0], [315, 3.28, 1], [294, 3.7, 0], [299, 3.5, 1], [303, 3.78, 0],
         [277, 3.67, 0], [287, 3.66, 0], [259, 3.83, 0], [280, 3.8, 0], [280, 3.98, 0], [210, 2.78, 1],
         [460, 3.42, 2], [280, 3.59, 0]],
 'std': [131500.37, 0.29]}
NAMA = M["names"]
K = len(NAMA)
COL = ["#5B8DEF", "#F2994A", "#9B7BEA", "#4CC2A8", "#E86F8A"][:K]
KAT = ["Bensin", "Transportasi", "Pulsa/Langganan", "Akademik", "Makan", "Minum/Ngopi", "Jajan", "Lainnya"]
HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
AKT = ["Kuliah", "Belajar mandiri", "Organisasi", "Bekerja", "Kegiatan lain"]
SARAN = {
    0: "Pola keuangan dan akademik sudah baik. Pertahankan, dan Anda bisa menjadi mentor sebaya.",
    1: "Pengeluaran moderat namun IPK di bawah rata-rata. Tambah jam belajar dan cari pendampingan akademik.",
    2: "Prestasi baik, tetapi pengeluaran tinggi. Susun anggaran mingguan dan kurangi jajan/kategori lainnya.",
}
DF = pd.DataFrame(M["pts"], columns=["pengeluaran", "ipk", "cluster"])
DF["nama"] = DF["cluster"].map(NAMA)
CENT = pd.DataFrame({
    "pengeluaran": [c[0] * M["std"][0] / 1000 + M["mean"][0] / 1000 for c in M["cent"]],
    "ipk": [c[1] * M["std"][1] + M["mean"][1] for c in M["cent"]],
    "nama": [NAMA[i] for i in range(K)]})


def klasifikasi(total, ipk):
    """Nearest-centroid pada skala z-score (identik dengan K-Means.predict)."""
    z = np.array([(min(total, M["cap"]) - M["mean"][0]) / M["std"][0],
                  (ipk - M["mean"][1]) / M["std"][1]])
    d = np.linalg.norm(np.array(M["cent"]) - z, axis=1)
    w = 1 / (d + 1e-6)
    return int(d.argmin()), w / w.sum()


def persentil(v, kolom):
    return int(round((DF[kolom] <= v).mean() * 100))


def rp(n):
    return "Rp" + f"{int(round(n)):,}".replace(",", ".")


# ================================ TAMPILAN ===================================
st.set_page_config(page_title="ClusterMhs", page_icon="📊", layout="wide")
st.markdown("""<style>
:root{--ac:#5B8DEF;--ac2:#E4ECFD;--tx:#1E2A44;--mut:#6B7A99;--bd:#E3EAF7}
.stApp{background:#EEF3FC;color:var(--tx)}
header[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:1.6rem;max-width:1250px}
section[data-testid="stSidebar"]{background:#CFE0FB;border-radius:0 24px 24px 0}
section[data-testid="stSidebar"] *{color:#1E2A44!important}
div[role="radiogroup"] label{background:transparent;border-radius:14px;padding:10px 14px;margin-bottom:4px;width:100%}
div[role="radiogroup"] label:has(input:checked){background:#fff;box-shadow:0 6px 24px rgba(91,141,239,.15)}
div[role="radiogroup"] label>div:first-child{display:none}
h1,h2,h3,p,label,span{color:var(--tx)}
.card{background:#fff;border-radius:20px;padding:18px 20px;box-shadow:0 6px 24px rgba(91,141,239,.12);border:1px solid var(--bd);margin-bottom:14px}
.card h4{margin:0 0 8px;font-size:14px;color:var(--tx)}
.stat .ic{width:38px;height:38px;border-radius:12px;background:var(--ac2);display:grid;place-items:center;font-size:18px}
.stat b{display:block;font-size:26px;margin-top:8px;color:var(--tx)}.stat span{color:var(--mut);font-size:12px}
.pill{display:inline-block;padding:3px 12px;border-radius:99px;font-size:12px;font-weight:600;color:#fff!important}
.bar{height:8px;border-radius:9px;background:var(--ac2);overflow:hidden;margin:3px 0 9px}.bar i{display:block;height:100%}
.row{display:flex;justify-content:space-between;font-size:13px;color:var(--tx)}
.stack{display:flex;height:10px;border-radius:9px;overflow:hidden;margin:6px 0}
.mut{color:var(--mut)!important;font-size:13px}
.stButton>button,.stFormSubmitButton>button,.stDownloadButton>button{background:var(--ac);color:#fff;border:0;border-radius:12px;padding:.5rem 1.4rem;font-weight:600}
.stNumberInput input{background:#F5F8FE!important;color:var(--tx)!important;border-radius:12px}
</style>""", unsafe_allow_html=True)


st.markdown('<style>\n.stApp::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;\nbackground:url("data:image/svg+xml,%3Csvg%20xmlns%3D%27http%3A//www.w3.org/2000/svg%27%20width%3D%27900%27%20height%3D%27700%27%20viewBox%3D%270%200%20900%20700%27%20fill%3D%27none%27%20stroke%3D%27%235B8DEF%27%20stroke-opacity%3D%27.28%27%20stroke-width%3D%271.5%27%3E%3Cpath%20d%3D%27M520%2080L640%20150L770%2090L850%20210L700%20270L640%20150M700%20270L560%20340L430%20260L520%2080M560%20340L640%20470L780%20420L850%20210M430%20260L330%20380L460%20480L560%20340M640%20470L600%20600L720%20640L780%20420M330%20380L200%20340%27/%3E%3Cg%20fill%3D%27%235B8DEF%27%20fill-opacity%3D%27.35%27%20stroke%3D%27none%27%3E%3Ccircle%20cx%3D%27520%27%20cy%3D%2780%27%20r%3D%278%27/%3E%3Ccircle%20cx%3D%27640%27%20cy%3D%27150%27%20r%3D%2711%27/%3E%3Ccircle%20cx%3D%27770%27%20cy%3D%2790%27%20r%3D%277%27/%3E%3Ccircle%20cx%3D%27850%27%20cy%3D%27210%27%20r%3D%279%27/%3E%3Ccircle%20cx%3D%27700%27%20cy%3D%27270%27%20r%3D%2712%27/%3E%3Ccircle%20cx%3D%27560%27%20cy%3D%27340%27%20r%3D%279%27/%3E%3Ccircle%20cx%3D%27430%27%20cy%3D%27260%27%20r%3D%277%27/%3E%3Ccircle%20cx%3D%27330%27%20cy%3D%27380%27%20r%3D%278%27/%3E%3Ccircle%20cx%3D%27460%27%20cy%3D%27480%27%20r%3D%277%27/%3E%3Ccircle%20cx%3D%27640%27%20cy%3D%27470%27%20r%3D%2710%27/%3E%3Ccircle%20cx%3D%27780%27%20cy%3D%27420%27%20r%3D%278%27/%3E%3Ccircle%20cx%3D%27600%27%20cy%3D%27600%27%20r%3D%277%27/%3E%3Ccircle%20cx%3D%27720%27%20cy%3D%27640%27%20r%3D%279%27/%3E%3Ccircle%20cx%3D%27200%27%20cy%3D%27340%27%20r%3D%276%27/%3E%3C/g%3E%3C/svg%3E") no-repeat right -80px top 10px/780px,url("data:image/svg+xml,%3Csvg%20xmlns%3D%27http%3A//www.w3.org/2000/svg%27%20width%3D%27800%27%20height%3D%27600%27%20viewBox%3D%270%200%20800%20600%27%20fill%3D%27none%27%20stroke%3D%27%239B7BEA%27%20stroke-opacity%3D%27.30%27%20stroke-width%3D%271.6%27%3E%3Cpath%20d%3D%27M0%20420H140V360H280V300H400M0%20500H200V450H340V390H470M0%20340H80V270H210M120%20600V540H260V500H380M300%20600V560H520V470%27/%3E%3Crect%20x%3D%27400%27%20y%3D%27270%27%20width%3D%2760%27%20height%3D%2760%27%20rx%3D%278%27/%3E%3Crect%20x%3D%27470%27%20y%3D%27360%27%20width%3D%2746%27%20height%3D%2746%27%20rx%3D%277%27/%3E%3Crect%20x%3D%27210%27%20y%3D%27240%27%20width%3D%2740%27%20height%3D%2740%27%20rx%3D%276%27/%3E%3Cg%20fill%3D%27%239B7BEA%27%20fill-opacity%3D%27.35%27%20stroke%3D%27none%27%3E%3Ccircle%20cx%3D%27140%27%20cy%3D%27360%27%20r%3D%276%27/%3E%3Ccircle%20cx%3D%27280%27%20cy%3D%27300%27%20r%3D%276%27/%3E%3Ccircle%20cx%3D%27340%27%20cy%3D%27390%27%20r%3D%276%27/%3E%3Ccircle%20cx%3D%27520%27%20cy%3D%27470%27%20r%3D%277%27/%3E%3Ccircle%20cx%3D%2780%27%20cy%3D%27270%27%20r%3D%276%27/%3E%3C/g%3E%3C/svg%3E") no-repeat left -60px bottom -30px/640px}\n.block-container{position:relative;z-index:1}\n.blob{position:fixed;border-radius:50%;filter:blur(70px);opacity:.38;z-index:0;pointer-events:none;animation:fl 16s ease-in-out infinite alternate}\n.b1{width:340px;height:340px;background:#9DBBFA;top:-80px;left:30%}.b2{width:300px;height:300px;background:#C9B8F7;bottom:-60px;right:8%;animation-delay:-6s}\n@keyframes fl{to{transform:translate(40px,30px) scale(1.15)}}\n@keyframes up{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}\n@keyframes pulse{50%{opacity:.35;transform:scale(1.4)}}\n.card{animation:up .55s ease both;transition:transform .25s,box-shadow .25s}\n.card:hover{transform:translateY(-3px);box-shadow:0 12px 32px rgba(91,141,239,.22)}\n.hero{background:linear-gradient(120deg,#5B8DEF,#86A9F6 55%,#B6A3F4);border-radius:28px;padding:30px 34px;display:flex;justify-content:space-between;align-items:center;gap:20px;overflow:hidden;box-shadow:0 14px 40px rgba(91,141,239,.35);margin-bottom:16px;animation:up .6s ease both}\n.hero *{color:#fff!important}.hero h1{font-size:28px;line-height:1.25;margin:10px 0 8px;max-width:560px}.hero p{margin:0;opacity:.92;max-width:520px}\n.tag{background:rgba(255,255,255,.22);border-radius:99px;padding:4px 14px;font-size:12px;font-weight:600}\n.hero circle.n{animation:pulse 2.4s ease-in-out infinite;transform-box:fill-box;transform-origin:center}\n.ins{display:flex;gap:12px;align-items:center}.ins .ic{flex:none}.ins b{font-size:15px}\n.step{display:flex;gap:14px;align-items:flex-start}.step .no{flex:none;width:34px;height:34px;border-radius:50%;background:var(--ac);color:#fff;display:grid;place-items:center;font-weight:700}\n.chip{display:inline-block;padding:2px 10px;border-radius:8px;font-size:12px;font-weight:600;background:var(--ac2);color:var(--tx)}\nbutton[kind="secondary"]{border-radius:12px!important;border:1px solid #D5E1F8!important;background:#fff!important;color:#1E2A44!important}\nbutton[kind="secondary"]:hover{border-color:#5B8DEF!important;color:#5B8DEF!important}\nbutton[data-baseweb="tab"]{font-weight:600}\n@media(max-width:800px){.hero svg{display:none}.hero h1{font-size:22px}}\n</style><div class="blob b1"></div><div class="blob b2"></div>', unsafe_allow_html=True)


def card(inner, extra=""):
    st.markdown(f'<div class="card" {extra}>{inner}</div>', unsafe_allow_html=True)


def pill(c):
    return f'<span class="pill" style="background:{COL[c]}">Cluster {c}</span>'


def judul(t, s):
    st.markdown(f"<h1 style='margin:0;font-size:26px'>{t}</h1><p class='mut' style='margin:2px 0 16px'>{s}</p>",
                unsafe_allow_html=True)


def peta(df, user=None, tinggi=330, sim=None):
    skala = alt.Scale(domain=[NAMA[i] for i in range(K)], range=COL)
    pts = alt.Chart(df).mark_circle(size=70, opacity=.78).encode(
        x=alt.X("pengeluaran:Q", title="Pengeluaran per minggu (ribu Rp)"),
        y=alt.Y("ipk:Q", title="IPK", scale=alt.Scale(zero=False)),
        color=alt.Color("nama:N", scale=skala, legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["nama", "pengeluaran", "ipk"])
    ctr = alt.Chart(CENT).mark_point(shape="diamond", size=260, filled=True, color="#F5C518",
                                     stroke="#333", strokeWidth=1).encode(x="pengeluaran:Q", y="ipk:Q",
                                                                          tooltip=["nama"])
    layers = [pts, ctr]
    if user:
        u = pd.DataFrame({"pengeluaran": [min(user[0], M["cap"]) / 1000], "ipk": [user[1]], "n": ["Anda"]})
        layers.append(alt.Chart(u).mark_point(shape="triangle-up", size=330, filled=True, color="#E5484D",
                                              stroke="#fff", strokeWidth=2).encode(x="pengeluaran:Q", y="ipk:Q",
                                                                                   tooltip=["n"]))
    if sim:
        (a, b) = sim
        L = pd.DataFrame({"x": [min(a[0], M["cap"]) / 1000, min(b[0], M["cap"]) / 1000], "y": [a[1], b[1]]})
        layers.append(alt.Chart(L).mark_line(color="#E5484D", strokeDash=[6, 4], strokeWidth=2).encode(x="x:Q", y="y:Q"))
        layers.append(alt.Chart(L.iloc[[1]]).mark_point(shape="cross", size=380, color="#16A34A", strokeWidth=4)
                      .encode(x="x:Q", y="y:Q"))
    st.altair_chart(alt.layer(*layers).properties(height=tinggi).configure_view(strokeWidth=0)
                    .configure(background="transparent"), use_container_width=True)


# ================================= FUNGSI BANTU ==============================
PRESET = {"Hemat": (3.8, [15000, 20000, 15000, 20000, 15000, 25000, 20000], [18, 10, 4, 0, 10]),
          "Sedang": (3.3, [35000, 40000, 35000, 30000, 45000, 30000, 35000], [18, 6, 3, 2, 10]),
          "Boros": (3.6, [80000, 90000, 70000, 85000, 100000, 120000, 110000], [20, 8, 4, 3, 10])}


def isi(n):
    i, d, j = PRESET[n]
    st.session_state["ipk"] = i
    for k in range(7):
        st.session_state[f"d{k}"] = d[k]
    for k in range(5):
        st.session_state[f"a{k}"] = float(j[k])
    st.toast(f"Contoh “{n}” dimuat", icon="✨")


def keyakinan(share):
    a, b = np.sort(share)[::-1][:2]
    return ("Tinggi", "#16A34A") if a - b > .4 else ("Sedang", "#F2994A") if a - b > .2 else ("Dekat batas cluster", "#E5484D")


def butuh(target, tot, ipk):
    o_ipk = next((round(float(x), 2) for x in np.arange(ipk, 4.001, .01) if klasifikasi(tot, x)[0] == target), None)
    o_p = next((int(t) for t in np.arange(tot, -1, -5000) if klasifikasi(t, ipk)[0] == target), None)
    return o_ipk, o_p


@st.cache_data
def metrik():
    Z = np.c_[(DF.pengeluaran * 1000 - M["mean"][0]) / M["std"][0], (DF.ipk - M["mean"][1]) / M["std"][1]]
    lab = DF.cluster.values
    D = np.linalg.norm(Z[:, None] - Z[None], axis=2)
    sil = []
    for i in range(len(Z)):
        sm = lab == lab[i]
        sm[i] = False
        a = D[i, sm].mean() if sm.any() else 0
        b = min(D[i, lab == c].mean() for c in set(lab) if c != lab[i])
        sil.append((b - a) / max(a, b))
    C = np.array(M["cent"])
    S = [np.linalg.norm(Z[lab == c] - C[c], axis=1).mean() for c in range(K)]
    db = np.mean([max((S[i] + S[j]) / np.linalg.norm(C[i] - C[j]) for j in range(K) if j != i) for i in range(K)])
    return float(np.mean(sil)), float(db)


HERO_SVG = """<svg width="300" height="170" viewBox="0 0 300 170" fill="none"><rect x="20" y="14" width="210" height="136" rx="16" fill="#fff" fill-opacity=".2" stroke="#fff" stroke-opacity=".7"/>
<rect x="38" y="104" width="22" height="30" rx="5" fill="#fff" fill-opacity=".85"/><rect x="70" y="80" width="22" height="54" rx="5" fill="#fff" fill-opacity=".85"/><rect x="102" y="92" width="22" height="42" rx="5" fill="#fff" fill-opacity=".85"/><rect x="134" y="58" width="22" height="76" rx="5" fill="#fff" fill-opacity=".85"/>
<path d="M38 70L84 48L122 62L170 30L214 40" stroke="#fff" stroke-width="3" stroke-linecap="round"/><circle class="n" cx="84" cy="48" r="5" fill="#fff"/><circle class="n" cx="170" cy="30" r="5" fill="#fff"/>
<circle cx="262" cy="54" r="22" fill="#fff" fill-opacity=".25" stroke="#fff"/><circle class="n" cx="262" cy="54" r="6" fill="#fff"/><circle cx="270" cy="124" r="16" fill="#fff" fill-opacity=".25" stroke="#fff"/><circle class="n" cx="270" cy="124" r="5" fill="#fff"/>
<path d="M230 70L252 62M230 110L256 120" stroke="#fff" stroke-width="2" stroke-dasharray="4 4"/></svg>"""

# ================================= NAVIGASI ==================================
MENU = {"🏠  Dashboard": "dash", "✨  Cek Cluster Saya": "cek", "🎯  Simulasi What-If": "sim",
        "🧩  Profil & Bandingkan": "profil", "🔎  Eksplorasi Data": "eks", "📘  Tentang Metode": "info"}
st.sidebar.markdown("### ◎ ClusterMhs")
st.sidebar.caption("Clustering Mahasiswa Tanjungpinang")
hal = MENU[st.sidebar.radio("Menu", list(MENU), label_visibility="collapsed")]
with st.sidebar.expander("💡 Panduan singkat"):
    st.markdown("1. **Cek Cluster Saya**: isi IPK & pengeluaran.\n2. **Simulasi**: coba ubah kebiasaan.\n"
                "3. **Profil**: pahami tiap kelompok.\n\n*Data bersifat anonim dan tidak dikirim ke server.*")
st.sidebar.caption("K-Means · Pengeluaran × IPK\n\nUniversitas Maritim Raja Ali Haji")
st.session_state.setdefault("hist", [])
st.session_state.setdefault("res", None)
if "ipk" not in st.session_state:
    isi("Sedang")

# ================================ DASHBOARD ==================================
if hal == "dash":
    st.markdown(f"""<div class="hero"><div><span class="tag">⚡ K-Means · Data Mining</span>
<h1>Kenali pola keuangan & prestasi akademik mahasiswa Tanjungpinang</h1>
<p>Temukan kelompokmu, bandingkan dengan {len(DF)} mahasiswa lain, dan simulasikan langkah menuju kebiasaan yang lebih baik.</p></div>{HERO_SVG}</div>""",
                unsafe_allow_html=True)
    info = [("👥", len(DF), "Total responden"), ("🎓", f"{DF.ipk.mean():.2f}", "Rata-rata IPK"),
            ("💸", rp(DF.pengeluaran.mean() * 1000), "Rata-rata pengeluaran/minggu"), ("🧩", K, "Cluster (K-Means)")]
    for kol, (i, v, t) in zip(st.columns(4), info):
        with kol:
            card(f'<div class="stat"><div class="ic">{i}</div><b>{v}</b><span>{t}</span></div>')
    best, boros, besar = int(np.argmax(M["ipk"])), int(np.argmax(M["peng"])), int(np.argmax(M["n"]))
    ins = [("🏆", "IPK tertinggi", f"{NAMA[best]} ({M['ipk'][best]})"),
           ("🔥", "Pengeluaran terbesar", f"{NAMA[boros]} ({rp(M['peng'][boros])}/mgg)"),
           ("👪", "Kelompok terbesar", f"{NAMA[besar]} ({M['n'][besar]} orang)")]
    for kol, (i, t, v) in zip(st.columns(3), ins):
        with kol:
            card(f'<div class="ins"><div class="ic" style="width:38px;height:38px;border-radius:12px;background:#E4ECFD;display:grid;place-items:center">{i}</div><div><span class="mut">{t}</span><br><b>{html.escape(v)}</b></div></div>')
    a, b = st.columns([1.7, 1])
    with a:
        card("<h4>Sebaran Cluster: Pengeluaran (X) vs IPK (Y)</h4><span class='mut'>Arahkan kursor ke titik untuk detail. ◆ = pusat cluster.</span>")
        peta(DF)
    with b:
        card("<h4>Proporsi Cluster</h4>")
        pie = pd.DataFrame({"nama": [NAMA[i] for i in range(K)], "n": M["n"]})
        st.altair_chart(alt.Chart(pie).mark_arc(innerRadius=62).encode(
            theta="n:Q", color=alt.Color("nama:N", scale=alt.Scale(domain=list(pie.nama), range=COL),
                                         legend=alt.Legend(title=None, orient="bottom")),
            tooltip=["nama", "n"]).properties(height=330).configure(background="transparent"),
            use_container_width=True)
    card("<h4>Rata-rata Jam Aktivitas per Minggu</h4>")
    long = pd.DataFrame([(AKT[j], NAMA[i], M["akt"][i][j]) for i in range(K) for j in range(5)],
                        columns=["aktivitas", "nama", "jam"])
    st.altair_chart(alt.Chart(long).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X("aktivitas:N", title=None, sort=AKT), xOffset="nama:N", y=alt.Y("jam:Q", title="Jam/minggu"),
        color=alt.Color("nama:N", scale=alt.Scale(domain=[NAMA[i] for i in range(K)], range=COL),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["nama", "aktivitas", "jam"]).properties(height=260).configure(background="transparent"),
        use_container_width=True)

# ============================== CEK CLUSTER ==================================
elif hal == "cek":
    judul("Cek Cluster Saya", "Masukkan data Anda, sistem menentukan kelompok yang paling sesuai")
    st.caption("Mulai cepat dengan contoh:")
    pc = st.columns([1, 1, 1, 4])
    for kol, n in zip(pc, PRESET):
        kol.button(f"{'🌱' if n == 'Hemat' else '⚖️' if n == 'Sedang' else '🔥'} {n}", on_click=isi, args=(n,), key="p" + n)
    kiri, kanan = st.columns(2)
    with kiri:
        with st.form("form"):
            ipk = st.number_input("IPK terakhir", 0.0, 4.0, step=0.01, key="ipk", help="Skala 0–4")
            st.markdown("**Pengeluaran per hari (Rp)**")
            kk = st.columns(4) + st.columns(3)
            peng = [kk[i].number_input(HARI[i], 0, 5_000_000, step=5000, key=f"d{i}") for i in range(7)]
            st.markdown("**Aktivitas per minggu (jam)** · opsional")
            ak = st.columns(5)
            jam = [ak[i].number_input(AKT[i], 0.0, 100.0, step=0.5, key=f"a{i}") for i in range(5)]
            kirim = st.form_submit_button("🔍 Kategorikan")
        if kirim:
            total = float(sum(peng))
            if total == 0:
                st.warning("Semua pengeluaran bernilai 0. Pastikan Anda mengisi pengeluaran yang sebenarnya.")
            if ipk == 0:
                st.warning("IPK bernilai 0. Periksa kembali isian Anda.")
            c, share = klasifikasi(total, ipk)
            st.session_state["res"] = dict(total=total, ipk=ipk, c=c, share=share, jam=jam)
            st.session_state["hist"].insert(0, {"IPK": ipk, "Pengeluaran/minggu": total, "Cluster": c, "Nama": NAMA[c]})
            st.toast(f"Anda masuk Cluster {c}: {NAMA[c]}", icon="🎉")
    R = st.session_state["res"]
    with kanan:
        if not R:
            card("<h4>Hasil</h4><p class='mut'>Isi data lalu tekan “Kategorikan”, atau pilih contoh di atas.</p>")
        else:
            c = R["c"]
            kt, kw = keyakinan(R["share"])
            bar = "".join(f'<div class="row"><span>{html.escape(NAMA[i])}</span><span>{R["share"][i]*100:.0f}%</span></div>'
                          f'<div class="bar"><i style="width:{R["share"][i]*100:.0f}%;background:{COL[i]}"></i></div>' for i in range(K))
            akt = "".join(f'<div class="row"><span>{AKT[j]}</span><span>{R["jam"][j]:g} jam · rata-rata {M["akt"][c][j]}</span></div>'
                          f'<div class="bar"><i style="width:{min(100, R["jam"][j]/max(R["jam"][j], M["akt"][c][j], 1)*100):.0f}%;background:{COL[c]}"></i></div>'
                          for j in range(5))
            cap = f" (dibatasi {rp(M['cap'])})" if R["total"] > M["cap"] else ""
            card(f"""<h4>Anda masuk…</h4>{pill(c)} <span class='chip' style='color:{kw}'>Keyakinan: {kt}</span>
<h2 style='margin:8px 0 2px'>{html.escape(NAMA[c])}</h2><p class='mut'>{rp(R['total'])}/minggu{cap} · IPK {R['ipk']}</p><p>{SARAN.get(c, '')}</p>
<div class='row'><span>Pengeluaran lebih tinggi dari</span><b>{persentil(R['total']/1000, 'pengeluaran')}% responden</b></div>
<div class='row'><span>IPK lebih tinggi dari</span><b>{persentil(R['ipk'], 'ipk')}% responden</b></div>
<h4 style='margin-top:14px'>Kedekatan ke tiap cluster</h4>{bar}
<h4 style='margin-top:10px'>Aktivitas Anda vs rata-rata cluster</h4>{akt}""", f'style="border-left:5px solid {COL[c]}"')
            st.info("Ingin tahu cara pindah cluster? Buka menu **🎯 Simulasi What-If**.")
    a, b = st.columns([1.7, 1])
    with a:
        card("<h4>Posisi Anda pada Peta Cluster</h4><span class='mut'>▲ merah = Anda</span>")
        peta(DF, (R["total"], R["ipk"]) if R else None)
    with b:
        card("<h4>Riwayat Pengecekan (sesi ini)</h4>")
        if st.session_state["hist"]:
            h = pd.DataFrame(st.session_state["hist"][:8])
            st.dataframe(h, hide_index=True, use_container_width=True)
            st.download_button("⬇ Unduh riwayat (CSV)", h.to_csv(index=False).encode(), "riwayat_cluster.csv", "text/csv")
        else:
            st.caption("Belum ada riwayat.")

# ============================ SIMULASI WHAT-IF ===============================
elif hal == "sim":
    judul("Simulasi What-If", "Coba ubah IPK atau pengeluaran, lihat apakah Anda pindah cluster")
    R = st.session_state["res"]
    b_tot, b_ipk = (R["total"], R["ipk"]) if R else (245000.0, 3.4)
    if not R:
        st.info("Belum ada data dari menu Cek Cluster. Simulasi memakai contoh: IPK 3,40 dan Rp245.000/minggu.")
    kiri, kanan = st.columns([1, 1.6])
    with kiri:
        card("<h4>Atur Skenario</h4>")
        s_ipk = st.slider("IPK simulasi", 0.0, 4.0, float(b_ipk), 0.01)
        s_pct = st.slider("Perubahan pengeluaran (%)", -70, 70, 0, 5)
        s_tot = b_tot * (1 + s_pct / 100)
        c0, _ = klasifikasi(b_tot, b_ipk)
        c1, sh = klasifikasi(s_tot, s_ipk)
        card(f"""<div class='row'><span>Sekarang</span><span>{pill(c0)}</span></div><div class='row' style='margin-top:6px'><span>Skenario</span><span>{pill(c1)}</span></div>
<p style='margin:10px 0 0'><b>{html.escape(NAMA[c1])}</b><br><span class='mut'>{rp(s_tot)}/minggu · IPK {s_ipk:.2f}</span></p>""")
        if c0 != c1:
            st.success(f"Skenario ini memindahkan Anda dari Cluster {c0} ke Cluster {c1}.")
        else:
            st.info("Skenario ini belum mengubah cluster Anda.")
    with kanan:
        card("<h4>Jalur pada Peta</h4><span class='mut'>▲ posisi sekarang · ＋ hijau = skenario</span>")
        peta(DF, (b_tot, b_ipk), 340, sim=((b_tot, b_ipk), (s_tot, s_ipk)))
    card("<h4>Cari Jalur ke Cluster Tujuan</h4>")
    tj = st.selectbox("Cluster tujuan", range(K), format_func=lambda i: f"Cluster {i} · {NAMA[i]}")
    o_ipk, o_p = butuh(tj, b_tot, b_ipk)
    if klasifikasi(b_tot, b_ipk)[0] == tj:
        st.success("Anda sudah berada di cluster ini.")
    else:
        x1, x2 = st.columns(2)
        x1.metric("Opsi A · Naikkan IPK ke", f"≥ {o_ipk:.2f}" if o_ipk else "Tidak cukup", f"+{o_ipk - b_ipk:.2f}" if o_ipk else None)
        x2.metric("Opsi B · Turunkan pengeluaran ke", f"≤ {rp(o_p)}" if o_p is not None else "Tidak cukup",
                  f"-{rp(b_tot - o_p)}" if o_p is not None else None, delta_color="inverse")
        st.caption("Masing-masing opsi dihitung dengan mengubah satu variabel saja, variabel lain tetap.")

# ============================ PROFIL & BANDINGKAN ============================
elif hal == "profil":
    judul("Profil & Bandingkan Cluster", "Pahami karakteristik tiap kelompok mahasiswa")
    t1, t2 = st.tabs(["Profil Tiap Cluster", "Bandingkan Dua Cluster"])
    with t1:
        for kol, c in zip(st.columns(K), range(K)):
            with kol:
                kp = M["komp"][c]
                seg = "".join(f'<i style="width:{v*100}%;background:hsl({215+i*18},70%,{40+i*6}%)" title="{KAT[i]}"></i>' for i, v in enumerate(kp))
                card(f"""{pill(c)}<h2 style='margin:8px 0 2px'>{html.escape(NAMA[c])}</h2>
<p class='mut'>{M['n'][c]} mahasiswa ({M['n'][c]/len(DF)*100:.0f}%)</p>
<div class='row'><span>Rata-rata IPK</span><b>{M['ipk'][c]}</b></div><div class='row'><span>Pengeluaran/minggu</span><b>{rp(M['peng'][c])}</b></div>
<div class='row'><span>Belajar · Organisasi · Kerja</span><b>{M['akt'][c][1]} · {M['akt'][c][2]} · {M['akt'][c][3]} jam</b></div>
<h4 style='margin-top:14px'>Komposisi pengeluaran</h4><div class='stack'>{seg}</div>
<p class='mut'>Terbesar: {KAT[int(np.argmax(kp))]} ({max(kp)*100:.0f}%)</p><p>{SARAN.get(c, '')}</p>""", f'style="border-top:5px solid {COL[c]}"')
    with t2:
        p, q = st.columns(2)
        ca = p.selectbox("Cluster A", range(K), 0, format_func=lambda i: NAMA[i])
        cb = q.selectbox("Cluster B", range(K), min(1, K - 1), format_func=lambda i: NAMA[i])
        baris = [("Jumlah mahasiswa", M["n"][ca], M["n"][cb]), ("Rata-rata IPK", M["ipk"][ca], M["ipk"][cb]),
                 ("Pengeluaran/minggu (Rp)", M["peng"][ca], M["peng"][cb])] + \
                [(f"Jam {AKT[j].lower()}", M["akt"][ca][j], M["akt"][cb][j]) for j in range(5)] + \
                [(f"% {KAT[j]}", round(M["komp"][ca][j] * 100, 1), round(M["komp"][cb][j] * 100, 1)) for j in range(8)]
        T = pd.DataFrame(baris, columns=["Metrik", NAMA[ca], NAMA[cb]])
        T["Selisih (B − A)"] = (T.iloc[:, 2] - T.iloc[:, 1]).round(2)
        st.dataframe(T, hide_index=True, use_container_width=True, height=470)

# ============================= EKSPLORASI DATA ===============================
elif hal == "eks":
    judul("Eksplorasi Data", "Filter data responden dan lihat sebarannya")
    f1, f2, f3 = st.columns(3)
    pilih = f1.multiselect("Cluster", list(range(K)), default=list(range(K)), format_func=lambda i: NAMA[i])
    r_ipk = f2.slider("Rentang IPK", 2.5, 4.0, (2.5, 4.0), 0.01)
    mx = int(DF.pengeluaran.max())
    r_p = f3.slider("Pengeluaran/minggu (ribu Rp)", 0, mx, (0, mx), 10)
    S = DF[DF.cluster.isin(pilih) & DF.ipk.between(*r_ipk) & DF.pengeluaran.between(*r_p)]
    card(f"<h4>{len(S)} dari {len(DF)} responden terpilih</h4>")
    if len(S):
        ta, tb, tc = st.tabs(["Peta", "Distribusi", "Tabel"])
        with ta:
            peta(S)
        with tb:
            sk = alt.Scale(domain=[NAMA[i] for i in range(K)], range=COL)
            h1 = alt.Chart(S).mark_bar(opacity=.85).encode(x=alt.X("ipk:Q", bin=alt.Bin(maxbins=18), title="IPK"), y="count()",
                                                           color=alt.Color("nama:N", scale=sk, legend=None)).properties(height=250)
            h2 = alt.Chart(S).mark_bar(opacity=.85).encode(x=alt.X("pengeluaran:Q", bin=alt.Bin(maxbins=18), title="Pengeluaran (ribu Rp)"),
                                                           y="count()", color=alt.Color("nama:N", scale=sk, legend=None)).properties(height=250)
            st.altair_chart((h1 | h2).configure(background="transparent"), use_container_width=True)
        with tc:
            st.dataframe(S.rename(columns={"pengeluaran": "pengeluaran (ribu Rp)"}), hide_index=True, use_container_width=True)
            st.download_button("⬇ Unduh data terfilter (CSV)", S.to_csv(index=False).encode(), "data_terfilter.csv", "text/csv")
    else:
        st.warning("Tidak ada data yang cocok dengan filter. Longgarkan rentang filter.")

# =============================== TENTANG METODE ==============================
else:
    judul("Tentang Metode", "Bagaimana sistem ini bekerja dan seberapa baik hasilnya")
    sil, db = metrik()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Algoritma", "K-Means")
    m2.metric("Jumlah cluster", K)
    m3.metric("Silhouette", f"{sil:.3f}", help="Makin tinggi makin baik (>0,25 sedang, >0,5 kuat)")
    m4.metric("Davies-Bouldin", f"{db:.3f}", help="Makin rendah makin baik")
    st.caption("Metrik dihitung ulang dari data tertanam sehingga bisa berbeda tipis dari angka di laporan Colab.")
    langkah = [("Business Understanding", "Memetakan kelompok mahasiswa berdasarkan pengeluaran dan prestasi akademik."),
               ("Data Understanding", "Data primer Google Form: pengeluaran 7 hari, IPK, dan jam aktivitas."),
               ("Preprocessing", "Anonimisasi, validasi, parsing rincian teks, penanganan outlier (IQR), standardisasi z-score."),
               ("Model", "K-Means pada dua fitur: total pengeluaran mingguan dan IPK."),
               ("Evaluasi", "Silhouette, Davies-Bouldin, Calinski-Harabasz, stabilitas bootstrap, uji Kruskal-Wallis."),
               ("Penerapan", "Website ini: data baru dikelompokkan ke centroid terdekat pada skala z-score.")]
    for i, (j, d) in enumerate(langkah, 1):
        card(f'<div class="step"><div class="no">{i}</div><div><b>{j}</b><br><span class="mut">{d}</span></div></div>')
    st.caption(f"Batas pengeluaran (outlier, Q3 + 1,5·IQR): {rp(M['cap'])}/minggu. Nilai di atasnya dibatasi sebelum diklasifikasikan.")
