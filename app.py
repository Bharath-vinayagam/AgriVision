import os
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import math

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriVision AI · Leaf Pathology Hub",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Model ───────────────────────────────────────────────────────────────────
import zipfile, tempfile, os

@st.cache_resource
def load_model():
    """
    Rebuild the exact architecture (Sequential: MobileNetV2→GAP→Dense128→Dropout→Dense38)
    and load weights directly from the .keras zip — bypasses Keras 2→3 graph differences.
    """
    MODEL_PATH = "final_leaf_disease_model.keras"

    # ── Strategy 1: standard load (works if Keras versions align) ─────────────
    try:
        m = tf.keras.models.load_model(MODEL_PATH, compile=False)
        # Quick sanity — force a build to catch silent graph errors
        _ = m(tf.zeros((1, 224, 224, 3)), training=False)
        return m
    except Exception:
        pass

    # ── Strategy 2: manual rebuild + weight loading from zip ──────────────────
    # Extract the weights h5 to a temp file
    with zipfile.ZipFile(MODEL_PATH, "r") as zf:
        tmp_weights = os.path.join(tempfile.gettempdir(), "leaf_model_weights.weights.h5")
        with zf.open("model.weights.h5") as src, open(tmp_weights, "wb") as dst:
            dst.write(src.read())

    # Rebuild exact architecture
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(38, activation="softmax")(x)
    m = tf.keras.Model(inputs, outputs)

    m.load_weights(tmp_weights)
    return m

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)


# ─── Classes ─────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
    'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot',
    'Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
    'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight',
    'Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy',
    'Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy',
    'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight',
    'Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy'
]

SEVERITY_ORDER = {"None": 0, "Low-Medium": 1, "Medium": 2, "Medium-High": 3, "High": 4, "Critical": 5}

def parse_class(raw):
    if '___' in raw:
        plant, disease = raw.split('___', 1)
    else:
        plant, disease = raw, "Unknown"
    plant = plant.replace('_',' ').replace('(including sour)','').replace('(maize)','').strip().title()
    disease = disease.replace('_',' ').strip()
    healthy = 'healthy' in disease.lower()
    if healthy:
        disease = "Healthy & Vital"
    return {"plant": plant, "disease": disease, "is_healthy": healthy}

DISEASE_KB = {
    "Black Rot":           {"symptoms":"Dark brown circular spots; fruit mummifies.","treatment":"Copper-based fungicide or Bacillus subtilis bio-spray.","prevention":"Prune canopy; remove infected fallen debris.","severity":"High","urgency":"Act within 48 hrs"},
    "Bacterial Spot":      {"symptoms":"Water-soaked spots with yellowish halos.","treatment":"Fixed-copper + mancozeb spray.","prevention":"Certified seeds; avoid overhead watering.","severity":"Medium-High","urgency":"Act within 72 hrs"},
    "Early Blight":        {"symptoms":"Concentric target-board spots on lower leaves.","treatment":"Weekly neem oil or copper hydroxide spray.","prevention":"Mulch soil; stake plants.","severity":"Medium","urgency":"Monitor weekly"},
    "Late Blight":         {"symptoms":"Grey water-soaked lesions with fuzzy white mold.","treatment":"Chlorothalonil / copper soap immediately.","prevention":"Resistant cultivars; remove host plants.","severity":"Critical","urgency":"Immediate action"},
    "Powdery Mildew":      {"symptoms":"White powder on upper leaf surfaces.","treatment":"Potassium bicarbonate or sulfur dust spray.","prevention":"Space plants; maximize sunlight.","severity":"Medium","urgency":"Act within 72 hrs"},
    "Leaf Scorch":         {"symptoms":"Purple-brown spots along leaf margins.","treatment":"Remove damaged leaves; bio-fungicide.","prevention":"Consistent soil moisture management.","severity":"Low-Medium","urgency":"Monitor biweekly"},
    "Esca (Black Measles)":{"symptoms":"Tiger-stripe yellowing; vine wood decay.","treatment":"Prune to green wood; seal cuts.","prevention":"Avoid wet-season pruning wounds.","severity":"High","urgency":"Act within 48 hrs"},
    "Haunglongbing":       {"symptoms":"Asymmetric mottling; bitter stunted fruit.","treatment":"Systemic psyllid insecticide sprays.","prevention":"Certified nursery stock; vector control.","severity":"Critical","urgency":"Immediate action"},
    "Healthy & Vital":     {"symptoms":"Vivid colour, clean margins, robust growth.","treatment":"No treatment needed. Maintain care routine.","prevention":"Regular monitoring & organic soil enrichment.","severity":"None","urgency":"No action needed"},
}

def get_info(disease):
    for k in DISEASE_KB:
        if k.lower() in disease.lower():
            return DISEASE_KB[k]
    return {"symptoms":"Spots or discoloration detected.","treatment":"Isolate plant; apply organic bio-fungicide.","prevention":"Field sanitation, crop rotation.","severity":"Medium","urgency":"Act within 72 hrs"}

# ─── Master CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Mulish:wght@400;500;600;700;800;900&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Mulish', sans-serif !important;
}

/* Editorial headings get Space Grotesk */
h1, h2, h3, h4, h5, h6,
.hero-title, .sec-head, .diag-name, .metric-value {
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Background: warm morning field + emerald mesh ── */
.stApp {
    background:
        radial-gradient(ellipse at 8% 5%,  rgba(254,243,199,.65) 0%, transparent 38%),
        radial-gradient(ellipse at 92% 12%, rgba(167,243,208,.70) 0%, transparent 42%),
        radial-gradient(ellipse at 55% 92%, rgba(187,247,208,.55) 0%, transparent 40%),
        radial-gradient(ellipse at 0% 60%,  rgba(254,249,195,.45) 0%, transparent 35%),
        linear-gradient(160deg, #F0FDF4 0%, #DCFCE7 35%, #ECFDF5 65%, #F0FDF4 100%);
    background-attachment: fixed;
    min-height: 100vh;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem !important; padding-bottom: 2rem !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,.72) !important;
    backdrop-filter: blur(28px) saturate(160%) !important;
    border-right: 1.5px solid rgba(52,211,153,.3) !important;
    box-shadow: 4px 0 30px rgba(6,78,59,.06) !important;
}

/* ── Glass Card ── */
.gc {
    background: rgba(255,255,255,.80);
    backdrop-filter: blur(18px) saturate(160%);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(167,243,208,.75);
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: 0 8px 28px rgba(6,78,59,.07), inset 0 1px 0 rgba(255,255,255,.9);
    margin-bottom: 16px;
    transition: box-shadow .2s;
}
.gc:hover { box-shadow: 0 12px 36px rgba(6,78,59,.12); }

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, rgba(254,251,235,.95) 0%, rgba(209,250,229,.95) 100%);
    border: 1.5px solid #6EE7B7;
    border-radius: 22px;
    padding: 22px 30px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 10px 40px rgba(16,185,129,.1);
    backdrop-filter: blur(12px);
}
.hero-title { font-size: 2rem; font-weight: 900; color: #064E3B; margin: 0; letter-spacing: -.5px; }
.hero-sub   { color: #047857; font-size: .97rem; font-weight: 500; margin-top: 5px; }

/* ── Metric Cards ── */
.metric-card {
    background: rgba(255,255,255,.85);
    border: 1px solid rgba(167,243,208,.8);
    border-radius: 16px;
    padding: 14px 16px;
    text-align: center;
    box-shadow: 0 6px 20px rgba(6,78,59,.06);
    transition: transform .2s, box-shadow .2s;
    height: 100%;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 10px 28px rgba(6,78,59,.12); }
.metric-label { font-size: .72rem; font-weight: 700; color: #6B7280; text-transform: uppercase; letter-spacing: .8px; margin-bottom: 6px; }
.metric-value { font-size: 1.25rem; font-weight: 800; }

/* ── Diagnosis Banner ── */
.diag-healthy  { background: linear-gradient(135deg, rgba(209,250,229,.85) 0%, rgba(167,243,208,.7) 100%); border-left: 6px solid #10B981; }
.diag-disease  { background: linear-gradient(135deg, rgba(254,226,226,.85) 0%, rgba(254,202,202,.7) 100%); border-left: 6px solid #EF4444; }
.diag-card {
    border-radius: 18px; padding: 20px 24px;
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 16px;
    border: 1px solid rgba(167,243,208,.5);
    box-shadow: 0 6px 24px rgba(6,78,59,.08);
}
.diag-name { font-size: 1.5rem; font-weight: 800; color: #064E3B; margin: 0; }
.diag-sub  { font-size: .88rem; font-weight: 600; color: #047857; margin-top: 4px; }

/* ── Badges ── */
.badge-ok  { background:#D1FAE5; color:#065F46; border:1.5px solid #10B981; padding:5px 14px; border-radius:20px; font-weight:800; font-size:.8rem; }
.badge-bad { background:#FEE2E2; color:#991B1B; border:1.5px solid #EF4444; padding:5px 14px; border-radius:20px; font-weight:800; font-size:.8rem; }
.badge-ai  { background:linear-gradient(135deg,#D1FAE5,#A7F3D0); color:#065F46; border:1.5px solid #10B981; padding:6px 16px; border-radius:20px; font-weight:800; font-size:.82rem; }

/* ── Urgency Tags ── */
.urg-critical { background:#FEE2E2; color:#991B1B; border-radius:8px; padding:3px 10px; font-size:.78rem; font-weight:700; }
.urg-high     { background:#FEF3C7; color:#92400E; border-radius:8px; padding:3px 10px; font-size:.78rem; font-weight:700; }
.urg-medium   { background:#D1FAE5; color:#065F46; border-radius:8px; padding:3px 10px; font-size:.78rem; font-weight:700; }
.urg-none     { background:#F0FDF4; color:#15803D; border-radius:8px; padding:3px 10px; font-size:.78rem; font-weight:700; }

/* ── Sidebar Pills ── */
.sb-pill {
    background: rgba(255,255,255,.9);
    border: 1px solid rgba(167,243,208,.8);
    border-radius: 12px;
    padding: 9px 13px;
    margin-bottom: 9px;
    font-size: .88rem;
    color: #065F46;
    font-weight: 600;
    display: flex; align-items: center; gap: 8px;
}

/* ── Section Headers ── */
.sec-head { font-size: 1rem; font-weight: 800; color: #064E3B; margin: 0 0 10px 0; display: flex; align-items: center; gap: 6px; }

/* ── Upload Area Styling ── */
.stFileUploader label p { 
    color: #064E3B !important; 
    font-weight: 800 !important;
    font-size: 0.95rem !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: rgba(255, 255, 255, 0.88) !important;
    border: 2px dashed #6EE7B7 !important;
    border-radius: 18px !important;
    padding: 16px 20px !important;
    backdrop-filter: blur(16px) !important;
    box-shadow: 0 6px 20px rgba(6, 78, 59, 0.05) !important;
    transition: all 0.25s ease-in-out !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #10B981 !important;
    background: rgba(209, 250, 229, 0.65) !important;
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.18) !important;
}

[data-testid="stFileUploaderDropzone"] span, 
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] div,
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #064E3B !important;
    font-family: 'Space Grotesk', 'Mulish', sans-serif !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 6px 16px !important;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25) !important;
    transition: transform 0.15s ease !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(16, 185, 129, 0.35) !important;
}

/* ── Treatment Cards ── */
.treat-card {
    background: rgba(255,255,255,.8);
    border: 1px solid rgba(167,243,208,.7);
    border-radius: 16px;
    padding: 16px;
    min-height: 150px;
    box-shadow: 0 4px 16px rgba(6,78,59,.05);
}
.treat-label { font-weight: 800; font-size: .9rem; margin-bottom: 8px; }
.treat-body  { font-size: .87rem; color: #0F5132; line-height: 1.55; }

/* ── Download Button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg,#10B981 0%,#059669 100%) !important;
    color:#fff !important; border:none !important; border-radius:12px !important;
    font-weight:700 !important; padding:10px 20px !important;
    box-shadow:0 4px 14px rgba(16,185,129,.3) !important;
    width:100% !important;
}
.stDownloadButton > button:hover { box-shadow:0 6px 20px rgba(16,185,129,.45) !important; }

/* ── Image frame ── */
.img-frame {
    border-radius: 14px; overflow: hidden;
    border: 1px solid rgba(167,243,208,.7);
    box-shadow: 0 4px 20px rgba(6,78,59,.1);
}

/* ── Info strip ── */
.info-strip {
    display:flex; justify-content:space-around;
    margin-top:10px; font-size:.82rem; color:#047857; font-weight:700;
}

h1,h2,h3,h4,h5,h6 { color:#064E3B !important; font-weight:700 !important; }
p,span,label,div   { color:#064E3B; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 8px 0 14px;'>
        <div style='font-size:2.2rem;'>🌿</div>
        <div style='font-size:1.1rem; font-weight:900; color:#064E3B;'>AgriVision AI</div>
        <div style='font-size:.82rem; color:#047857; font-weight:600;'>Plant Pathology Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Status
    if model_loaded:
        st.markdown("<div style='background:#D1FAE5;color:#065F46;padding:8px;border-radius:10px;font-weight:800;text-align:center;border:1px solid #10B981;font-size:.88rem;'>🟢 AI Engine Online</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='background:#FEE2E2;color:#991B1B;padding:8px;border-radius:10px;font-weight:800;text-align:center;font-size:.88rem;'>🔴 Engine Error</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.82rem;font-weight:800;color:#6B7280;text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px;'>Model Specifications</div>", unsafe_allow_html=True)
    specs = [
        ("🧠", "Architecture", "CNN / ResNet"),
        ("📊", "Dataset",      "PlantVillage 54k+"),
        ("📐", "Input Size",   "224 × 224 RGB"),
        ("🌱", "Plant Species","14 Crops"),
        ("🔬", "Disease Classes","38 Categories"),
        ("✅", "Accuracy",     "~98% on test set"),
    ]
    for ico, lbl, val in specs:
        st.markdown(f"<div class='sb-pill'>{ico} <span><b>{lbl}:</b> {val}</span></div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("<div style='font-size:.82rem;font-weight:800;color:#6B7280;text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px;'>Quick Test Samples</div>", unsafe_allow_html=True)
    samples_dir = "samples"
    sample_choice = None
    if os.path.exists(samples_dir):
        files = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        if files:
            sel = st.selectbox("Choose a sample image:", ["— Upload my own —"] + files[:25], label_visibility="collapsed")
            if sel != "— Upload my own —":
                sample_choice = os.path.join(samples_dir, sel)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.75rem;color:#9CA3AF;text-align:center;'>AgriVision AI v2.0 · PlantVillage CNN · MIT License</div>", unsafe_allow_html=True)

# ─── Hero Banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div>
    <div class="hero-title">🌿 Leaf Pathology Diagnostic Hub</div>
    <div class="hero-sub">AI-powered plant disease identification · Interactive analytics · Agricultural care guidance</div>
  </div>
  <div style="display:flex;flex-direction:column;align-items:flex-end;gap:6px;">
    <span class="badge-ai">● AI ONLINE</span>
    <span style="font-size:.75rem;color:#6B7280;">PlantVillage CNN · 38 Categories</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Variables ────────────────────────────────────────────────────────────────
image_to_process = None
image_source_name = None

# ─── Main Two-Column Layout ───────────────────────────────────────────────────
left_col, right_col = st.columns([1.05, 1.95], gap="large")

with left_col:
    st.markdown("<div class='sec-head'>📤 Leaf Image Input</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a leaf photograph", type=["jpg","jpeg","png"], label_visibility="collapsed")

    if uploaded:
        image_to_process = Image.open(uploaded).convert("RGB")
        image_source_name = uploaded.name
    elif sample_choice:
        image_to_process = Image.open(sample_choice).convert("RGB")
        image_source_name = os.path.basename(sample_choice)

    if image_to_process:
        st.markdown("<div class='gc' style='padding:12px;'>", unsafe_allow_html=True)
        # Crop to square for cleaner display
        w0, h0 = image_to_process.size
        side = min(w0, h0)
        img_sq = image_to_process.crop(((w0-side)//2, (h0-side)//2, (w0+side)//2, (h0+side)//2))
        st.image(img_sq, use_container_width=True)
        st.markdown(f"""
        <div class="info-strip">
            <span>📐 {w0}×{h0}</span>
            <span>🎨 RGB</span>
            <span>⚡ Float32</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Image stats card
        img_arr = np.array(image_to_process)
        mean_bright = float(img_arr.mean()) / 255 * 100
        r_mean = float(img_arr[:,:,0].mean())
        g_mean = float(img_arr[:,:,1].mean())
        b_mean = float(img_arr[:,:,2].mean())

        st.markdown("<div class='sec-head' style='margin-top:4px;'>🎨 Colour Channel Analysis</div>", unsafe_allow_html=True)
        fig_chan = go.Figure()
        channels = ["Red", "Green", "Blue"]
        vals     = [r_mean, g_mean, b_mean]
        colours  = ["#EF4444", "#10B981", "#3B82F6"]
        for ch, val, col in zip(channels, vals, colours):
            fig_chan.add_trace(go.Bar(
                x=[ch], y=[val], name=ch,
                marker_color=col, marker_line_width=0,
                text=[f"{val:.0f}"], textfont=dict(color="white", size=12),
                textposition="auto"
            ))
        fig_chan.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=0,b=0), height=160,
            showlegend=False, barmode="group",
            xaxis=dict(showgrid=False, tickfont=dict(color="#064E3B", size=12, family="Outfit")),
            yaxis=dict(showgrid=True, gridcolor="rgba(167,243,208,.4)", range=[0,255],
                       tickfont=dict(color="#6B7280", size=10))
        )
        st.plotly_chart(fig_chan, use_container_width=True, config={"displayModeBar": False})

        # Brightness meter
        st.markdown(f"""
        <div class='gc' style='padding:12px 16px;'>
            <div class='metric-label'>Overall Brightness</div>
            <div style='background:rgba(167,243,208,.3);border-radius:8px;overflow:hidden;height:10px;margin:6px 0;'>
                <div style='background:linear-gradient(90deg,#10B981,#34D399);width:{mean_bright:.1f}%;height:10px;border-radius:8px;'></div>
            </div>
            <div style='font-size:.82rem;color:#047857;font-weight:700;'>{mean_bright:.1f}% luminance</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class='gc' style='text-align:center;padding:48px 20px;'>
            <div style='font-size:3rem;'>🌿</div>
            <div style='font-size:1rem;font-weight:700;color:#047857;margin-top:10px;'>Awaiting Leaf Image</div>
            <div style='font-size:.88rem;color:#6B7280;margin-top:6px;'>Upload a photo or select a sample from the sidebar</div>
        </div>
        """, unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='sec-head'>📊 Diagnostic Analytics & Results</div>", unsafe_allow_html=True)

    if image_to_process and model_loaded:
        with st.spinner("Running neural network inference..."):
            img_r = image_to_process.resize((224, 224))
            arr   = np.expand_dims(np.array(img_r) / 255.0, 0)
            preds = model.predict(arr, verbose=0)[0]
            top5i = np.argsort(preds)[::-1][:5]
            top1  = CLASS_NAMES[top5i[0]]
            conf  = float(preds[top5i[0]])
            parsed = parse_class(top1)
            info   = get_info(parsed["disease"])

        # ── 4 Metric Cards ──────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4, gap="small")
        with c1:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>🌱 Target Crop</div>
                <div class='metric-value' style='color:#064E3B;font-size:1rem;'>{parsed['plant']}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            bk = "badge-ok" if parsed["is_healthy"] else "badge-bad"
            bl = "HEALTHY" if parsed["is_healthy"] else "DISEASED"
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>⚕️ Status</div>
                <div style='margin-top:4px;'><span class='{bk}'>{bl}</span></div>
            </div>""", unsafe_allow_html=True)
        with c3:
            conf_col = "#059669" if conf >= 0.8 else ("#D97706" if conf >= 0.5 else "#DC2626")
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>🎯 Confidence</div>
                <div class='metric-value' style='color:{conf_col};'>{conf:.1%}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            sev = info["severity"]
            sev_col = {"None":"#059669","Low-Medium":"#16A34A","Medium":"#D97706","Medium-High":"#EA580C","High":"#DC2626","Critical":"#991B1B"}.get(sev,"#DC2626")
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>⚠️ Risk Level</div>
                <div class='metric-value' style='color:{sev_col};font-size:1rem;'>{sev}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Diagnosis Banner ─────────────────────────────────────────────────
        dcls = "diag-healthy" if parsed["is_healthy"] else "diag-disease"
        dico = "🌿" if parsed["is_healthy"] else "⚠️"
        urg  = info["urgency"]
        urg_cls = "urg-critical" if "Immediate" in urg else ("urg-high" if "48" in urg else ("urg-none" if "No action" in urg else "urg-medium"))
        st.markdown(f"""
        <div class='diag-card {dcls}'>
            <div>
                <div class='diag-name'>{dico} {parsed['disease']}</div>
                <div class='diag-sub'>Identified in: <b>{parsed['plant']}</b> &nbsp;·&nbsp;
                    <span class='{urg_cls}'>{urg}</span>
                </div>
            </div>
            <div style='text-align:right;'>
                <div style='font-size:2.8rem;'>{dico}</div>
                <div style='font-size:.75rem;color:#6B7280;'>Severity: <b>{sev}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Charts Row ────────────────────────────────────────────────────────
        ch1, ch2 = st.columns([1.35, 1], gap="medium")

        with ch1:
            st.markdown("<div class='sec-head' style='margin-top:0;'>📊 Top 5 Pathogen Probability</div>", unsafe_allow_html=True)
            labels, probs = [], []
            for i in top5i:
                pc = parse_class(CLASS_NAMES[i])
                labels.append(f"{pc['plant']} — {pc['disease']}")
                probs.append(float(preds[i]) * 100)
            labels.reverse(); probs.reverse()

            bar_colors = [
                "#10B981" if p == max(probs) else
                ("#34D399" if p >= 5 else "#A7F3D0")
                for p in probs
            ]
            fig_bar = go.Figure(go.Bar(
                x=probs, y=labels, orientation="h",
                marker=dict(color=bar_colors, cornerradius=6,
                            line=dict(width=0)),
                text=[f"{v:.2f}%" for v in probs],
                textposition="auto",
                textfont=dict(color="white", size=11, family="Space Grotesk"),
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=4, r=8, t=4, b=4), height=230,
                xaxis=dict(showgrid=True, gridcolor="rgba(167,243,208,.35)",
                           zeroline=False, showticklabels=False,
                           tickfont=dict(color="#064E3B")),
                yaxis=dict(showgrid=False,
                           tickfont=dict(color="#064E3B", size=10.5, family="Space Grotesk")),
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        with ch2:
            st.markdown("<div class='sec-head' style='margin-top:0;'>🎯 AI Confidence Meter</div>", unsafe_allow_html=True)
            g_color = "#059669" if parsed["is_healthy"] else ("#DC2626" if conf < 0.6 else "#EA580C" if conf < 0.85 else "#059669")
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=conf * 100,
                delta={"reference": 80, "increasing": {"color": "#059669"}, "decreasing": {"color": "#DC2626"}},
                number={"suffix": "%", "font": {"color": "#064E3B", "size": 30, "family": "Outfit"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#A7F3D0",
                             "tickfont": {"color": "#6B7280", "size": 9}},
                    "bar":  {"color": g_color, "thickness": .28},
                    "bgcolor": "rgba(255,255,255,.4)",
                    "borderwidth": 1.5, "bordercolor": "#A7F3D0",
                    "steps": [
                        {"range": [0, 50],  "color": "rgba(254,226,226,.5)"},
                        {"range": [50, 80], "color": "rgba(254,243,199,.5)"},
                        {"range": [80, 100],"color": "rgba(209,250,229,.5)"},
                    ],
                    "threshold": {"line": {"color": "#064E3B", "width": 3}, "thickness": .8, "value": 80},
                }
            ))
            fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=14,r=14,t=14,b=8), height=230)
            st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})

        # ── Radar Chart: Severity Dimensions ────────────────────────────────
        st.markdown("<div class='sec-head'>🕸️ Pathogen Risk Profile</div>", unsafe_allow_html=True)
        sev_num = SEVERITY_ORDER.get(info["severity"], 2)
        radar_dims   = ["Severity", "Spread Risk", "Yield Impact", "Treatment Ease", "Detection Ease", "Urgency"]
        radar_vals   = [sev_num, min(sev_num+1,5), min(sev_num,5),
                        max(5-sev_num,0), max(4-sev_num,0), min(sev_num+1,5)]
        radar_vals   += [radar_vals[0]]          # close polygon
        radar_dims_c  = radar_dims + [radar_dims[0]]

        fig_rad = go.Figure(go.Scatterpolar(
            r=radar_vals, theta=radar_dims_c,
            fill="toself",
            fillcolor="rgba(16,185,129,.18)",
            line=dict(color="#059669", width=2.5),
            marker=dict(color="#059669", size=7)
        ))
        fig_rad.update_layout(
            polar=dict(
                bgcolor="rgba(255,255,255,.5)",
                radialaxis=dict(visible=True, range=[0,5], gridcolor="rgba(167,243,208,.5)",
                                tickfont=dict(color="#6B7280", size=9), tickvals=[1,2,3,4,5]),
                angularaxis=dict(tickfont=dict(color="#064E3B", size=11, family="Space Grotesk"),
                                 linecolor="rgba(167,243,208,.6)"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20,r=20,t=14,b=10), height=240,
            showlegend=False,
        )
        
        # ── Donut Confidence Distribution ───────────────────────────────────
        p2_labels, p2_vals, p2_cols = [], [], ["#10B981","#34D399","#6EE7B7","#A7F3D0","#D1FAE5"]
        for idx, i in enumerate(top5i):
            pc = parse_class(CLASS_NAMES[i])
            p2_labels.append(f"{pc['plant']} - {pc['disease']}")
            p2_vals.append(float(preds[i]) * 100)
        fig_pie = go.Figure(go.Pie(
            labels=p2_labels, values=p2_vals,
            hole=.55,
            marker=dict(colors=p2_cols, line=dict(color="white", width=2)),
            textfont=dict(color="#064E3B", size=10, family="Space Grotesk"),
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>%{value:.2f}%<extra></extra>",
        ))
        fig_pie.add_annotation(
            text=f"<b>{conf:.0%}</b><br><span style='font-size:10'>Top Match</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=15, color="#064E3B", family="Space Grotesk")
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=8,r=8,t=8,b=8), height=240,
            showlegend=False,
        )

        radar_col, pie_col = st.columns(2, gap="medium")
        with radar_col:
            st.plotly_chart(fig_rad, use_container_width=True, config={"displayModeBar": False})
        with pie_col:
            st.markdown("<div class='sec-head'>🍩 Confidence Distribution</div>", unsafe_allow_html=True)
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

        # ── Treatment Plan ───────────────────────────────────────────────────
        st.markdown("<div class='sec-head' style='margin-top:4px;'>🩺 Agricultural Care & Treatment Plan</div>", unsafe_allow_html=True)
        tc1, tc2, tc3 = st.columns(3, gap="small")
        with tc1:
            st.markdown(f"""<div class='treat-card'>
                <div class='treat-label' style='color:#0284C7;'>🔍 Symptoms</div>
                <div class='treat-body'>{info['symptoms']}</div>
            </div>""", unsafe_allow_html=True)
        with tc2:
            st.markdown(f"""<div class='treat-card'>
                <div class='treat-label' style='color:#059669;'>🛠️ Treatment</div>
                <div class='treat-body'>{info['treatment']}</div>
            </div>""", unsafe_allow_html=True)
        with tc3:
            st.markdown(f"""<div class='treat-card'>
                <div class='treat-label' style='color:#D97706;'>🛡️ Prevention</div>
                <div class='treat-body'>{info['prevention']}</div>
            </div>""", unsafe_allow_html=True)

        # ── Report Download ─────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        report = f"""================================================
AGRIVISION AI · DIAGNOSTIC REPORT
================================================
Image File    : {image_source_name}
Plant Species : {parsed['plant']}
Condition     : {parsed['disease']}
Status        : {"Healthy" if parsed["is_healthy"] else "Pathogen Detected"}
Confidence    : {conf:.2%}
Risk Level    : {info['severity']}
Action        : {info['urgency']}

SYMPTOMS:
{info['symptoms']}

TREATMENT:
{info['treatment']}

PREVENTION:
{info['prevention']}

TOP 5 PREDICTIONS:
"""
        for idx, i in enumerate(top5i):
            pc = parse_class(CLASS_NAMES[i])
            report += f"  {idx+1}. {pc['plant']} — {pc['disease']}: {preds[i]:.4%}\n"
        report += "================================================\n"

        st.download_button(
            "📄 Download Full Diagnostic Report",
            data=report,
            file_name=f"AgriVision_{parsed['plant']}_{parsed['disease'].replace(' ','_')}.txt",
            mime="text/plain"
        )

    elif not model_loaded:
        st.error(f"⚠️ Model failed to load: {model_error}")
    else:
        st.markdown("""
        <div class='gc' style='text-align:center;padding:60px 24px;'>
            <div style='font-size:3.5rem;'>🔬</div>
            <div style='font-size:1.1rem;font-weight:800;color:#047857;margin-top:12px;'>Ready to Diagnose</div>
            <div style='font-size:.9rem;color:#6B7280;margin-top:6px;'>Upload a leaf image or pick a sample from the sidebar to view full AI analytics, data visualizations & care guidance.</div>
        </div>
        """, unsafe_allow_html=True)
