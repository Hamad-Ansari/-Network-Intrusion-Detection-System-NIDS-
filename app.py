import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys, os, time

sys.path.insert(0, os.path.dirname(__file__))
from utils.preprocessing import (
    load_preprocessors, validate_columns, preprocess_dataframe,
    preprocess_manual_input, EXPECTED_FEATURES, CAT_COLS,
    PROTOCOL_TYPES, FLAG_TYPES, SERVICE_TYPES
)
from utils.helpers import (
    load_model, predict, is_attack, get_status_color,
    get_attack_category, MODEL_DISPLAY_NAMES, MODEL_ACCURACIES
)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NIDS — Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark background */
[data-testid="stAppViewContainer"] {
    background: #0a0e1a;
    color: #e0e6f0;
}
[data-testid="stSidebar"] {
    background: #0d1224;
    border-right: 1px solid #1a2540;
}
[data-testid="stHeader"] { background: transparent; }

/* Neon card style */
.neon-card {
    background: #0d1530;
    border: 1px solid #1a2e5a;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 0 18px rgba(0,200,120,0.04);
}
.neon-card-red {
    background: #1a0d0d;
    border: 1px solid #5a1a1a;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 0 18px rgba(255,50,50,0.08);
}
.neon-card-green {
    background: #0a1a12;
    border: 1px solid #1a5a30;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 0 20px rgba(0,255,120,0.08);
}

/* Stat boxes */
.stat-box {
    background: #0d1530;
    border: 1px solid #1a3060;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.stat-value { font-size: 2rem; font-weight: 700; color: #00ff88; }
.stat-label { font-size: 0.8rem; color: #6b82a8; text-transform: uppercase; letter-spacing: 1px; }
.stat-value-red { font-size: 2rem; font-weight: 700; color: #ff4444; }

/* Header */
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00ff88, #00bfff, #7b2fff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
}
.hero-sub { color: #6b82a8; font-size: 1rem; margin-top: -0.5rem; }

/* Threat badge */
.badge-safe {
    display: inline-block;
    background: #003322;
    border: 1px solid #00ff88;
    color: #00ff88;
    border-radius: 20px;
    padding: 0.3rem 1rem;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 1px;
}
.badge-threat {
    display: inline-block;
    background: #330000;
    border: 1px solid #ff4444;
    color: #ff4444;
    border-radius: 20px;
    padding: 0.3rem 1rem;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 1px;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%,100% { box-shadow: 0 0 6px rgba(255,68,68,0.4); }
    50% { box-shadow: 0 0 18px rgba(255,68,68,0.9); }
}

/* Sidebar label */
.sidebar-section {
    color: #00bfff;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 1rem 0 0.5rem;
}

/* Divider */
.neon-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1a3060, transparent);
    margin: 1.5rem 0;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1a3060; border-radius: 3px; }

/* DataFrames */
[data-testid="stDataFrame"] { border-radius: 8px; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #00ff88, #00bfff);
    color: #000;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.8rem;
    font-size: 1rem;
    letter-spacing: 0.5px;
    transition: all 0.2s;
}
.stButton > button:hover {
    box-shadow: 0 0 20px rgba(0,255,136,0.4);
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)


# ─── Header ──────────────────────────────────────────────────────────────────
col_logo, col_title, col_status = st.columns([1, 6, 2])
with col_logo:
    st.markdown("<div style='font-size:3.5rem;padding-top:0.3rem'>🛡️</div>", unsafe_allow_html=True)
with col_title:
    st.markdown('<div class="hero-title">Network Intrusion Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">AI-Powered Cybersecurity • KDD Cup 1999 • Real-time Threat Analysis</div>', unsafe_allow_html=True)
with col_status:
    st.markdown("""
    <div style='text-align:right; padding-top:0.8rem'>
        <span style='color:#00ff88;font-size:0.75rem;letter-spacing:1px'>⬤ SYSTEM ONLINE</span><br>
        <span style='color:#6b82a8;font-size:0.7rem'>All engines active</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔐 NIDS Control Panel")
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">⚙️ Model Selection</div>', unsafe_allow_html=True)
    model_key = st.selectbox(
        "Choose ML Model",
        options=list(MODEL_DISPLAY_NAMES.keys()),
        format_func=lambda x: MODEL_DISPLAY_NAMES[x],
        index=0,
    )
    st.markdown(
        f"<div style='color:#6b82a8;font-size:0.75rem'>Training accuracy: "
        f"<span style='color:#00ff88'>{MODEL_ACCURACIES[model_key]*100:.2f}%</span></div>",
        unsafe_allow_html=True
    )

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">📂 Input Mode</div>', unsafe_allow_html=True)
    input_mode = st.radio(
        "Select input method",
        ["Upload CSV File", "Manual Input"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">ℹ️ About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#4a5e82;font-size:0.78rem;line-height:1.7'>
    Dataset: KDD Cup 1999<br>
    Features: 41 network attributes<br>
    Classes: Normal + 11 attack types<br>
    Models: 7 ML algorithms
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
    with st.expander("📋 Expected CSV Columns"):
        st.code("\n".join(EXPECTED_FEATURES), language="text")


# ─── Load resources ──────────────────────────────────────────────────────────
@st.cache_resource
def get_preprocessors():
    return load_preprocessors('models/preprocessors.pkl')

@st.cache_resource
def get_model(name):
    return load_model(name)


preprocessors = get_preprocessors()
le_target = preprocessors['le_target']
classes = list(le_target.classes_)


# ─── Helper: Plotly theme ─────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(13,21,48,0.9)',
    font=dict(color='#a0b4cc', family='monospace'),
    margin=dict(l=20, r=20, t=40, b=20),
)

def prob_bar_chart(proba_row, classes):
    colors = ['#ff4444' if c != 'normal' else '#00ff88' for c in classes]
    fig = go.Figure(go.Bar(
        x=classes, y=proba_row * 100,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in proba_row * 100],
        textposition='outside',
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Class Probability Distribution (%)",
        yaxis=dict(title="Probability (%)", gridcolor='#1a2e5a', range=[0, 110]),
        xaxis=dict(gridcolor='#1a2e5a'),
        height=300,
    )
    return fig

def pie_chart(labels):
    counts = pd.Series(labels).value_counts()
    colors = ['#00ff88' if l == 'normal' else '#ff4444' for l in counts.index]
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values,
        marker=dict(colors=colors, line=dict(color='#0a0e1a', width=2)),
        hole=0.45,
        textinfo='label+percent',
        textfont=dict(size=12),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Normal vs Attack Distribution",
        height=350,
        showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#a0b4cc')),
    )
    return fig

def attack_breakdown_chart(labels):
    counts = pd.Series(labels).value_counts()
    colors = ['#00ff88' if l == 'normal' else px.colors.qualitative.Set1[i % 9]
              for i, l in enumerate(counts.index)]
    fig = go.Figure(go.Bar(
        x=counts.values, y=counts.index,
        orientation='h',
        marker_color=colors,
        text=counts.values,
        textposition='outside',
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Connection Type Breakdown",
        xaxis=dict(title="Count", gridcolor='#1a2e5a'),
        yaxis=dict(gridcolor='#1a2e5a'),
        height=max(250, len(counts) * 40),
    )
    return fig

def feature_importance_chart(model, feature_names):
    if not hasattr(model, 'feature_importances_'):
        return None
    importances = model.feature_importances_
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    fi_df = fi_df.sort_values('importance', ascending=True).tail(20)
    fig = go.Figure(go.Bar(
        x=fi_df['importance'], y=fi_df['feature'],
        orientation='h',
        marker=dict(
            color=fi_df['importance'],
            colorscale=[[0, '#1a3060'], [0.5, '#00bfff'], [1, '#00ff88']],
            showscale=False,
        ),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Top 20 Feature Importances",
        xaxis=dict(title="Importance", gridcolor='#1a2e5a'),
        yaxis=dict(gridcolor='#1a2e5a'),
        height=500,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# INPUT MODE: Upload CSV
# ══════════════════════════════════════════════════════════════════════════════
if input_mode == "Upload CSV File":
    st.markdown("## 📂 Batch Analysis — Upload Network Traffic Data")

    upload_col, info_col = st.columns([3, 2])
    with upload_col:
        st.markdown('<div class="neon-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop your CSV file here",
            type=['csv'],
            help="Upload a CSV file with the same columns as the KDD Cup dataset."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with info_col:
        st.markdown("""
        <div class="neon-card">
        <div style='color:#00bfff;font-weight:600;margin-bottom:0.5rem'>📋 File Requirements</div>
        <div style='color:#6b82a8;font-size:0.82rem;line-height:1.9'>
        ✓ CSV format (.csv)<br>
        ✓ 41 feature columns<br>
        ✓ Optional: <code>connection</code> label column<br>
        ✓ Values matching KDD feature ranges<br>
        ✓ Categorical: protocol_type, service, flag
        </div>
        </div>
        """, unsafe_allow_html=True)

    # Load sample data button
    use_sample = st.button("🔄 Use Sample Dataset (Workshop Data)")
    if use_sample:
        uploaded_file = None
        df_raw = pd.read_csv('data/sample.csv').head(500)
        st.session_state['sample_loaded'] = df_raw

    if uploaded_file is not None or 'sample_loaded' in st.session_state:
        if uploaded_file is not None:
            df_raw = pd.read_csv(uploaded_file)
            st.session_state.pop('sample_loaded', None)
        else:
            df_raw = st.session_state['sample_loaded']

        # Validate
        valid, missing_cols = validate_columns(df_raw)

        st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 👁 Dataset Preview")

        # Stats row
        has_label = 'connection' in df_raw.columns
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(f"""<div class='stat-box'>
                <div class='stat-value'>{len(df_raw):,}</div>
                <div class='stat-label'>Total Records</div></div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""<div class='stat-box'>
                <div class='stat-value'>{df_raw.shape[1]}</div>
                <div class='stat-label'>Features</div></div>""", unsafe_allow_html=True)
        with s3:
            missing_pct = df_raw.isnull().mean().mean() * 100
            st.markdown(f"""<div class='stat-box'>
                <div class='stat-value'>{missing_pct:.1f}%</div>
                <div class='stat-label'>Missing Values</div></div>""", unsafe_allow_html=True)
        with s4:
            status_color = 'stat-value' if valid else 'stat-value-red'
            status_text = '✓ Valid' if valid else '✗ Issues'
            st.markdown(f"""<div class='stat-box'>
                <div class='{status_color}'>{status_text}</div>
                <div class='stat-label'>Schema Check</div></div>""", unsafe_allow_html=True)

        st.dataframe(
            df_raw.head(10).style.set_properties(**{'background-color': '#0d1530', 'color': '#a0b4cc', 'border': '1px solid #1a3060'}),
            use_container_width=True
        )

        if not valid:
            st.error(f"⚠️ Missing columns: {', '.join(missing_cols)}")
        else:
            st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
            st.markdown("### 🚀 Run Intrusion Detection")

            run_btn = st.button(f"⚡ Analyze with {MODEL_DISPLAY_NAMES[model_key]}", use_container_width=True)

            if run_btn:
                with st.spinner("🔍 Analyzing network traffic patterns..."):
                    time.sleep(0.5)
                    model = get_model(model_key)
                    X_input = preprocess_dataframe(df_raw, preprocessors)
                    pred_labels, proba, class_names = predict(model, X_input, preprocessors)

                # Results
                df_results = df_raw.copy()
                df_results['🎯 Prediction'] = pred_labels
                df_results['⚠️ Threat?'] = ['🔴 ATTACK' if is_attack(l) else '🟢 SAFE' for l in pred_labels]
                df_results['🏷️ Category'] = [get_attack_category(l) for l in pred_labels]

                if proba is not None:
                    conf = np.max(proba, axis=1) * 100
                    df_results['📊 Confidence'] = [f"{c:.1f}%" for c in conf]

                # Summary KPIs
                n_attack = sum(is_attack(l) for l in pred_labels)
                n_normal = len(pred_labels) - n_attack
                threat_rate = n_attack / len(pred_labels) * 100

                st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
                st.markdown("### 🎯 Detection Results")

                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.markdown(f"""<div class='stat-box'>
                        <div class='stat-value'>{n_normal:,}</div>
                        <div class='stat-label'>✅ Normal</div></div>""", unsafe_allow_html=True)
                with k2:
                    st.markdown(f"""<div class='stat-box'>
                        <div class='stat-value-red'>{n_attack:,}</div>
                        <div class='stat-label'>🚨 Attacks Detected</div></div>""", unsafe_allow_html=True)
                with k3:
                    st.markdown(f"""<div class='stat-box'>
                        <div class='stat-value' style='color:#ff8800'>{threat_rate:.1f}%</div>
                        <div class='stat-label'>⚡ Threat Rate</div></div>""", unsafe_allow_html=True)
                with k4:
                    avg_conf = np.mean(np.max(proba, axis=1)) * 100 if proba is not None else 0
                    st.markdown(f"""<div class='stat-box'>
                        <div class='stat-value' style='color:#00bfff'>{avg_conf:.1f}%</div>
                        <div class='stat-label'>📊 Avg Confidence</div></div>""", unsafe_allow_html=True)

                # Alert banner
                if threat_rate > 50:
                    st.markdown("""<div class='neon-card-red'>
                        <span style='font-size:1.4rem'>🚨</span>
                        <span style='color:#ff4444;font-weight:700;font-size:1.1rem'> HIGH THREAT ENVIRONMENT DETECTED</span>
                        <span style='color:#aa4444;font-size:0.85rem;margin-left:1rem'>Majority of traffic flagged as malicious. Immediate action recommended.</span>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div class='neon-card-green'>
                        <span style='font-size:1.4rem'>✅</span>
                        <span style='color:#00ff88;font-weight:700;font-size:1.1rem'> NETWORK LARGELY SECURE</span>
                        <span style='color:#448844;font-size:0.85rem;margin-left:1rem'>Majority of traffic appears normal. Monitor flagged connections below.</span>
                    </div>""", unsafe_allow_html=True)

                # Charts
                st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
                st.markdown("### 📊 Analytics Dashboard")

                chart1, chart2 = st.columns(2)
                with chart1:
                    st.plotly_chart(pie_chart(pred_labels), use_container_width=True)
                with chart2:
                    st.plotly_chart(attack_breakdown_chart(pred_labels), use_container_width=True)

                # Feature importance
                fi_fig = feature_importance_chart(get_model(model_key), EXPECTED_FEATURES)
                if fi_fig:
                    st.plotly_chart(fi_fig, use_container_width=True)

                # Results table
                st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
                st.markdown("### 📋 Detailed Predictions")

                display_cols = ['🎯 Prediction', '⚠️ Threat?', '🏷️ Category', '📊 Confidence'] + \
                               ['duration', 'protocol_type', 'service', 'src_bytes', 'dst_bytes']
                display_cols = [c for c in display_cols if c in df_results.columns]
                st.dataframe(df_results[display_cols], use_container_width=True, height=400)

                # Download
                csv_out = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download Predictions CSV",
                    data=csv_out,
                    file_name="nids_predictions.csv",
                    mime="text/csv",
                )


# ══════════════════════════════════════════════════════════════════════════════
# INPUT MODE: Manual Input
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("## ✍️ Manual Traffic Analysis")
    st.markdown('<div class="neon-card">', unsafe_allow_html=True)
    st.markdown("Enter network connection details below. Pre-filled with typical normal traffic values.")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.form("manual_input_form"):
        st.markdown("### 🔌 Connection Basics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            protocol_type = st.selectbox("Protocol Type", PROTOCOL_TYPES, index=0)
        with c2:
            service = st.selectbox("Service", SERVICE_TYPES, index=SERVICE_TYPES.index('http'))
        with c3:
            flag = st.selectbox("Connection Flag", FLAG_TYPES, index=FLAG_TYPES.index('SF'))
        with c4:
            duration = st.number_input("Duration (s)", min_value=0, max_value=60000, value=0)

        st.markdown("### 📦 Traffic Volume")
        v1, v2, v3, v4 = st.columns(4)
        with v1:
            src_bytes = st.number_input("Src Bytes", min_value=0, max_value=6000000, value=181)
        with v2:
            dst_bytes = st.number_input("Dst Bytes", min_value=0, max_value=700000, value=5450)
        with v3:
            land = st.selectbox("Land", [0, 1], index=0)
        with v4:
            wrong_fragment = st.number_input("Wrong Fragment", min_value=0, max_value=10, value=0)

        st.markdown("### 🔒 Login & Access Features")
        la1, la2, la3, la4 = st.columns(4)
        with la1:
            logged_in = st.selectbox("Logged In", [0, 1], index=1)
        with la2:
            num_failed_logins = st.number_input("Failed Logins", min_value=0, max_value=10, value=0)
        with la3:
            num_compromised = st.number_input("Num Compromised", min_value=0, max_value=100, value=0)
        with la4:
            root_shell = st.selectbox("Root Shell", [0, 1], index=0)

        with st.expander("⚙️ Advanced Features (Rate-based)"):
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                count = st.slider("Count", 0, 511, 8)
                srv_count = st.slider("Srv Count", 0, 511, 8)
            with r2:
                serror_rate = st.slider("SError Rate", 0.0, 1.0, 0.0, 0.01)
                rerror_rate = st.slider("RError Rate", 0.0, 1.0, 0.0, 0.01)
            with r3:
                same_srv_rate = st.slider("Same Srv Rate", 0.0, 1.0, 1.0, 0.01)
                diff_srv_rate = st.slider("Diff Srv Rate", 0.0, 1.0, 0.0, 0.01)
            with r4:
                dst_host_count = st.slider("Dst Host Count", 0, 255, 9)
                dst_host_srv_count = st.slider("Dst Host Srv Count", 0, 255, 9)

        submit = st.form_submit_button(
            f"⚡ Analyze with {MODEL_DISPLAY_NAMES[model_key]}",
            use_container_width=True
        )

    if submit:
        input_dict = {
            'duration': duration,
            'protocol_type': protocol_type,
            'service': service,
            'flag': flag,
            'src_bytes': src_bytes,
            'dst_bytes': dst_bytes,
            'land': land,
            'wrong_fragment': wrong_fragment,
            'urgent': 0,
            'hot': 0,
            'num_failed_logins': num_failed_logins,
            'logged_in': logged_in,
            'num_compromised': num_compromised,
            'root_shell': root_shell,
            'su_attempted': 0,
            'num_root': 0,
            'num_file_creations': 0,
            'num_shells': 0,
            'num_access_files': 0,
            'num_outbound_cmds': 0,
            'is_host_login': 0,
            'is_guest_login': 0,
            'count': count,
            'srv_count': srv_count,
            'serror_rate': serror_rate,
            'srv_serror_rate': serror_rate,
            'rerror_rate': rerror_rate,
            'srv_rerror_rate': rerror_rate,
            'same_srv_rate': same_srv_rate,
            'diff_srv_rate': diff_srv_rate,
            'srv_diff_host_rate': 0.0,
            'dst_host_count': dst_host_count,
            'dst_host_srv_count': dst_host_srv_count,
            'dst_host_same_srv_rate': same_srv_rate,
            'dst_host_diff_srv_rate': diff_srv_rate,
            'dst_host_same_src_port_rate': 0.0,
            'dst_host_srv_diff_host_rate': 0.0,
            'dst_host_serror_rate': serror_rate,
            'dst_host_srv_serror_rate': serror_rate,
            'dst_host_rerror_rate': rerror_rate,
            'dst_host_srv_rerror_rate': rerror_rate,
        }

        with st.spinner("🔍 Running threat analysis..."):
            time.sleep(0.4)
            model = get_model(model_key)
            X_input = preprocess_manual_input(input_dict, preprocessors)
            pred_labels, proba, class_names = predict(model, X_input, preprocessors)

        label = pred_labels[0]
        attack = is_attack(label)
        category = get_attack_category(label)

        st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🎯 Threat Analysis Result")

        res1, res2 = st.columns([2, 3])
        with res1:
            if attack:
                st.markdown(f"""
                <div class='neon-card-red' style='text-align:center;padding:2rem'>
                    <div style='font-size:4rem'>🚨</div>
                    <div style='color:#ff4444;font-size:1.8rem;font-weight:800;margin:0.5rem 0'>{label.upper()}</div>
                    <div><span class='badge-threat'>⚠ THREAT DETECTED</span></div>
                    <div style='color:#6b2222;margin-top:1rem;font-size:0.9rem'>Category: {category}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='neon-card-green' style='text-align:center;padding:2rem'>
                    <div style='font-size:4rem'>✅</div>
                    <div style='color:#00ff88;font-size:1.8rem;font-weight:800;margin:0.5rem 0'>NORMAL</div>
                    <div><span class='badge-safe'>✓ SAFE</span></div>
                    <div style='color:#225533;margin-top:1rem;font-size:0.9rem'>No intrusion detected</div>
                </div>
                """, unsafe_allow_html=True)

        with res2:
            if proba is not None:
                st.plotly_chart(prob_bar_chart(proba[0], class_names), use_container_width=True)
            conf = np.max(proba[0]) * 100 if proba is not None else 100
            st.markdown(f"""
            <div class='neon-card'>
            <div style='color:#00bfff;font-weight:600;margin-bottom:0.8rem'>📋 Analysis Summary</div>
            <table style='width:100%;font-size:0.88rem;color:#a0b4cc;border-collapse:collapse'>
            <tr><td style='padding:0.3rem;color:#6b82a8'>Model Used</td><td style='padding:0.3rem'>{MODEL_DISPLAY_NAMES[model_key]}</td></tr>
            <tr><td style='padding:0.3rem;color:#6b82a8'>Prediction</td><td style='padding:0.3rem;color:{"#ff4444" if attack else "#00ff88"}'><b>{label}</b></td></tr>
            <tr><td style='padding:0.3rem;color:#6b82a8'>Category</td><td style='padding:0.3rem'>{category}</td></tr>
            <tr><td style='padding:0.3rem;color:#6b82a8'>Confidence</td><td style='padding:0.3rem;color:#00bfff'>{conf:.1f}%</td></tr>
            <tr><td style='padding:0.3rem;color:#6b82a8'>Protocol</td><td style='padding:0.3rem'>{protocol_type}</td></tr>
            <tr><td style='padding:0.3rem;color:#6b82a8'>Service</td><td style='padding:0.3rem'>{service}</td></tr>
            <tr><td style='padding:0.3rem;color:#6b82a8'>Flag</td><td style='padding:0.3rem'>{flag}</td></tr>
            </table>
            </div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Model Performance Comparison (always shown at bottom)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
st.markdown("### 🏆 Model Performance Comparison")

perf_df = pd.DataFrame([
    {'Model': MODEL_DISPLAY_NAMES[k], 'Accuracy (%)': round(v * 100, 2)}
    for k, v in sorted(MODEL_ACCURACIES.items(), key=lambda x: -x[1])
])

fig_perf = go.Figure(go.Bar(
    x=perf_df['Accuracy (%)'],
    y=perf_df['Model'],
    orientation='h',
    marker=dict(
        color=perf_df['Accuracy (%)'],
        colorscale=[[0, '#ff4444'], [0.5, '#ff8800'], [1, '#00ff88']],
        showscale=False,
    ),
    text=[f"{v:.2f}%" for v in perf_df['Accuracy (%)']],
    textposition='outside',
))
fig_perf.update_layout(
    **PLOTLY_LAYOUT,
    title="Training Accuracy Comparison",
    xaxis=dict(title="Accuracy (%)", gridcolor='#1a2e5a', range=[95, 101]),
    yaxis=dict(gridcolor='#1a2e5a'),
    height=320,
)
st.plotly_chart(fig_perf, use_container_width=True)

# Footer
st.markdown("""
<div style='text-align:center;color:#2a3a5a;font-size:0.75rem;margin-top:2rem;padding:1rem'>
    🛡️ NIDS — Network Intrusion Detection System &nbsp;|&nbsp; 
    KDD Cup 1999 Dataset &nbsp;|&nbsp; 
    Built with Streamlit &nbsp;|&nbsp; 
    ML-Powered Cybersecurity
</div>
""", unsafe_allow_html=True)
