"""
Streamlit Dashboard — Bank Risk Analytics Platform
===================================================
Corre con: streamlit run dashboards/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import re
from html import escape
from dotenv import load_dotenv
load_dotenv()  # carga variables desde .env

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ai.reporter import generate_report, generate_portfolio_report  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Risk Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS global ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Fondo general */
[data-testid="stAppViewContainer"] {
    background-color: #0f172a;
}
[data-testid="stSidebar"] {
    background-color: #1e293b;
    border-right: 1px solid #334155;
}

/* Cards métricas */
[data-testid="stMetric"] {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 1.6rem !important;
    font-weight: 700;
}
[data-testid="stMetricDelta"] {
    font-size: 0.82rem !important;
}

/* Tabs */
[data-testid="stTabs"] button {
    color: #94a3b8;
    font-size: 0.88rem;
    font-weight: 500;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #f1f5f9 !important;
    border-bottom: 2px solid #3b82f6 !important;
}

/* Título principal */
h1 { color: #f1f5f9 !important; font-weight: 700; }
h2, h3 { color: #e2e8f0 !important; font-weight: 600; }

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #334155;
    border-radius: 8px;
}

/* Botón primario */
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s !important;
}
[data-testid="baseButton-primary"]:hover {
    opacity: 0.85 !important;
}

/* Search bar: input + botón alineados */
[data-testid="stForm"] [data-testid="stHorizontalBlock"] {
    gap: 8px !important;
    align-items: flex-end;
}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button {
    height: 42px !important;
    font-weight: 600;
    width: 100% !important;
}

/* Spinner personalizado */
.custom-spinner-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 16px;
    padding: 40px;
}
.custom-spinner {
    width: 48px;
    height: 48px;
    border: 4px solid #334155;
    border-top: 4px solid #3b82f6;
    border-radius: 50%;
    animation: spin 0.9s cubic-bezier(0.4, 0, 0.2, 1) infinite;
}
@keyframes spin {
    0%   { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
.spinner-text {
    color: #94a3b8;
    font-size: 0.9rem;
    font-weight: 500;
    letter-spacing: 0.03em;
}

/* Badge de riesgo */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.badge-high   { background: #7f1d1d; color: #fca5a5; }
.badge-medium { background: #78350f; color: #fcd34d; }
.badge-low    { background: #064e3b; color: #6ee7b7; }

/* Separador */
hr { border-color: #334155 !important; }

/* Caption */
[data-testid="stCaptionContainer"] { color: #64748b !important; }

/* Selectbox */
[data-testid="stSelectbox"] label { color: #94a3b8 !important; font-size: 0.82rem; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #1e293b; }
::-webkit-scrollbar-thumb { background: #475569; border-radius: 3px; }

/* Ocultar título "undefined" de Plotly (SVG) */
text.gtitle, .g-gtitle { visibility: hidden !important; height: 0 !important; }
svg text.gtitle tspan { fill: transparent !important; }

/* Ocultar el RUNNING... de Streamlit */
[data-testid="stStatusWidget"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Quitar fondo blanco del header */
[data-testid="stToolbar"] { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Paleta corporativa ────────────────────────────────────────────────────────
RISK_COLORS  = {"high": "#dc2626", "medium": "#d97706", "low": "#059669"}
FRAUD_COLORS = {"normal": "#059669", "suspicious": "#d97706", "high_risk": "#ea580c", "critical": "#dc2626"}
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#94a3b8",
    font_size=12,
    title=dict(text="", font=dict(color="#e2e8f0")),
    xaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
    yaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#334155"),
    margin=dict(l=10, r=10, t=30, b=10),
)

OUTPUTS    = ROOT / "outputs"
CONFIG_PATH = ROOT / "config" / "config.yaml"


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)

@st.cache_data
def load_clients():
    p = OUTPUTS / "clients_scored.csv"
    if not p.exists():
        return pd.DataFrame()
    # engine='python' evita ArrowStringArray que rompe plotly groupby
    df = pd.read_csv(p, engine="python")
    for col in df.select_dtypes(exclude=["number"]).columns:
        df[col] = df[col].astype(str)
    return df

@st.cache_data
def load_transactions():
    p = OUTPUTS / "transactions_scored.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, engine="python")
    for col in df.select_dtypes(exclude=["number"]).columns:
        df[col] = df[col].astype(str)
    return df

@st.cache_data
def load_summary():
    p = OUTPUTS / "summary.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_importance():
    p = OUTPUTS / "feature_importance.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def apply_theme(fig):
    fig.update_layout(**PLOTLY_THEME)
    fig.update_layout(title_text="")
    return fig


# ── Spinner personalizado ─────────────────────────────────────────────────────
def spinner_html(text: str = "Procesando..."):
    from html import escape
    safe_text = escape(text)
    return st.markdown(f"""
    <div class="custom-spinner-wrapper">
        <div class="custom-spinner"></div>
        <div class="spinner-text">{safe_text}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Pipeline runner ───────────────────────────────────────────────────────────
ALLOWED_SCENARIOS = {"normal", "crisis", "rate_hike"}

def run_pipeline_ui(scenario: str, label: str = ""):
    import subprocess, uuid, logging
    from html import escape

    # CN-008: allowlist explícita para evitar command injection
    if scenario not in ALLOWED_SCENARIOS:
        st.error("Escenario no válido.")
        return

    placeholder = st.empty()
    with placeholder.container():
        spinner_html(f"Simulando: {escape(label or scenario)}")

    result = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--scenario", scenario],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    placeholder.empty()

    if result.returncode == 0:
        st.success("Pipeline completado.")
        st.cache_data.clear()
        st.rerun()
    else:
        # CN-006: no exponer stderr al usuario — loguear internamente
        ref_id = str(uuid.uuid4())[:8].upper()
        logging.error(f"[{ref_id}] Pipeline failed:\n{result.stderr}")
        st.error(f"Error en la simulación. Referencia: {ref_id}")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='display:flex; align-items:center; gap:10px; margin-bottom:4px'>
        <span style='font-size:1.6rem'>🏦</span>
        <div>
            <div style='font-size:1rem; font-weight:700; color:#f1f5f9'>Risk Analytics</div>
            <div style='font-size:0.72rem; color:#64748b'>Bank Platform v1.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    config = load_config()
    scenario_options = {v["label"]: k for k, v in config["scenarios"].items()}
    selected_label   = st.selectbox("Escenario económico", list(scenario_options.keys()))
    selected_scenario = scenario_options[selected_label]

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if st.button("▶  Simular Escenario", type="primary", use_container_width=True):
        run_pipeline_ui(selected_scenario, selected_label)

    st.markdown("---")

    # Backend IA activo
    try:
        from src.ai.reporter import _active_backend, GROQ_MODEL
        backend = _active_backend()
        backend_info = {
            "groq":     ("🟢", f"Groq — {GROQ_MODEL}"),
            "ollama":   ("🔵", "Ollama — local"),
            "template": ("🟡", "Template — sin IA"),
        }
        dot, label = backend_info[backend]
        st.markdown(f"""
        <div style='background:#0f172a; border:1px solid #334155; border-radius:8px; padding:10px 14px;'>
            <div style='font-size:0.72rem; color:#64748b; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.05em'>Backend IA</div>
            <div style='font-size:0.85rem; color:#e2e8f0; font-weight:500'>{escape(dot)} {escape(label)}</div>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        import logging
        logging.warning(f"[Backend IA] No se pudo detectar: {e}")

    # Crédito del desarrollador
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 4px 0;'>
        <div style='font-size:0.68rem; color:#334155; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px'>Desarrollado por</div>
        <div style='font-size:0.82rem; font-weight:600; color:#475569'>Mathias Agulló</div>
    </div>
    """, unsafe_allow_html=True)


# ── Cargar datos ──────────────────────────────────────────────────────────────
clients      = load_clients()
transactions = load_transactions()
summary      = load_summary()
importance   = load_importance()

if clients.empty:
    st.markdown("""
    <div style='display:flex; flex-direction:column; align-items:center; justify-content:center;
                height:60vh; gap:16px; color:#64748b;'>
        <div style='font-size:3rem'>🏦</div>
        <div style='font-size:1.1rem; font-weight:600; color:#94a3b8'>Sin datos</div>
        <div style='font-size:0.85rem'>Ejecuta el pipeline desde el panel lateral</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
scenario_label = summary.get("scenario_label", selected_label)
elapsed        = summary.get("elapsed_seconds", 0)

st.markdown(f"""
<div style='margin-bottom:24px'>
    <h1 style='margin:0; font-size:1.6rem; font-weight:700; color:#f1f5f9'>
        Bank Risk Analytics Dashboard
    </h1>
    <div style='color:#64748b; font-size:0.82rem; margin-top:4px'>
        Escenario activo: <span style='color:#94a3b8; font-weight:500'>{escape(str(scenario_label))}</span>
        &nbsp;·&nbsp; {len(clients):,} clientes
        &nbsp;·&nbsp; Pipeline: {escape(str(elapsed))}s
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
risk_dist   = clients["risk_level"].value_counts()
high_risk   = int(risk_dist.get("high", 0))
medium_risk = int(risk_dist.get("medium", 0))
avg_pd      = clients["pd"].mean()
avg_score   = clients["credit_score"].mean()
fraud_rate  = summary.get("fraud", {}).get("fraud_rate", 0)
market_risk = summary.get("macro", {}).get("market_risk", "N/A")
model_meta  = summary.get("model_metrics", {})
auc         = model_meta.get("auc_roc_ensemble", 0)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Alto Riesgo",       f"{high_risk:,}",      f"{high_risk/len(clients):.1%} cartera")
col2.metric("PD Promedio",       f"{avg_pd:.1%}")
col3.metric("Credit Score",      f"{avg_score:.0f}",    "/ 850")
col4.metric("Tasa de Fraude",    f"{fraud_rate:.1%}")
col5.metric("Riesgo de Mercado", market_risk.upper())
col6.metric("AUC-ROC Modelo",    f"{auc:.3f}" if auc else "—")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Portafolio",
    "🏦  Riesgo Crediticio",
    "🔍  Fraude",
    "📈  Macro",
    "🤖  AI Reports",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PORTAFOLIO
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_a, col_b = st.columns(2)

    RISK_LABELS = {"high": "Alto", "medium": "Medio", "low": "Bajo"}
    RISK_COLORS_ES = {"Alto": "#dc2626", "Medio": "#d97706", "Bajo": "#059669"}

    with col_a:
        st.subheader("Distribución de Riesgo")
        risk_counts = clients["risk_level"].map(lambda x: RISK_LABELS.get(str(x).lower(), x)).value_counts().reset_index()
        risk_counts.columns = ["Nivel", "Clientes"]
        fig = px.pie(
            risk_counts,
            names="Nivel",
            values="Clientes",
            color="Nivel",
            color_discrete_map=RISK_COLORS_ES,
            hole=0.55,
        )
        fig.update_traces(textposition="outside", textinfo="percent+label",
                          marker=dict(line=dict(color="#0f172a", width=2)))
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Distribución de PD por Nivel")
        # Convertir a string Python nativo para evitar KeyError con ArrowStringArray
        clients_plot = clients.copy()
        clients_plot["risk_level"] = clients_plot["risk_level"].astype(str).str.lower()
        fig = go.Figure()
        for level, label, color in [("low", "Bajo", RISK_COLORS["low"]), ("medium", "Medio", RISK_COLORS["medium"]), ("high", "Alto", RISK_COLORS["high"])]:
            subset = clients_plot[clients_plot["risk_level"] == level]["pd"]
            fig.add_trace(go.Histogram(
                x=subset, name=label, nbinsx=40,
                marker_color=color, opacity=0.75,
                hovertemplate=f"<b>{level}</b><br>PD: %{{x:.2f}}<br>Clientes: %{{y}}<extra></extra>",
            ))
        fig.update_layout(barmode="overlay", xaxis_title="Probabilidad de Default", yaxis_title="Clientes")
        fig.add_vline(x=0.30, line_dash="dash", line_color="#d97706",
                      annotation_text="REVISAR", annotation_font_color="#d97706",
                      annotation_position="top right",
                      annotation_font_size=11)
        fig.add_vline(x=0.60, line_dash="dash", line_color="#dc2626",
                      annotation_text="RECHAZAR", annotation_font_color="#dc2626",
                      annotation_position="top right",
                      annotation_font_size=11)
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Riesgo Promedio por Región")
    region_risk = clients.groupby("region").agg(
        avg_pd=("pd", "mean"),
        clientes=("client_id", "count"),
        alto_riesgo=("risk_level", lambda x: (x == "high").sum()),
    ).reset_index()
    region_risk["pct_alto"] = region_risk["alto_riesgo"] / region_risk["clientes"]
    fig = px.bar(
        region_risk.sort_values("avg_pd", ascending=False),
        x="region", y="avg_pd",
        color="pct_alto",
        color_continuous_scale=[[0, "#059669"], [0.5, "#d97706"], [1, "#dc2626"]],
        labels={"avg_pd": "PD Promedio", "region": "Región", "pct_alto": "% Alto Riesgo"},
        text=region_risk.sort_values("avg_pd", ascending=False)["clientes"],
    )
    fig.update_traces(textposition="outside", textfont_color="#94a3b8")
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RIESGO CREDITICIO
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Credit Score vs Probabilidad de Default")
    sample = clients.sample(min(500, len(clients)), random_state=1).copy()
    sample["risk_level"] = sample["risk_level"].astype(object).astype(str)
    fig = go.Figure()
    for level, label, color in [("low", "Bajo", RISK_COLORS["low"]), ("medium", "Medio", RISK_COLORS["medium"]), ("high", "Alto", RISK_COLORS["high"])]:
        s = sample[sample["risk_level"] == level]
        fig.add_trace(go.Scatter(
            x=s["credit_score"], y=s["pd"],
            mode="markers", name=label,
            marker=dict(color=color, opacity=0.65, size=6),
            customdata=s[["client_id", "income", "debt_ratio", "employment_status", "decision"]].values,
            hovertemplate="<b>%{customdata[4]}</b><br>Score: %{x}<br>PD: %{y:.1%}<br>ID: %{customdata[0]}<extra></extra>",
        ))
    fig.update_layout(xaxis_title="Credit Score", yaxis_title="Probabilidad de Default", **PLOTLY_THEME)
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Decisiones del Banco")
        decision_colors = {"APROBAR": "#059669", "REVISAR": "#d97706", "RECHAZAR": "#dc2626"}
        decision_counts = clients["decision"].astype(str).value_counts().reset_index()
        decision_counts.columns = ["Decisión", "Clientes"]
        fig = go.Figure()
        for dec, color in decision_colors.items():
            row = decision_counts[decision_counts["Decisión"] == dec]
            if not row.empty:
                fig.add_trace(go.Bar(
                    x=[dec], y=[int(row["Clientes"].iloc[0])],
                    name=dec, marker_color=color,
                    marker_line_color="#0f172a", marker_line_width=1,
                    text=[int(row["Clientes"].iloc[0])], textposition="outside",
                ))
        fig.update_layout(showlegend=False, yaxis_title="Clientes", **PLOTLY_THEME)
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        if not importance.empty:
            st.subheader("Feature Importance — Top 10")
            FEATURE_LABELS = {
                "loan_to_income":        "Deuda / Ingreso",
                "income_log":            "Ingreso (log)",
                "credit_maturity":       "Antigüedad Crediticia",
                "savings_ratio":         "Ratio de Ahorro",
                "financial_burden":      "Carga Financiera",
                "rate_x_loan_income":    "Tasa × Deuda/Ingreso",
                "mkt_x_burden":          "Mercado × Carga Fin.",
                "age":                   "Edad",
                "payment_history_score": "Historial de Pagos",
                "missed_payments":       "Pagos Atrasados",
                "debt_ratio":            "Ratio de Deuda",
                "credit_score":          "Credit Score",
                "unemp_x_employment":    "Desempleo × Empleo",
                "unemp_x_debt":          "Desempleo × Deuda",
                "rate_x_debt":           "Tasa × Deuda",
                "stress_x_missed":       "Stress × Atrasos",
                "usdclp_x_income":       "USD/CLP × Ingreso",
                "copper_x_region":       "Cobre × Región",
                "employment_code":       "Estado Laboral",
                "region_code":           "Región",
                "rule_flag":             "Alerta Normativa",
            }
            imp_plot = importance.head(10).copy()
            imp_plot["feature"] = imp_plot["feature"].map(lambda x: FEATURE_LABELS.get(x, x))
            fig = px.bar(
                imp_plot,
                x="importance", y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale=[[0, "#1e3a5f"], [1, "#3b82f6"]],
                labels={"importance": "Importancia", "feature": ""},
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Buscar Cliente")
    _search_col, _ = st.columns([1, 2])
    with _search_col:
        with st.form(key="form_search_tab2", border=False):
            _col_input, _col_btn = st.columns([3, 1])
            with _col_input:
                client_id_input = st.number_input(
                    "Client ID", min_value=1,
                    max_value=int(clients["client_id"].max()), value=1,
                    label_visibility="collapsed",
                    key="search_client_tab2",
                )
            with _col_btn:
                st.form_submit_button("Buscar →", use_container_width=True)
    client_row = clients[clients["client_id"] == client_id_input]
    if not client_row.empty:
        row = client_row.iloc[0]
        rl = str(row["risk_level"]).lower()
        # CN-008: validar rl contra valores conocidos antes de usarlo en clase CSS
        rl_safe = rl if rl in {"high", "medium", "low"} else "low"
        rl_label = {"high": "ALTO", "medium": "MEDIO", "low": "BAJO"}.get(rl, rl.upper())
        decision_colors_text = {"APROBAR": "#6ee7b7", "REVISAR": "#fcd34d", "RECHAZAR": "#fca5a5"}
        decision_bg          = {"APROBAR": "#064e3b", "REVISAR": "#78350f", "RECHAZAR": "#7f1d1d"}
        dec = str(row["decision"])
        # CN-008: validar dec contra valores conocidos
        dec_safe = dec if dec in decision_bg else "REVISAR"

        st.markdown(f"""
        <div style='background:#1e293b; border:1px solid #334155; border-radius:12px;
                    padding:20px 24px; display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px;'>
            <div>
                <div style='font-size:0.72rem; color:#64748b; text-transform:uppercase; letter-spacing:0.05em'>Nivel de Riesgo</div>
                <div class='badge badge-{rl_safe}' style='margin-top:6px'>{escape(rl_label)}</div>
            </div>
            <div>
                <div style='font-size:0.72rem; color:#64748b; text-transform:uppercase; letter-spacing:0.05em'>Probabilidad Default</div>
                <div style='font-size:1.3rem; font-weight:700; color:#f1f5f9; margin-top:4px'>{row["pd"]:.1%}</div>
            </div>
            <div>
                <div style='font-size:0.72rem; color:#64748b; text-transform:uppercase; letter-spacing:0.05em'>Credit Score</div>
                <div style='font-size:1.3rem; font-weight:700; color:#f1f5f9; margin-top:4px'>{row["credit_score"]:.0f}<span style='font-size:0.9rem; color:#64748b'>/850</span></div>
            </div>
            <div>
                <div style='font-size:0.72rem; color:#64748b; text-transform:uppercase; letter-spacing:0.05em'>Decisión</div>
                <div style='margin-top:6px; display:inline-block; background:{decision_bg.get(dec_safe,"#1e293b")};
                            color:{decision_colors_text.get(dec_safe,"#f1f5f9")}; padding:3px 12px;
                            border-radius:20px; font-size:0.82rem; font-weight:600'>{escape(dec_safe)}</div>
            </div>
            <div>
                <div style='font-size:0.72rem; color:#64748b; text-transform:uppercase; letter-spacing:0.05em'>Deuda / Ingreso</div>
                <div style='font-size:1.1rem; font-weight:600; color:#e2e8f0; margin-top:4px'>{row["debt_ratio"]:.1%}</div>
            </div>
            <div>
                <div style='font-size:0.72rem; color:#64748b; text-transform:uppercase; letter-spacing:0.05em'>Pagos Atrasados</div>
                <div style='font-size:1.1rem; font-weight:600; color:#e2e8f0; margin-top:4px'>{int(row["missed_payments"])}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Top 10 — Mayor Probabilidad de Default")
    if not clients.empty:
        top10 = clients.nlargest(10, "pd")[
            ["client_id", "pd", "risk_level", "decision",
             "credit_score", "income", "debt_ratio", "missed_payments"]
        ].copy()
        top10["risk_level"] = top10["risk_level"].map(lambda x: {"high":"Alto","medium":"Medio","low":"Bajo"}.get(str(x).lower(), x))
        top10["pd"]         = top10["pd"].map("{:.1%}".format)
        top10["income"]     = top10["income"].map("${:,.0f}".format)
        top10["debt_ratio"] = top10["debt_ratio"].map("{:.0%}".format)
        top10.columns = ["ID", "Prob. Default", "Nivel Riesgo", "Decisión",
                         "Credit Score", "Ingreso", "Ratio Deuda", "Pagos Atrasados"]
        RISK_COLOR_MAP = {"high": "#fca5a5", "medium": "#fcd34d", "low": "#6ee7b7"}
        headers = "".join(f"<th style='padding:8px 14px;text-align:left;color:#94a3b8;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.05em;border-bottom:1px solid #334155'>{c}</th>" for c in top10.columns)
        rows = ""
        for _, row in top10.iterrows():
            cells = ""
            for col, val in row.items():
                style = ""
                if col == "Nivel Riesgo":
                    style = f"color:{RISK_COLOR_MAP.get(str(val).lower(), '#e2e8f0')};font-weight:600"
                elif col == "Decisión":
                    dc = {"APROBAR": "#6ee7b7", "REVISAR": "#fcd34d", "RECHAZAR": "#fca5a5"}
                    style = f"color:{dc.get(str(val), '#e2e8f0')};font-weight:600"
                cells += f"<td style='padding:8px 14px;color:#e2e8f0;font-size:0.85rem;border-bottom:1px solid #1e293b;{style}'>{escape(str(val))}</td>"
            rows += f"<tr>{cells}</tr>"
        st.markdown(f"""
        <table style='width:100%;border-collapse:collapse;background:#1e293b;border-radius:8px;overflow:hidden'>
            <thead><tr>{headers}</tr></thead>
            <tbody>{rows}</tbody>
        </table>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FRAUDE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if transactions.empty:
        st.info("No hay datos de transacciones.")
    else:
        flagged = transactions["fraud_flag"].sum()
        avg_fraud_amount = (transactions[transactions["fraud_flag"] == 1]["amount"].mean()
                            if flagged > 0 else 0)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Transacciones", f"{len(transactions):,}")
        col_b.metric("Flagueadas Fraude",   f"{flagged:,}")
        col_c.metric("Monto Prom. Fraude",  f"${avg_fraud_amount:,.0f}" if flagged > 0 else "—")

        col_a, col_b = st.columns(2)

        FRAUD_LABELS = {
            "normal":    "Normal",
            "suspicious": "Sospechoso",
            "high_risk": "Alto Riesgo",
            "critical":  "Crítico",
        }
        FRAUD_COLORS_ES = {
            "Normal":      "#059669",
            "Sospechoso":  "#d97706",
            "Alto Riesgo": "#ea580c",
            "Crítico":     "#dc2626",
        }
        MERCHANT_LABELS = {
            "luxury":     "Lujo",
            "travel":     "Viajes",
            "utility":    "Servicios",
            "online":     "Online",
            "retail":     "Retail",
            "atm":        "Cajero",
            "restaurant": "Restaurant",
            "gas":        "Combustible",
            "grocery":    "Supermercado",
            "health":     "Salud",
        }

        with col_a:
            st.subheader("Categorías de Alerta")
            cat_counts = transactions["fraud_category"].map(
                lambda x: FRAUD_LABELS.get(str(x), str(x))
            ).value_counts().reset_index()
            cat_counts.columns = ["Categoría", "Transacciones"]
            fig = px.pie(
                cat_counts, names="Categoría", values="Transacciones",
                color="Categoría", color_discrete_map=FRAUD_COLORS_ES, hole=0.45,
            )
            fig.update_traces(textposition="outside", textinfo="percent+label",
                              marker=dict(line=dict(color="#0f172a", width=2)))
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Fraude por Tipo de Comercio")
            fraud_by_merchant = (transactions.groupby("merchant_category")["fraud_flag"]
                                 .mean().reset_index())
            fraud_by_merchant["merchant_category"] = fraud_by_merchant["merchant_category"].map(
                lambda x: MERCHANT_LABELS.get(str(x), str(x))
            )
            fraud_by_merchant.columns = ["Comercio", "Tasa de Fraude"]
            fig = px.bar(
                fraud_by_merchant.sort_values("Tasa de Fraude", ascending=False),
                x="Comercio", y="Tasa de Fraude",
                color="Tasa de Fraude",
                color_continuous_scale=[[0, "#1e3a5f"], [0.5, "#ea580c"], [1, "#dc2626"]],
            )
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Fraude por Hora del Día")
        hourly = transactions.groupby("hour_of_day")["fraud_flag"].mean().reset_index()
        fig = px.area(
            hourly, x="hour_of_day", y="fraud_flag",
            labels={"hour_of_day": "Hora", "fraud_flag": "Tasa de Fraude"},
            color_discrete_sequence=["#dc2626"],
        )
        fig.update_traces(fillcolor="rgba(220,38,38,0.15)", line_color="#dc2626")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MACRO
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    macro = summary.get("macro", {})

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Riesgo Mercado",       macro.get("market_risk", "N/A").upper())
    col_b.metric("Stress Macro",         macro.get("macro_stress", "N/A").upper())
    col_c.metric("Desempleo Proyectado", f"{macro.get('unemployment_forecast', 0):.1f}%")
    col_d.metric("Tasa Proyectada",      f"{macro.get('rate_forecast', 0):.1f}%")

    st.subheader("Escenarios Económicos")
    scenarios_data = []
    for sc_key, sc_val in config["scenarios"].items():
        is_active = sc_key == summary.get("scenario", "normal")
        scenarios_data.append({
            "Escenario":    sc_val["label"],
            "Δ Desempleo":   f"+{sc_val.get('unemployment_delta', 0):.1f}%",
            "Δ Tasa":        f"+{sc_val.get('rate_delta', 0):.1f}%",
            "Caída Ingreso": f"-{sc_val.get('income_drop', 0)*100:.0f}%",
            "Stress Pagos":  f"+{sc_val.get('payment_stress', 0)*100:.0f}%",
            "Activo":        "✅" if is_active else "",
        })
    df_scenarios = pd.DataFrame(scenarios_data)
    # Renderizar como HTML para evitar LargeUtf8 bug en Streamlit 1.31 + pandas 3.x
    headers = "".join(f"<th style='padding:8px 16px;text-align:left;color:#94a3b8;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.05em;border-bottom:1px solid #334155'>{c}</th>" for c in df_scenarios.columns)
    rows = ""
    for _, row in df_scenarios.iterrows():
        cells = "".join(f"<td style='padding:8px 16px;color:#e2e8f0;font-size:0.88rem;border-bottom:1px solid #1e293b'>{v}</td>" for v in row)
        rows += f"<tr>{cells}</tr>"
    st.markdown(f"""
    <table style='width:100%;border-collapse:collapse;background:#1e293b;border-radius:8px;overflow:hidden'>
        <thead><tr>{headers}</tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """, unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — AI REPORTS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:

    # ── Métricas rápidas ──────────────────────────────────────────────────────
    if not clients.empty:
        total     = len(clients)
        n_high    = int((clients["risk_level"].astype(str) == "high").sum())
        n_review  = int((clients["decision"].astype(str) == "REVISAR").sum())
        n_reject  = int((clients["decision"].astype(str) == "RECHAZAR").sum())
        avg_pd    = clients["pd"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clientes Alto Riesgo", f"{n_high:,}",  f"{n_high/total:.1%} del total")
        c2.metric("PD Promedio Cartera",  f"{avg_pd:.1%}")
        c3.metric("Pendientes Revisión",  f"{n_review:,}")
        c4.metric("Rechazados",           f"{n_reject:,}", f"{n_reject/total:.1%} del total")

    st.markdown("---")

    # ── Resumen de cartera con IA ─────────────────────────────────────────────
    st.subheader("Resumen de Cartera")
    st.caption("Análisis narrativo del estado actual del portafolio generado por IA.")

    if st.button("Generar Resumen de Cartera", use_container_width=True):
        placeholder2 = st.empty()
        with placeholder2.container():
            spinner_html("Analizando cartera completa...")

        summary_data   = load_summary()
        macro_forecast = summary_data.get("macro", {})

        n_medium = int((clients["risk_level"] == "medium").sum()) if "risk_level" in clients.columns else 0
        n_review = int((clients["decision"] == "REVISAR").sum())  if "decision"   in clients.columns else 0

        portfolio_stats = {
            "total_clients":   total,
            "avg_pd":          float(clients["pd"].mean()),
            "avg_score":       float(clients["credit_score"].mean()),
            "high_risk_pct":   n_high   / total,
            "medium_risk_pct": n_medium / total,
            "low_risk_pct":    (total - n_high - n_medium) / total,
            "reject_pct":      n_reject / total,
            "review_pct":      n_review / total,
        }
        fraud_s = {
            "fraud_rate":         float(transactions["fraud_flag"].mean()) if not transactions.empty and "fraud_flag" in transactions.columns else 0.0,
            "total_fraud_alerts": int(transactions["fraud_flag"].sum())    if not transactions.empty and "fraud_flag" in transactions.columns else 0,
        }
        report_portfolio = generate_portfolio_report(
            portfolio_stats=portfolio_stats,
            macro=macro_forecast,
            fraud_summary=fraud_s,
            scenario_name=selected_label,
        )
        placeholder2.empty()

        from html import escape
        fmt = escape(report_portfolio)  # CN-005: escapar HTML antes de formatear
        fmt = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#f1f5f9;font-size:0.92rem">\1:</strong> ', fmt)
        fmt = re.sub(r'_([^_]+)_', r'<span style="color:#475569;font-size:0.78rem">\1</span>', fmt)
        fmt = fmt.replace(chr(10), "<br>")
        fmt = re.sub(r'(<br>\s*){2,}', '<br>', fmt)
        fmt = re.sub(r'^(<br>)+', '', fmt)
        st.markdown(f"""
        <div style='background:#1e293b;border:1px solid #334155;border-radius:12px;
                    padding:24px 28px;line-height:1.8;color:#cbd5e1;font-size:0.9rem;'>
            {fmt}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Reporte Individual por Cliente")

    _search_col5, _ = st.columns([1, 2])
    with _search_col5:
        with st.form(key="form_search_tab5", border=False):
            _col_id, _col_gen = st.columns([3, 1])
            with _col_id:
                selected_id = st.number_input(
                    "Client ID",
                    min_value=1,
                    max_value=int(clients["client_id"].max()) if not clients.empty else 1000,
                    value=1,
                    label_visibility="collapsed",
                    key="search_client_tab5",
                )
            with _col_gen:
                _generate_clicked = st.form_submit_button("Generar →", use_container_width=True, type="primary")

    if _generate_clicked:
        client_row = clients[clients["client_id"] == selected_id]
        if client_row.empty:
            st.error(f"Cliente {selected_id} no encontrado.")
        else:
            placeholder = st.empty()
            with placeholder.container():
                spinner_html("Consultando modelo de lenguaje...")

            client_dict  = client_row.iloc[0].to_dict()
            client_txns  = transactions[transactions["client_id"] == selected_id] if not transactions.empty else pd.DataFrame()
            fraud_summary = {
                "fraud_rate": float(transactions["fraud_flag"].mean()) if "fraud_flag" in transactions.columns else 0.0,
                "high_risk_transactions": int((client_txns["fraud_flag"] == 1).sum())
                    if not client_txns.empty and "fraud_flag" in client_txns.columns else 0,
            }
            macro_forecast = summary.get("macro", {})

            report_text = generate_report(
                client=client_dict,
                macro_forecast=macro_forecast,
                macro_analysis=macro_forecast,
                fraud_summary=fraud_summary,
                scenario_name=selected_label,
            )
            placeholder.empty()

            from html import escape
            formatted = escape(report_text)  # CN-005: escapar HTML antes de formatear
            formatted = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#f1f5f9;font-size:0.92rem">\1:</strong> ', formatted)
            # Italics discretos al final (label del modelo)
            formatted = re.sub(r'_([^_]+)_', r'<span style="color:#475569;font-size:0.78rem">\1</span>', formatted)
            formatted = formatted.replace(chr(10), "<br>")
            formatted = re.sub(r'(<br>\s*){2,}', '<br>', formatted)
            formatted = re.sub(r'^(<br>)+', '', formatted)
            st.markdown(f"""
            <div style='background:#1e293b; border:1px solid #334155; border-radius:12px;
                        padding:24px 28px; line-height:1.8; color:#cbd5e1; font-size:0.9rem;'>
                {formatted}
            </div>
            """, unsafe_allow_html=True)
