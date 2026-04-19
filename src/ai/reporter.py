"""
AI Reports: genera análisis narrativo de riesgo.
Prioridad de backends:
  1. Groq  (cloud, gratis, rápido) → variable de entorno GROQ_API_KEY (en .env)
  2. Ollama (local)                → si está corriendo en localhost:11434
  3. Template fallback             → siempre disponible
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"   # mejor calidad en tier gratuito

# CN-003: validar que OLLAMA_URL apunte solo a localhost (evitar SSRF)
_raw_ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
try:
    from urllib.parse import urlparse as _urlparse
    _p = _urlparse(_raw_ollama_url)
    if not (_p.scheme in ("http", "https") and _p.hostname in ("localhost", "127.0.0.1")):
        import logging as _log
        _log.warning("[OLLAMA] OLLAMA_URL apunta fuera de localhost (%s) — usando default seguro.", _p.hostname)
        _raw_ollama_url = "http://localhost:11434/api/generate"
except Exception:
    _raw_ollama_url = "http://localhost:11434/api/generate"
OLLAMA_URL = _raw_ollama_url


# ── Detección de backend ──────────────────────────────────────────────────────

def _groq_key() -> str | None:
    return os.environ.get("GROQ_API_KEY") or None


def _is_ollama_running() -> bool:
    try:
        import requests
        return requests.get("http://localhost:11434/api/tags", timeout=2).status_code == 200
    except Exception:
        return False


def _active_backend() -> str:
    if _groq_key():
        return "groq"
    if _is_ollama_running():
        return "ollama"
    return "template"


# ── Prompt ────────────────────────────────────────────────────────────────────

def _build_prompt(client: dict, macro: dict, fraud_summary: dict, scenario_name: str, shap_text: str = "") -> str:
    pd_val      = client.get("pd", 0.0)
    risk_level  = client.get("risk_level", "unknown")
    decision    = client.get("decision", "REVISAR")
    credit_score = client.get("credit_score", 600)
    fraud_rate  = fraud_summary.get("fraud_rate", 0.0)
    fraud_alerts = fraud_summary.get("high_risk_transactions", 0)

    return f"""Eres un analista de riesgo crediticio senior de un banco chileno.
Genera un reporte ejecutivo conciso (maximo 250 palabras) con lenguaje profesional bancario.

== PERFIL DEL CLIENTE ==
- ID: {client.get('client_id', 'N/A')}
- Edad: {client.get('age', 'N/A')} anios
- Ingreso mensual: ${client.get('income', 0):,.0f} CLP
- Ratio de deuda: {client.get('debt_ratio', 0):.0%}
- Estado laboral: {client.get('employment_status', 'desconocido')}
- Historial crediticio: {client.get('credit_history_months', 0)} meses
- Pagos atrasados: {client.get('missed_payments', 0)}
- Monto solicitado: ${client.get('loan_amount', 0):,.0f} CLP

== RESULTADO DEL MODELO ==
- Probabilidad de Default (PD): {pd_val:.1%}
- Nivel de riesgo: {str(risk_level).upper()}
- Credit Score: {credit_score}/850
- Decision recomendada: {decision}

== CONTEXTO MACROECONOMICO ({scenario_name}) ==
- Desempleo proyectado: {macro.get('unemployment_forecast', 8.5):.1f}%
- TPM / Tasa proyectada: {macro.get('rate_forecast', 5.0):.1f}%
- Inflacion proyectada: {macro.get('inflation_forecast', 4.0):.1f}%
- Riesgo de mercado: {macro.get('market_risk', 'medium').upper()}
- Estres macro: {macro.get('macro_stress', 'low').upper()}
- UF actual: ${macro.get('uf_value', 'N/D')}
- USD/CLP oficial: ${macro.get('usdclp_official', 'N/D')}

== ALERTAS DE FRAUDE ==
- Tasa de fraude del portafolio: {fraud_rate:.1%}
- Transacciones de alto riesgo del cliente: {fraud_alerts}

{shap_text}

Estructura tu reporte con estas 4 secciones:
1. **Resumen ejecutivo** (2-3 oraciones)
2. **Factores de riesgo principales** (usa los factores SHAP si están disponibles, si no usa el perfil del cliente)
3. **Contexto macroeconomico** (1-2 oraciones)
4. **Decision y condiciones recomendadas** (especifico y accionable)

Se directo. El banco necesita saber QUE DECISION TOMAR y POR QUE."""


# ── Backends ──────────────────────────────────────────────────────────────────

def _call_groq(prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=_groq_key(), timeout=30.0)  # CN-017: timeout explícito
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def _call_ollama(prompt: str, model: str = "llama3.2:3b") -> str:
    import requests
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60,  # CN-017: reducido de 120s a 60s
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


# ── API pública ───────────────────────────────────────────────────────────────

def generate_report(
    client: dict,
    macro_forecast: dict,
    macro_analysis: dict,
    fraud_summary: dict,
    scenario_name: str = "Escenario Base",
    model: str = "llama3.2:3b",
    shap_explanation: dict | None = None,
) -> str:
    """
    Genera reporte narrativo. Usa Groq > Ollama > template según disponibilidad.
    """
    from src.credit_risk.explainer import format_shap_for_prompt
    shap_text = format_shap_for_prompt(shap_explanation) if shap_explanation else ""

    combined_macro = {**macro_forecast, **macro_analysis}
    prompt  = _build_prompt(client, combined_macro, fraud_summary, scenario_name, shap_text)
    backend = _active_backend()

    if backend == "groq":
        try:
            text = _call_groq(prompt)
            return f"{text}\n\n_{GROQ_MODEL} via Groq_"
        except Exception as e:
            import logging
            logging.warning("[Groq] Error al generar reporte: %s — intentando Ollama", type(e).__name__)
            if _is_ollama_running():
                backend = "ollama"
            else:
                return _fallback_report(client, macro_forecast, scenario_name)

    if backend == "ollama":
        try:
            text = _call_ollama(prompt, model)
            return f"{text}\n\n_Ollama (local)_"
        except Exception as e:
            import logging
            logging.warning("[Ollama] Error al generar reporte: %s", type(e).__name__)

    return _fallback_report(client, macro_forecast, scenario_name)


def _fallback_report(client: dict, macro: dict, scenario_name: str) -> str:
    risk_level = client.get("risk_level", "unknown")
    pd_val     = client.get("pd", 0.0)
    decision   = client.get("decision", "REVISAR")

    risk_text = {
        "high":   "alto riesgo crediticio, con alta probabilidad de incumplimiento",
        "medium": "riesgo crediticio moderado, requiere condiciones especiales",
        "low":    "bajo riesgo crediticio, candidato apto para credito estandar",
    }.get(str(risk_level), "riesgo indeterminado")

    return f"""**Reporte de Riesgo Crediticio — {scenario_name}**

**Resumen ejecutivo**
El cliente presenta {risk_text}. Con una probabilidad de default de {pd_val:.1%}
y un score crediticio de {client.get('credit_score', 600)}/850, la decision recomendada es **{decision}**.

**Factores de riesgo principales**
- Ratio de deuda: {client.get('debt_ratio', 0):.0%} {'(elevado)' if client.get('debt_ratio', 0) > 0.5 else '(controlado)'}
- Pagos atrasados: {client.get('missed_payments', 0)} {'(senal de alerta)' if client.get('missed_payments', 0) > 2 else ''}
- Estado laboral: {client.get('employment_status', 'N/A')}
- Historial crediticio: {client.get('credit_history_months', 0)} meses

**Contexto macroeconomico**
Bajo el {scenario_name}, desempleo proyectado {macro.get('unemployment_forecast', 8.5):.1f}%,
TPM {macro.get('rate_forecast', 5.0):.1f}%, stress macro {macro.get('macro_stress', 'N/A').upper()}.

**Decision recomendada**
{_get_recommendation(decision, pd_val)}"""


def _get_recommendation(decision: str, pd: float) -> str:
    if decision == "RECHAZAR":
        return (f"Rechazar solicitud. PD={pd:.1%} supera umbral maximo. "
                "Cliente puede repostular en 6 meses con mejora en historial de pagos.")
    elif decision == "REVISAR":
        return (f"Aprobar con condiciones. PD={pd:.1%} requiere tasa diferenciada (+2-3%), "
                "garantia adicional y limite reducido al 70% del solicitado.")
    return (f"Aprobar. PD={pd:.1%} dentro de parametros normales. "
            "Proceder con tasa estandar y condiciones regulares.")


def generate_portfolio_report(
    portfolio_stats: dict,
    macro: dict,
    fraud_summary: dict,
    scenario_name: str = "Escenario Base",
) -> str:
    """
    Genera un resumen ejecutivo del portafolio completo. Usa Groq > template.
    """
    total       = portfolio_stats.get("total_clients", 0)
    avg_pd      = portfolio_stats.get("avg_pd", 0.0)
    high_pct    = portfolio_stats.get("high_risk_pct", 0.0)
    medium_pct  = portfolio_stats.get("medium_risk_pct", 0.0)
    low_pct     = portfolio_stats.get("low_risk_pct", 0.0)
    reject_pct  = portfolio_stats.get("reject_pct", 0.0)
    review_pct  = portfolio_stats.get("review_pct", 0.0)
    avg_score   = portfolio_stats.get("avg_score", 600)
    fraud_rate  = fraud_summary.get("fraud_rate", 0.0)
    fraud_total = fraud_summary.get("total_fraud_alerts", 0)

    # Clasificación interna del riesgo del portafolio (relativa al modelo, no a benchmarks reales)
    if avg_pd < 0.20:
        pd_nivel = "controlado para este escenario"
        pd_contexto = "portafolio en zona de operación normal del modelo"
    elif avg_pd < 0.35:
        pd_nivel = "moderadamente elevado"
        pd_contexto = "portafolio requiere monitoreo activo"
    elif avg_pd < 0.50:
        pd_nivel = "alto — escenario de stress"
        pd_contexto = "portafolio bajo presión significativa"
    else:
        pd_nivel = "crítico — escenario de stress severo"
        pd_contexto = "portafolio en zona de alerta máxima"

    high_conc = "normal" if high_pct < 0.20 else ("elevada" if high_pct < 0.35 else "crítica")

    prompt = f"""Eres un analista de riesgo senior de un banco retail chileno.
Genera un resumen ejecutivo del portafolio simulado bajo el escenario "{scenario_name}" (máximo 280 palabras).
Usa lenguaje profesional bancario. Este análisis corresponde a una simulación con datos sintéticos —
las métricas absolutas de PD son más altas que en carteras reales, lo relevante es la lectura
relativa del escenario y las señales de riesgo internas del portafolio.

== ESTADO DEL PORTAFOLIO ==
- Total clientes evaluados: {total:,}
- PD promedio: {avg_pd:.1%} — nivel {pd_nivel} ({pd_contexto})
- Distribución de riesgo: ALTO {high_pct:.1%} (concentración {high_conc}) | MEDIO {medium_pct:.1%} | BAJO {low_pct:.1%}
- Decisiones: RECHAZADOS {reject_pct:.1%} | EN REVISIÓN {review_pct:.1%} | APROBADOS {1-reject_pct-review_pct:.1%}
- Credit Score promedio: {avg_score:.0f}/850

== CONTEXTO MACROECONÓMICO ({scenario_name}) ==
- Desempleo proyectado: {macro.get('unemployment_forecast', 8.5):.1f}%
- TPM proyectada: {macro.get('rate_forecast', 5.0):.1f}%
- Inflación proyectada: {macro.get('inflation_forecast', 4.0):.1f}%
- Riesgo de mercado: {macro.get('market_risk', 'medium').upper()}
- Estrés macro: {macro.get('macro_stress', 'low').upper()}
- UF actual: ${macro.get('uf_value', 'N/D')}
- USD/CLP: ${macro.get('usdclp_official', 'N/D')}

== FRAUDE ==
- Tasa de fraude del portafolio: {fraud_rate:.1%}
- Total alertas de alto riesgo: {fraud_total:,}

Estructura tu respuesta con estas 4 secciones:
1. **Resumen ejecutivo** (2-3 oraciones: evalúa el portafolio en el contexto del escenario simulado)
2. **Principales factores de riesgo** (señales internas del portafolio que más impactan la PD)
3. **Contexto macroeconómico** (cómo las variables del escenario {scenario_name} presionan al portafolio)
4. **Recomendaciones** (acciones concretas y priorizadas: política de crédito, mora, fraude)

Enfócate en las señales relativas del portafolio y el escenario simulado, no en comparaciones con la banca real."""

    backend = _active_backend()

    if backend == "groq":
        try:
            text = _call_groq(prompt)
            return f"{text}\n\n_{GROQ_MODEL} via Groq_"
        except Exception as e:
            print(f"  [Groq] error: {e}")
            if _is_ollama_running():
                backend = "ollama"
            else:
                return _fallback_portfolio_report(portfolio_stats, macro, scenario_name)

    if backend == "ollama":
        try:
            text = _call_ollama(prompt)
            return f"{text}\n\n_Ollama (local)_"
        except Exception as e:
            print(f"  [Ollama] error: {e}")

    return _fallback_portfolio_report(portfolio_stats, macro, scenario_name)


def _fallback_portfolio_report(portfolio_stats: dict, macro: dict, scenario_name: str) -> str:
    total      = portfolio_stats.get("total_clients", 0)
    avg_pd     = portfolio_stats.get("avg_pd", 0.0)
    high_pct   = portfolio_stats.get("high_risk_pct", 0.0)
    reject_pct = portfolio_stats.get("reject_pct", 0.0)
    avg_score  = portfolio_stats.get("avg_score", 600)

    risk_level = "elevado" if avg_pd > 0.25 else "moderado" if avg_pd > 0.15 else "controlado"

    return f"""**Resumen de Cartera — {scenario_name}**

**Resumen ejecutivo:**
El portafolio de {total:,} clientes presenta un riesgo {risk_level}, con PD promedio de {avg_pd:.1%} y score crediticio de {avg_score:.0f}/850.
Un {high_pct:.1%} de los clientes se clasifica como alto riesgo, lo que requiere atención prioritaria.

**Principales factores de riesgo:**
- Concentración de alto riesgo: {high_pct:.1%} del portafolio
- Tasa de rechazo: {reject_pct:.1%} de las solicitudes
- Score promedio por debajo del umbral óptimo (750)

**Contexto macroeconómico:**
Bajo el {scenario_name}, desempleo proyectado {macro.get('unemployment_forecast', 8.5):.1f}%, TPM {macro.get('rate_forecast', 5.0):.1f}%.
Estrés macro {macro.get('macro_stress', 'N/A').upper()} — impacto {'significativo' if macro.get('macro_stress') == 'high' else 'moderado'} esperado en mora.

**Recomendaciones:**
{'Revisar política de otorgamiento y reforzar cobranza preventiva dado el alto estrés macro.' if macro.get('macro_stress') == 'high' else 'Mantener umbrales actuales. Monitorear clientes REVISAR mensualmente.'}
Priorizar gestión de los {int(total * high_pct):,} clientes de alto riesgo antes del próximo corte."""


def batch_generate_reports(
    clients_df,
    macro_forecast: dict,
    macro_analysis: dict,
    fraud_df,
    scenario_name: str = "Escenario Base",
    n_reports: int = 5,
    model: str = "llama3.2:3b",
    shap_explanations: dict | None = None,
) -> list[dict]:
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor, as_completed

    top_clients = clients_df.nlargest(n_reports, "pd")

    # Pre-calcular fraud_rate una sola vez (es igual para todos los clientes)
    global_fraud_rate = (
        float(fraud_df["fraud_flag"].mean())
        if fraud_df is not None and "fraud_flag" in fraud_df.columns
        else 0.0
    )

    def _generate_one(args: tuple) -> tuple[int, dict]:
        idx, row = args
        client_dict = row.to_dict()
        client_id   = client_dict.get("client_id")

        client_txns = fraud_df[fraud_df["client_id"] == client_id] if fraud_df is not None else pd.DataFrame()
        fraud_summary = {
            "fraud_rate": global_fraud_rate,
            "high_risk_transactions": int((client_txns["fraud_flag"] == 1).sum()) if not client_txns.empty and "fraud_flag" in client_txns.columns else 0,
        }

        shap_exp    = (shap_explanations or {}).get(client_id)
        report_text = generate_report(
            client=client_dict,
            macro_forecast=macro_forecast,
            macro_analysis=macro_analysis,
            fraud_summary=fraud_summary,
            scenario_name=scenario_name,
            model=model,
            shap_explanation=shap_exp,
        )

        return idx, {
            "client_id": client_id,
            "risk_level": client_dict.get("risk_level"),
            "pd": client_dict.get("pd"),
            "decision": client_dict.get("decision"),
            "report": report_text,
        }

    rows = list(enumerate(row for _, row in top_clients.iterrows()))

    # Groq free tier: máx 2 workers simultáneos para evitar rate-limit (30 req/min)
    max_workers = 2 if _groq_key() else 3
    results: dict[int, dict] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_generate_one, args): args[0] for args in rows}
        for future in as_completed(futures):
            idx, report = future.result()
            results[idx] = report

    # Devolver en el mismo orden que top_clients
    return [results[i] for i in range(len(rows))]
