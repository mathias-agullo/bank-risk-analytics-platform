"""
Generador de datos simulados: clientes y transacciones bancarias.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SIMULATED_DIR = Path(__file__).parent
PROCESSED_DIR = SIMULATED_DIR.parent / "processed"


def generate_clients(n: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Genera dataset de clientes con features de riesgo crediticio."""
    rng = np.random.default_rng(random_state)

    age = rng.integers(22, 70, n)
    income = rng.lognormal(mean=13.5, sigma=0.6, size=n).clip(300_000, 15_000_000)
    debt_ratio = rng.beta(2, 5, n).clip(0.0, 0.95)
    employment_status = rng.choice(
        ["employed", "self_employed", "unemployed", "retired"],
        p=[0.60, 0.20, 0.12, 0.08],
        size=n,
    )
    credit_history_months = rng.integers(0, 240, n)
    num_loans = rng.integers(0, 6, n)
    loan_amount = rng.lognormal(mean=14.5, sigma=0.8, size=n).clip(100_000, 50_000_000)
    missed_payments = rng.integers(0, 12, n)
    savings_ratio = rng.beta(1.5, 4, n).clip(0.0, 0.80)
    region = rng.choice(
        ["RM", "Valparaíso", "Biobío", "Maule", "Araucanía", "Antofagasta"],
        p=[0.40, 0.15, 0.15, 0.10, 0.10, 0.10],
        size=n,
    )
    loan_to_income = (loan_amount / income.clip(1)).clip(0, 20)

    # Fechas de originación distribuidas 2020–2024 (para validación temporal)
    days_range = (pd.Timestamp("2024-12-31") - pd.Timestamp("2020-01-01")).days
    day_offsets = rng.integers(0, days_range, n)
    origination_date = pd.Timestamp("2020-01-01") + pd.to_timedelta(day_offsets, unit="D")

    # Probabilidad de default — logit más rico que incluye savings y loan_to_income
    logit = (
        -3.5
        + 0.8  * (employment_status == "unemployed").astype(float)
        + 0.3  * (employment_status == "self_employed").astype(float)
        + 2.5  * debt_ratio
        - 0.005 * credit_history_months
        + 0.2  * num_loans
        + 0.15 * missed_payments
        - 0.003 * (income / 100_000)
        + 0.3  * loan_to_income.clip(0, 10)
        - 0.5  * savings_ratio
        + rng.normal(0, 0.3, n)   # ruido reducido a 0.3 (era 0.5)
    )
    prob_default = 1 / (1 + np.exp(-logit))
    default = (rng.uniform(size=n) < prob_default).astype(int)

    df = pd.DataFrame({
        "client_id":              range(1, n + 1),
        "age":                    age,
        "income":                 income.astype(int),
        "debt_ratio":             debt_ratio.round(4),
        "employment_status":      employment_status,
        "credit_history_months":  credit_history_months,
        "num_loans":              num_loans,
        "loan_amount":            loan_amount.astype(int),
        "missed_payments":        missed_payments,
        "savings_ratio":          savings_ratio.round(4),
        "region":                 region,
        "origination_date":       origination_date.strftime("%Y-%m-%d"),
        "default":                default,
    })
    return df


def generate_transactions(clients: pd.DataFrame, n: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """Genera dataset de transacciones con etiquetas de fraude."""
    rng = np.random.default_rng(random_state + 1)

    client_ids = rng.choice(clients["client_id"].values, size=n)
    amount = rng.lognormal(mean=10.5, sigma=1.2, size=n).clip(500, 5_000_000)
    merchant_category = rng.choice(
        ["retail", "restaurant", "travel", "online", "atm", "utility", "luxury"],
        p=[0.30, 0.20, 0.10, 0.20, 0.10, 0.07, 0.03],
        size=n,
    )
    hour_of_day = rng.integers(0, 24, n)
    is_foreign  = rng.choice([0, 1], p=[0.88, 0.12], size=n)
    is_weekend  = rng.choice([0, 1], p=[0.70, 0.30], size=n)
    channel     = rng.choice(["web", "mobile", "pos", "atm"], p=[0.25, 0.35, 0.30, 0.10], size=n)

    fraud_logit = (
        -4.5
        + 0.0000003 * amount
        + 1.2 * is_foreign
        + 0.5 * (merchant_category == "luxury").astype(float)
        + 0.4 * ((hour_of_day < 5) | (hour_of_day > 23)).astype(float)
        + 0.3 * is_weekend
        + rng.normal(0, 0.8, n)
    )
    fraud_prob = 1 / (1 + np.exp(-fraud_logit))
    fraud = (rng.uniform(size=n) < fraud_prob).astype(int)

    return pd.DataFrame({
        "transaction_id":    range(1, n + 1),
        "client_id":         client_ids,
        "amount":            amount.astype(int),
        "merchant_category": merchant_category,
        "hour_of_day":       hour_of_day,
        "is_foreign":        is_foreign,
        "is_weekend":        is_weekend,
        "channel":           channel,
        "fraud":             fraud,
    })


def run(n_clients: int = 5000, n_transactions: int = 10000, random_state: int = 42):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Generando datos de clientes...")
    clients = generate_clients(n_clients, random_state)
    clients.to_csv(PROCESSED_DIR / "clients.csv", index=False)
    print(f"  {len(clients)} clientes | default rate: {clients['default'].mean():.1%}")

    print("Generando transacciones...")
    transactions = generate_transactions(clients, n_transactions, random_state)
    transactions.to_csv(PROCESSED_DIR / "transactions.csv", index=False)
    print(f"  {len(transactions)} transacciones | fraud rate: {transactions['fraud'].mean():.1%}")

    return clients, transactions


if __name__ == "__main__":
    run()
