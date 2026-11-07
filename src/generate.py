import json
import math
import os
import re
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI

# Load environment variables (.env in project root)
load_dotenv()

BLABLADOR_API_KEY = os.getenv("BLABLADOR_API_KEY")
BLABLADOR_BASE_URL = "https://api.blablador.fz-juelich.de/v1"

# ==========================================
# 1. Historical Baseline & Data Engine
# ==========================================
current_code_path = Path(__file__).resolve().parent
project_path = current_code_path.parent

data_path = project_path / "data"
dynamic_path = data_path / "table_dynamic.csv"
static_path = data_path / "table_static.csv"
income_path = data_path / "table_paycheck.csv"

for p in [data_path, dynamic_path, static_path, income_path]:
    if not p.exists():
        raise FileNotFoundError(f"Required path {p} not found.")

dynamic_df = pd.read_csv(dynamic_path)
static_df = pd.read_csv(static_path)
income_df = pd.read_csv(income_path)

# Monthly Cash Flow & Sinking Fund Engine
monthly_income = income_df.groupby("Month")["Amount"].sum()
monthly_static = static_df.groupby("Month")["Amount"].sum()
monthly_dynamic = dynamic_df.groupby("Month")["Amount"].sum()

monthly_summary = pd.DataFrame(
    {
        "income": monthly_income,
        "static": monthly_static,
        "dynamic": monthly_dynamic,
    }
).fillna(0)

monthly_summary["target_sinking_fund"] = monthly_summary["income"] * 0.05
monthly_summary["raw_surplus"] = monthly_summary["income"] - (
    monthly_summary["static"] + monthly_summary["dynamic"]
)
monthly_summary["actual_sinking_fund"] = (
    monthly_summary["raw_surplus"]
    .clip(lower=0)
    .combine(monthly_summary["target_sinking_fund"], min)
)
monthly_summary["remaining_funds"] = monthly_summary["income"] - (
    monthly_summary["static"]
    + monthly_summary["dynamic"]
    + monthly_summary["actual_sinking_fund"]
)

# Exportable Historical Baselines
total_months = len(monthly_summary)
category_monthly_avg = (
    (dynamic_df.groupby("Category")["Amount"].sum() / total_months)
    .round(2)
    .sort_values(ascending=False)
)

frequent_merchants = (
    dynamic_df.groupby(["Category", "Description"])["Amount"]
    .agg(total_spent="sum", frequency="count")
    .sort_values(by="frequency", ascending=False)
    .head(10)
    .reset_index()
)

avg_income = float(monthly_summary["income"].mean())
avg_static = float(monthly_summary["static"].mean())
avg_dynamic = float(monthly_summary["dynamic"].mean())
avg_sinking_fund = float(monthly_summary["actual_sinking_fund"].mean())
avg_buffer = float(monthly_summary["remaining_funds"].mean())


# ==========================================
# 2. AI Coaching Engine (Blablador)
# ==========================================
def generate_ai_tradeoffs(deficit, category_averages, merchant_patterns, goal_price=None, months=None, is_recalibrated=False):
    """
    Calls Blablador's alias-fast model to generate trade-offs as structured JSON.
    """
    goal_text = f"${goal_price:,.2f} trip in {months} months" if goal_price and months else "savings goal"

    default_fallback = {
        "deficit": round(deficit, 2),
        "total_planned_savings": round(deficit, 2),
        "top_categories": [
            {
                "category": category_averages.index[0] if len(category_averages) > 0 else "Discretionary Spending",
                "current_monthly_spend": float(category_averages.iloc[0]) if len(category_averages) > 0 else 0.0,
                "recommended_cut": round(deficit, 2),
                "quick_win_rationale": "High discretionary spend with flexible reduction room.",
                "actionable_swaps": [
                    f"Trim non-essential purchases to save ~${deficit:.0f}/month.",
                    "Review recurring subscriptions and dining frequency."
                ]
            }
        ],
        "encouragement": f"Saving an extra ${deficit:.2f}/month keeps your plan on track for your {goal_text}."
    }

    if not BLABLADOR_API_KEY:
        print("[Notice] BLABLADOR_API_KEY missing. Using fallback JSON.")
        return default_fallback

    client = OpenAI(
        base_url=BLABLADOR_BASE_URL,
        api_key=BLABLADOR_API_KEY
    )

    context_note = (
        "Note: The user's original goal was over budget. This plan is calibrated to a realistic, "
        "sustainable 70% feasibility budget."
        if is_recalibrated
        else "This plan covers the calculated monthly gap."
    )

    system_prompt = (
        "You are an automated financial planning API engine. "
        "You must respond ONLY with a raw JSON object matching the requested schema. "
        "Do not include Markdown explanations or conversational text outside the JSON. "
        "Never suggest cutting static bills (rent, utilities, insurance)."
    )

    user_prompt = f"""
GOAL CONTEXT:
Save ${deficit:.2f}/month to fund a {goal_text}. {context_note}

AVERAGE MONTHLY DISCRETIONARY SPENDING (BY CATEGORY):
{category_averages.to_string()}

TOP FREQUENT MERCHANTS & PATTERNS:
{merchant_patterns[['Category', 'Description', 'total_spent', 'frequency']].to_string(index=False)}

Generate a single valid JSON object strictly matching this schema:
{{
  "deficit": {deficit:.2f},
  "total_planned_savings": <float, sum of recommended_cut values roughly equal to {deficit:.2f}>,
  "top_categories": [
    {{
      "category": "<Category Name>",
      "current_monthly_spend": <float>,
      "recommended_cut": <float>,
      "quick_win_rationale": "<1 short sentence why this category is flexible>",
      "actionable_swaps": [
        "<concrete habit tweak referencing actual merchants>",
        "<concrete habit tweak referencing actual merchants>"
      ]
    }}
  ],
  "encouragement": "<2 short motivational sentences on achieving the goal>"
}}
"""

    try:
        response = client.chat.completions.create(
            model="alias-fast",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1500,
            temperature=0.2
        )

        msg = response.choices[0].message
        content = msg.content or getattr(msg, "reasoning_content", None)

        if not content:
            return default_fallback

        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))

        return json.loads(content)

    except Exception as e:
        print(f"[Warning] Blablador JSON generation failed ({e}). Returning fallback schema.")
        return default_fallback


# ==========================================
# 3. Main Django Service Function
# ==========================================
def evaluate_financial_plan(budget: float, months: int) -> dict:
    """
    Primary interface for Django views.
    Takes user goal inputs and returns complete calculation + AI payload.
    """
    months = max(1, int(months))
    budget = float(budget)
    required_monthly = budget / months

    baseline = {
        "income": round(avg_income, 2),
        "static": round(avg_static, 2),
        "dynamic": round(avg_dynamic, 2),
        "sinking_fund": round(avg_sinking_fund, 2),
        "buffer": round(avg_buffer, 2)
    }

    # Case A: Effortless (Buffer covers required deposit)
    if avg_buffer >= required_monthly:
        return {
            "status": "effortless",
            "feasibility_score": 100.0,
            "goal_budget": budget,
            "timeline": months,
            "required_monthly": round(required_monthly, 2),
            "monthly_deficit": 0.0,
            "baseline": baseline,
            "message": f"You have an average buffer of ${avg_buffer:,.2f}/mo. You can fund this ${budget:,.2f} goal without altering your spending!"
        }

    # Deficit calculations
    deficit = required_monthly - avg_buffer
    effort_ratio = deficit / avg_dynamic
    feasibility_score = max(0.0, min(100.0, (1 - effort_ratio) * 100))

    # Case B: Achievable (Feasibility >= 65%)
    if feasibility_score >= 65.0:
        ai_plan = generate_ai_tradeoffs(
            deficit=deficit,
            category_averages=category_monthly_avg,
            merchant_patterns=frequent_merchants,
            goal_price=budget,
            months=months,
            is_recalibrated=False
        )
        return {
            "status": "achievable",
            "feasibility_score": round(feasibility_score, 1),
            "goal_budget": budget,
            "timeline": months,
            "required_monthly": round(required_monthly, 2),
            "monthly_deficit": round(deficit, 2),
            "baseline": baseline,
            "tradeoffs": ai_plan
        }

    # Case C: Low Feasibility (< 65%) -> Recalibrate
    zero_sacrifice_budget = avg_buffer * months
    target_70_cut = 0.30 * avg_dynamic
    target_70_budget = (avg_buffer + target_70_cut) * months

    max_safe_cut = 0.35 * avg_dynamic
    max_safe_monthly = avg_buffer + max_safe_cut
    max_safe_budget = max_safe_monthly * months

    extended_months = math.ceil(budget / max_safe_monthly)

    ai_plan = generate_ai_tradeoffs(
        deficit=target_70_cut,
        category_averages=category_monthly_avg,
        merchant_patterns=frequent_merchants,
        goal_price=target_70_budget,
        months=months,
        is_recalibrated=True
    )

    return {
        "status": "low_feasibility",
        "feasibility_score": round(feasibility_score, 1),
        "goal_budget": budget,
        "timeline": months,
        "required_monthly": round(required_monthly, 2),
        "monthly_deficit": round(deficit, 2),
        "baseline": baseline,
        "recalibrated": {
            "zero_sacrifice": round(zero_sacrifice_budget, 2),
            "balanced_70": round(target_70_budget, 2),
            "max_safe": round(max_safe_budget, 2),
            "extended_months": extended_months
        },
        "tradeoffs": ai_plan
    }


# ==========================================
# 4. CLI Standalone Runner
# ==========================================
if __name__ == "__main__":
    # Runs ONLY when executing `python -m src.generate` directly
    test_budget = 25000.0
    test_months = 12

    print("=" * 55)
    print("HISTORICAL TYPICAL MONTH BASELINE")
    print("=" * 55)
    print(f"Avg Monthly Income:        ${avg_income:,.2f}")
    print(f"Avg Static Bills:          ${avg_static:,.2f}")
    print(f"Avg Dynamic Spending:      ${avg_dynamic:,.2f}")
    print(f"Avg 5% Sinking Fund Saved: ${avg_sinking_fund:,.2f}")
    print(f"Avg Remaining Buffer:      ${avg_buffer:,.2f}")
    print("=" * 55)
    print(f"\nEvaluating Goal: ${test_budget:,.2f} in {test_months} months...")

    result = evaluate_financial_plan(budget=test_budget, months=test_months)
    print("\nRESULT PAYLOAD:")
    print(json.dumps(result, indent=2))