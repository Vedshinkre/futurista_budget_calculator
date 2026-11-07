import json
import math
import os
import re
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI

# Load environment variables (.env file in root)
load_dotenv()

BLABLADOR_API_KEY = os.getenv("BLABLADOR_API_KEY")
BLABLADOR_BASE_URL = "https://api.blablador.fz-juelich.de/v1"


# ==========================================
# 0. Structured AI Coaching Engine (JSON Mode)
# ==========================================
def generate_ai_tradeoffs(deficit, category_averages, merchant_patterns, goal_price=None, months=None, is_recalibrated=False):
    """
    Calls Blablador's alias-fast model to generate trade-offs as structured JSON.
    Uses regex extraction to avoid vLLM response_format bugs.
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
        "Do not include any Markdown text, explanations, or code blocks outside the JSON. "
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
            print("[Notice] Model returned empty content. Using fallback.")
            return default_fallback

        # Extract the JSON substring safely (handles markdown blocks if model adds them)
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))

        return json.loads(content)

    except Exception as e:
        print(f"[Warning] Blablador JSON generation failed ({e}). Returning fallback schema.")
        return default_fallback

# ==========================================
# 1. Path Management & Data Loading
# ==========================================
current_code_path = Path(__file__).resolve().parent
project_path = current_code_path.parent

data_path = project_path / "data"
dynamic_path = data_path / "table_dynamic.csv"
static_path = data_path / "table_static.csv"
income_path = data_path / "table_paycheck.csv"

for p in [data_path, dynamic_path, static_path, income_path]:
    if not p.exists():
        raise FileNotFoundError(f"Path {p} not found.")

dynamic_df = pd.read_csv(dynamic_path)
static_df = pd.read_csv(static_path)
income_df = pd.read_csv(income_path)

# ==========================================
# 2. Monthly Cash Flow & Sinking Fund Engine
# ==========================================
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

# 5% Baseline Sinking Fund Target
monthly_summary["target_sinking_fund"] = monthly_summary["income"] * 0.05

# Raw surplus after all living costs
monthly_summary["raw_surplus"] = monthly_summary["income"] - (
    monthly_summary["static"] + monthly_summary["dynamic"]
)

# Realistic deposit: cannot save negative cash, capped at target
monthly_summary["actual_sinking_fund"] = (
    monthly_summary["raw_surplus"]
    .clip(lower=0)
    .combine(monthly_summary["target_sinking_fund"], min)
)

monthly_summary["savings_shortfall"] = (
    monthly_summary["target_sinking_fund"]
    - monthly_summary["actual_sinking_fund"]
)

# Remaining unallocated cash buffer
monthly_summary["remaining_funds"] = monthly_summary["income"] - (
    monthly_summary["static"]
    + monthly_summary["dynamic"]
    + monthly_summary["actual_sinking_fund"]
)

# ==========================================
# 3. Dynamic Aggregations for LLM Context
# ==========================================
total_months = len(monthly_summary)

category_monthly_avg = (
    (dynamic_df.groupby("Category")["Amount"].sum() / total_months)
    .round(2)
    .sort_values(ascending=False)
)

# Top 10 frequent merchants for behavioral pattern recognition
frequent_merchants = (
    dynamic_df.groupby(["Category", "Description"])["Amount"]
    .agg(total_spent="sum", frequency="count")
    .sort_values(by="frequency", ascending=False)
    .head(10)
    .reset_index()
)

# ==========================================
# 4. User Goal & Historical Baselines
# ==========================================
trip_target_price = 25000.0
months_until_trip = 12
required_monthly = trip_target_price / months_until_trip

avg_income = monthly_summary["income"].mean()
avg_static = monthly_summary["static"].mean()
avg_dynamic = monthly_summary["dynamic"].mean()
avg_sinking_fund = monthly_summary["actual_sinking_fund"].mean()
avg_buffer = monthly_summary["remaining_funds"].mean()

print("=" * 55)
print("HISTORICAL TYPICAL MONTH BASELINE")
print("=" * 55)
print(f"Avg Monthly Income:        ${avg_income:,.2f}")
print(f"Avg Static Bills:          ${avg_static:,.2f}")
print(f"Avg Dynamic Spending:      ${avg_dynamic:,.2f}")
print(f"Avg 5% Sinking Fund Saved: ${avg_sinking_fund:,.2f}")
print(f"Avg Remaining Buffer:      ${avg_buffer:,.2f}")
print("=" * 55)

# ==========================================
# 5. Feasibility Evaluation & Adaptive Recalibration
# ==========================================
print(f"\nGOAL: ${trip_target_price:.2f} trip in {months_until_trip} months")
print(f"Required Monthly Allocation: ${required_monthly:.2f}/month")

if avg_buffer >= required_monthly:
    feasibility_score = 100.0
    print(f"\nFeasibility Score: {feasibility_score:.0f}% (Effortless)")
    print(
        f"You have an average buffer of ${avg_buffer:.2f}/mo. "
        "You can easily afford this without altering dynamic expenses!"
    )
else:
    deficit = required_monthly - avg_buffer
    effort_ratio = deficit / avg_dynamic
    feasibility_score = max(0.0, min(100.0, (1 - effort_ratio) * 100))

    print(f"\nFeasibility Score: {feasibility_score:.1f}%")
    print(f"Monthly Shortfall (Deficit): ${deficit:.2f}/month")

    # Threshold: Effort ratio <= 0.35 means Feasibility >= 65%
    if feasibility_score >= 65.0:
        print(f"Status: Achievable. Need to trim ${deficit:.2f}/mo from dynamic spending.")
        print("\n" + "=" * 55)
        print("GENERATING STRUCTURED AI TRADE-OFF JSON...")
        print("=" * 55)

        ai_recommendations = generate_ai_tradeoffs(
            deficit=deficit,
            category_averages=category_monthly_avg,
            merchant_patterns=frequent_merchants,
            goal_price=trip_target_price,
            months=months_until_trip,
            is_recalibrated=False
        )

        response_payload = {
            "status": "achievable",
            "feasibility_score": round(feasibility_score, 1),
            "original_goal": {
                "budget": trip_target_price,
                "months": months_until_trip,
                "monthly_required": round(required_monthly, 2),
                "monthly_deficit": round(deficit, 2)
            },
            "tradeoffs": ai_recommendations
        }
        print(json.dumps(response_payload, indent=2))

    else:
        # LOW FEASIBILITY / HIGH RISK (< 65% Score): Generate Adaptive Options
        print("Status: Low Feasibility / High Risk (Requires cutting > 35% of lifestyle).")

        # 1. Effortless / Zero-Sacrifice Budget (Deficit = 0, 100% Feasibility)
        zero_sacrifice_monthly = max(0.0, avg_buffer)
        zero_sacrifice_budget = zero_sacrifice_monthly * months_until_trip

        # 2. Recommended Sustainable Target Budget (70% Feasibility -> 30% Dynamic Cut)
        target_70_cut = 0.30 * avg_dynamic
        target_70_monthly = avg_buffer + target_70_cut
        target_70_budget = target_70_monthly * months_until_trip

        # 3. Maximum Safe Ceiling (65% Feasibility -> 35% Dynamic Cut Max Safe Limit)
        max_safe_cut = 0.35 * avg_dynamic
        max_safe_monthly = avg_buffer + max_safe_cut
        max_safe_budget = max_safe_monthly * months_until_trip

        # 4. Timeline Extension to afford the original $18,000 without exceeding safe cuts
        extended_months_needed = math.ceil(trip_target_price / max_safe_monthly)

        print("\n" + "=" * 55)
        print("RECALIBRATING BUDGET RECOMMENDATIONS...")
        print("=" * 55)
        print(f"• Zero-Sacrifice Budget (0% cuts, Deficit $0): ${zero_sacrifice_budget:,.2f}")
        print(f"• Recommended 70% Feasible Budget:           ${target_70_budget:,.2f}")
        print(f"• Max Safe Ceiling Budget:                   ${max_safe_budget:,.2f}")
        print(f"• Or extend timeline to {extended_months_needed} months to hit original ${trip_target_price:,.2f}")

        # Generate habit tweaks scaled to the realistic 70% budget target
        ai_recommendations = generate_ai_tradeoffs(
            deficit=target_70_cut,
            category_averages=category_monthly_avg,
            merchant_patterns=frequent_merchants,
            goal_price=target_70_budget,
            months=months_until_trip,
            is_recalibrated=True
        )

        response_payload = {
            "status": "low_feasibility_recalibrated",
            "original_feasibility_score": round(feasibility_score, 1),
            "original_goal": {
                "budget": trip_target_price,
                "months": months_until_trip,
                "monthly_required": round(required_monthly, 2),
                "monthly_deficit": round(deficit, 2)
            },
            "recalibrated_options": {
                "zero_sacrifice_budget": {
                    "budget": round(zero_sacrifice_budget, 2),
                    "monthly_allocation": round(zero_sacrifice_monthly, 2),
                    "monthly_lifestyle_cut": 0.0,
                    "feasibility": 100.0,
                    "description": "Zero lifestyle changes. 100% funded by existing cash buffer."
                },
                "recommended_balanced_budget": {
                    "budget": round(target_70_budget, 2),
                    "monthly_allocation": round(target_70_monthly, 2),
                    "monthly_lifestyle_cut": round(target_70_cut, 2),
                    "feasibility": 70.0,
                    "description": "Sustainable 30% trim on discretionary spending."
                },
                "max_safe_budget": {
                    "budget": round(max_safe_budget, 2),
                    "monthly_allocation": round(max_safe_monthly, 2),
                    "monthly_lifestyle_cut": round(max_safe_cut, 2),
                    "feasibility": 65.0,
                    "description": "Maximum safe ceiling before lifestyle strain."
                },
                "timeline_extension_option": {
                    "target_budget": trip_target_price,
                    "required_months": extended_months_needed,
                    "monthly_allocation": round(max_safe_monthly, 2),
                    "description": f"Keep the ${trip_target_price:,.0f} goal by extending saving duration to {extended_months_needed} months."
                }
            },
            "tradeoffs_for_recommended_budget": ai_recommendations
        }

        print("\n" + "=" * 55)
        print("STRUCTURED FRONTEND PAYLOAD:")
        print("=" * 55)
        print(json.dumps(response_payload, indent=2))