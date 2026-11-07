import json
import math
import os
import re
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI

load_dotenv()

BLABLADOR_API_KEY = os.getenv("BLABLADOR_API_KEY")
BLABLADOR_BASE_URL = "https://api.blablador.fz-juelich.de/v1"

# Ensure Django is initialized if run independently
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "hackathon_project.settings")
import django
try:
    django.setup()
except Exception:
    pass

from financial.models import Static_data, Dynamic_data, Salary_data


# ==========================================
# 1. Dynamic Data Fetcher (From SQLite Database)
# ==========================================
def get_live_financial_summary():
    """
    Pulls real-time data directly from Django models.
    Converts Decimal values to float and groups by Month.
    """
    dyn_records = list(Dynamic_data.objects.all().values("date", "description", "amount", "category", "month"))
    stat_records = list(Static_data.objects.all().values("date", "description", "amount", "category", "month"))
    inc_records = list(Salary_data.objects.all().values("date", "description", "amount", "category", "month"))

    dynamic_df = pd.DataFrame(dyn_records)
    static_df = pd.DataFrame(stat_records)
    income_df = pd.DataFrame(inc_records)

    # Standardize column types
    for df in [dynamic_df, static_df, income_df]:
        if not df.empty and "amount" in df.columns:
            df["amount"] = df["amount"].astype(float)

    # Monthly aggregations
    monthly_income = income_df.groupby("month")["amount"].sum() if not income_df.empty else pd.Series(dtype=float)
    monthly_static = static_df.groupby("month")["amount"].sum() if not static_df.empty else pd.Series(dtype=float)
    monthly_dynamic = dynamic_df.groupby("month")["amount"].sum() if not dynamic_df.empty else pd.Series(dtype=float)

    monthly_summary = pd.DataFrame({
        "income": monthly_income,
        "static": monthly_static,
        "dynamic": monthly_dynamic,
    }).fillna(0)

    # 5% Baseline Sinking Fund Target & Unallocated Buffer
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

    total_months = max(1, len(monthly_summary))

    category_monthly_avg = (
        (dynamic_df.groupby("category")["amount"].sum() / total_months)
        .round(2)
        .sort_values(ascending=False)
    ) if not dynamic_df.empty else pd.Series(dtype=float)

    frequent_merchants = (
        dynamic_df.groupby(["category", "description"])["amount"]
        .agg(total_spent="sum", frequency="count")
        .sort_values(by="frequency", ascending=False)
        .head(10)
        .reset_index()
    ) if not dynamic_df.empty else pd.DataFrame(columns=["category", "description", "total_spent", "frequency"])

    avg_income = float(monthly_summary["income"].mean()) if not monthly_summary.empty else 0.0
    avg_static = float(monthly_summary["static"].mean()) if not monthly_summary.empty else 0.0
    avg_dynamic = float(monthly_summary["dynamic"].mean()) if not monthly_summary.empty else 0.0
    avg_sinking_fund = float(monthly_summary["actual_sinking_fund"].mean()) if not monthly_summary.empty else 0.0
    avg_buffer = float(monthly_summary["remaining_funds"].mean()) if not monthly_summary.empty else 0.0

    return {
        "avg_income": avg_income,
        "avg_static": avg_static,
        "avg_dynamic": avg_dynamic,
        "avg_sinking_fund": avg_sinking_fund,
        "avg_buffer": avg_buffer,
        "category_monthly_avg": category_monthly_avg,
        "frequent_merchants": frequent_merchants,
    }


# ==========================================
# 2. AI Coaching Engine
# ==========================================
def generate_ai_tradeoffs(deficit, category_averages, merchant_patterns, goal_price=None, months=None, is_recalibrated=False):
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
        return default_fallback

    client = OpenAI(base_url=BLABLADOR_BASE_URL, api_key=BLABLADOR_API_KEY)

    context_note = (
        "Note: The user's original goal was over budget. This plan is calibrated to a realistic, "
        "sustainable 70% feasibility budget."
        if is_recalibrated
        else "This plan covers the calculated monthly gap."
    )

    system_prompt = (
        "You are an automated financial planning API engine. "
        "You must respond ONLY with a raw JSON object matching the requested schema. "
        "Do not include Markdown explanations outside the JSON. "
        "Never suggest cutting static bills (rent, utilities, insurance)."
    )

    merchants_repr = merchant_patterns[['category', 'description', 'total_spent', 'frequency']].to_string(index=False) if not merchant_patterns.empty else "No merchant data."

    user_prompt = f"""
GOAL CONTEXT:
Save ${deficit:.2f}/month to fund a {goal_text}. {context_note}

AVERAGE MONTHLY DISCRETIONARY SPENDING (BY CATEGORY):
{category_averages.to_string()}

TOP FREQUENT MERCHANTS & PATTERNS:
{merchants_repr}

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
# 3. Main Live Evaluation Function
# ==========================================
def evaluate_financial_plan(budget: float, months: int) -> dict:
    months = max(1, int(months))
    budget = float(budget)
    required_monthly = budget / months

    metrics = get_live_financial_summary()
    avg_buffer = metrics["avg_buffer"]
    avg_dynamic = metrics["avg_dynamic"]

    baseline = {
        "income": round(metrics["avg_income"], 2),
        "static": round(metrics["avg_static"], 2),
        "dynamic": round(metrics["avg_dynamic"], 2),
        "sinking_fund": round(metrics["avg_sinking_fund"], 2),
        "buffer": round(avg_buffer, 2)
    }

    # ==========================================
    # CASE 1: EFFORTLESS -> ZERO AI CALLS
    # ==========================================
    if avg_buffer >= required_monthly:
        return {
            "status": "effortless",
            "feasibility_score": 100.0,
            "goal_budget": budget,
            "timeline": months,
            "required_monthly": round(required_monthly, 2),
            "monthly_deficit": 0.0,
            "baseline": baseline,
            "tradeoffs": None,  # AI is never invoked
            "message": f"Your current cash buffer (${avg_buffer:,.2f}/mo) fully covers this goal. You do not need to cut any expenses."
        }

    deficit = required_monthly - avg_buffer
    effort_ratio = deficit / avg_dynamic if avg_dynamic > 0 else 1.0
    feasibility_score = max(0.0, min(100.0, (1 - effort_ratio) * 100))

    # ==========================================
    # CASE 2: ACHIEVABLE -> TARGETED AI CALL
    # ==========================================
    if feasibility_score >= 65.0:
        ai_plan = generate_ai_tradeoffs(
            deficit=deficit,
            category_averages=metrics["category_monthly_avg"],
            merchant_patterns=metrics["frequent_merchants"],
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

    # ==========================================
    # CASE 3: LOW FEASIBILITY -> RECALIBRATED AI CALL
    # ==========================================
    zero_sacrifice_budget = avg_buffer * months
    target_70_cut = 0.30 * avg_dynamic
    target_70_budget = (avg_buffer + target_70_cut) * months

    max_safe_cut = 0.35 * avg_dynamic
    max_safe_monthly = avg_buffer + max_safe_cut
    max_safe_budget = max_safe_monthly * months

    extended_months = math.ceil(budget / max_safe_monthly) if max_safe_monthly > 0 else 0

    ai_plan = generate_ai_tradeoffs(
        deficit=target_70_cut,
        category_averages=metrics["category_monthly_avg"],
        merchant_patterns=metrics["frequent_merchants"],
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

if __name__ == "__main__":
    import json
   
    print("Evaluating financial plan...")
    result = evaluate_financial_plan(budget=35000.0, months=12)
    
   
    print(json.dumps(result, indent=4))