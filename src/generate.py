import calendar
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

# Ensure Django environment is configured when run standalone
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "hackathon_project.settings")
import django
try:
    django.setup()
except Exception:
    pass

from financial.models import Static_data, Dynamic_data, Salary_data


# =====================================================================
# 1. Database Summaries & Multi-Year Aggregations
# =====================================================================
def get_live_financial_summary() -> dict:
    """
    Pulls live data directly from SQLite via Django ORM.
    Groups by exact Year-Month (YYYY-MM) derived from DateField
    to avoid multi-year month collisions (e.g., Oct 2023 vs Oct 2024).
    """
    dyn_records = list(Dynamic_data.objects.all().values("date", "description", "amount", "category", "month"))
    stat_records = list(Static_data.objects.all().values("date", "description", "amount", "category", "month"))
    inc_records = list(Salary_data.objects.all().values("date", "description", "amount", "category", "month"))

    dyn_df = pd.DataFrame(dyn_records)
    stat_df = pd.DataFrame(stat_records)
    inc_df = pd.DataFrame(inc_records)

    # Standardize amounts and extract calendar periods (YYYY-MM)
    for df in [dyn_df, stat_df, inc_df]:
        if not df.empty:
            if "amount" in df.columns:
                df["amount"] = df["amount"].astype(float)
            if "date" in df.columns:
                df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
                df["period"] = df["date_dt"].dt.strftime("%Y-%m").fillna(df["month"])
            else:
                df["period"] = df["month"]

    monthly_income = inc_df.groupby("period")["amount"].sum() if not inc_df.empty else pd.Series(dtype=float)
    monthly_static = stat_df.groupby("period")["amount"].sum() if not stat_df.empty else pd.Series(dtype=float)
    monthly_dynamic = dyn_df.groupby("period")["amount"].sum() if not dyn_df.empty else pd.Series(dtype=float)

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
        (dyn_df.groupby("category")["amount"].sum() / total_months)
        .round(2)
        .sort_values(ascending=False)
    ) if not dyn_df.empty else pd.Series(dtype=float)

    frequent_merchants = (
        dyn_df.groupby(["category", "description"])["amount"]
        .agg(total_spent="sum", frequency="count")
        .sort_values(by="frequency", ascending=False)
        .head(10)
        .reset_index()
    ) if not dyn_df.empty else pd.DataFrame(columns=["category", "description", "total_spent", "frequency"])

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


def get_available_periods() -> list[dict]:
    """
    Returns all unique Year-Month combinations available across records
    sorted in descending order for frontend dropdown selectors.
    """
    records = Dynamic_data.objects.dates("date", "month", order="DESC")
    return [
        {
            "year": d.year,
            "month": d.month,
            "label": f"{calendar.month_name[d.month]} {d.year}",
            "value": f"{d.year}-{d.month:02d}"
        }
        for d in records
    ]


# =====================================================================
# 2. AI Prompt Engine A: Goal-Oriented Behavioral Trade-Offs
# =====================================================================
def generate_ai_tradeoffs(deficit: float, category_averages: pd.Series, merchant_patterns: pd.DataFrame,
                           goal_price: float = None, months: int = None, is_recalibrated: bool = False) -> dict:
    goal_text = f"${goal_price:,.2f} goal in {months} months" if goal_price and months else "savings target"

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
        print(f"[Warning] Blablador goal trade-off generation failed ({e}). Returning fallback.")
        return default_fallback


# =====================================================================
# 3. AI Prompt Engine B: General Monthly Financial Health Audit
# =====================================================================
def generate_monthly_ai_insights(month_label: str, income: float, static: float, dynamic: float,
                                 category_totals: dict, baseline_dynamic: float) -> dict:
    """
    Audits a single month's finances against historical averages to detect
    lifestyle creep and unallocated cash opportunities.
    """
    net_savings = income - static - dynamic

    top_cat = max(category_totals, key=category_totals.get) if category_totals else "Discretionary"
    top_cat_amt = category_totals.get(top_cat, 0.0)

    default_fallback = {
        "health_grade": "B" if net_savings >= 0 else "D",
        "spending_diagnosis": f"In {month_label}, you earned ${income:,.2f} and spent ${static + dynamic:,.2f}, leaving ${net_savings:,.2f} in net savings.",
        "flagged_categories": [
            {
                "category": top_cat,
                "comment": f"Highest discretionary spend this month at ${top_cat_amt:,.2f}.",
                "tip": "Set a weekly discretionary envelope to keep non-essential purchases controlled."
            }
        ],
        "wealth_building_tip": "Route remaining unallocated surplus directly into a dedicated high-yield sinking fund."
    }

    if not BLABLADOR_API_KEY:
        return default_fallback

    client = OpenAI(base_url=BLABLADOR_BASE_URL, api_key=BLABLADOR_API_KEY)

    system_prompt = (
        "You are an expert personal finance auditor. Analyze a user's monthly spending, "
        "compare it to their baseline, identify lifestyle leaks, and provide actionable tips. "
        "Return ONLY a raw JSON object matching the requested schema."
    )

    user_prompt = f"""
MONTH UNDER REVIEW: {month_label}
- Net Income: ${income:,.2f}
- Static Bills: ${static:,.2f}
- Discretionary Spend: ${dynamic:,.2f} (Historical Baseline: ${baseline_dynamic:,.2f})
- Net Cash Saved: ${net_savings:,.2f}

MONTHLY DISCRETIONARY BREAKDOWN:
{json.dumps(category_totals, indent=2)}

Generate a JSON response strictly matching this structure:
{{
  "health_grade": "<A, B, C, or D based on savings rate>",
  "spending_diagnosis": "<2-3 sentences evaluating how this month went compared to baseline>",
  "flagged_categories": [
    {{
      "category": "<Category Name with highest or unusual spend>",
      "comment": "<Why this stands out>",
      "tip": "<Actionable way to trim it next month>"
    }}
  ],
  "wealth_building_tip": "<1 concrete suggestion on what to do with leftover cash>"
}}
"""

    try:
        response = client.chat.completions.create(
            model="alias-fast",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )

        content = response.choices[0].message.content or ""
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))

        return json.loads(content)

    except Exception as e:
        print(f"[Warning] Blablador monthly health audit failed ({e}). Returning fallback.")
        return default_fallback


# =====================================================================
# 4. Monthly Finance Analysis Entry Point (Month + Year)
# =====================================================================
def analyze_monthly_finances(year: int, month: int) -> dict:
    """
    Pulls transactions for an exact Year and Month (e.g., year=2024, month=10).
    Aggregates spend, converts Decimals to float, and runs the health audit.
    """
    month_label = f"{calendar.month_name[month]} {year}"

    dyn_records = list(Dynamic_data.objects.filter(date__year=year, date__month=month).values())
    stat_records = list(Static_data.objects.filter(date__year=year, date__month=month).values())
    inc_records = list(Salary_data.objects.filter(date__year=year, date__month=month).values())

    dyn_df = pd.DataFrame(dyn_records)
    stat_df = pd.DataFrame(stat_records)
    inc_df = pd.DataFrame(inc_records)

    # Cast Decimal columns to float
    for df in [dyn_df, stat_df, inc_df]:
        if not df.empty and "amount" in df.columns:
            df["amount"] = df["amount"].astype(float)

    total_income = float(inc_df["amount"].sum()) if not inc_df.empty else 0.0
    total_static = float(stat_df["amount"].sum()) if not stat_df.empty else 0.0
    total_dynamic = float(dyn_df["amount"].sum()) if not dyn_df.empty else 0.0

    total_spent = total_static + total_dynamic
    net_savings = total_income - total_spent
    savings_rate = (net_savings / total_income * 100) if total_income > 0 else 0.0

    category_totals = (
        dyn_df.groupby("category")["amount"].sum().round(2).to_dict()
        if not dyn_df.empty else {}
    )

    baseline_metrics = get_live_financial_summary()
    baseline_dynamic = baseline_metrics["avg_dynamic"]

    ai_audit = generate_monthly_ai_insights(
        month_label=month_label,
        income=total_income,
        static=total_static,
        dynamic=total_dynamic,
        category_totals=category_totals,
        baseline_dynamic=baseline_dynamic
    )

    return {
        "period": month_label,
        "year": year,
        "month": month,
        "income": round(total_income, 2),
        "static_burn": round(total_static, 2),
        "dynamic_spend": round(total_dynamic, 2),
        "net_savings": round(net_savings, 2),
        "savings_rate": round(savings_rate, 1),
        "categories": category_totals,
        "ai_audit": ai_audit
    }


# =====================================================================
# 5. Goal Planning Evaluation Entry Point
# =====================================================================
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

    # Case 1: Effortless -> Bypasses AI completely
    if avg_buffer >= required_monthly:
        return {
            "status": "effortless",
            "feasibility_score": 100.0,
            "goal_budget": budget,
            "timeline": months,
            "required_monthly": round(required_monthly, 2),
            "monthly_deficit": 0.0,
            "baseline": baseline,
            "tradeoffs": None,
            "message": f"Your monthly buffer (${avg_buffer:,.2f}) completely covers this goal. No cuts required."
        }

    deficit = required_monthly - avg_buffer
    effort_ratio = deficit / avg_dynamic if avg_dynamic > 0 else 1.0
    feasibility_score = max(0.0, min(100.0, (1 - effort_ratio) * 100))

    # Case 2: Achievable -> Targeted AI Trade-Off Plan
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

    # Case 3: Low Feasibility -> Recalibrated Budget Tiers & AI Plan
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


# =====================================================================
# Standalone CLI Test Verification
# =====================================================================
if __name__ == "__main__":
    print("==================================================")
    print("1. Available Tracked Periods in Database:")
    print("==================================================")
    periods = get_available_periods()
    for p in periods[:6]:
        print(f"  • {p['label']} ({p['value']})")

    print("\n==================================================")
    print("2. Testing Savings Goal Plan ($18,000 / 12 mo):")
    print("==================================================")
    goal_test = evaluate_financial_plan(budget=18000.0, months=12)
    print(f"Status: {goal_test['status']} | Feasibility: {goal_test['feasibility_score']}%")
    if goal_test.get("tradeoffs"):
        print(f"AI Cut Needed: ${goal_test['tradeoffs']['deficit']}/mo")
        print(f"Categories Flagged: {[c['category'] for c in goal_test['tradeoffs']['top_categories']]}")

    print("\n==================================================")
    print("3. Testing Monthly Health Audit (Most Recent Month):")
    print("==================================================")
    if periods:
        sample = periods[0]
        audit_test = analyze_monthly_finances(year=sample["year"], month=sample["month"])
        print(f"Period: {audit_test['period']}")
        print(f"Net Income: ${audit_test['income']} | Dynamic Spent: ${audit_test['dynamic_spend']}")
        print(f"AI Grade: {audit_test['ai_audit']['health_grade']}")
        print(f"Diagnosis: {audit_test['ai_audit']['spending_diagnosis']}")
    else:
        print("No transactions currently found in SQLite database. Seed the database first.")