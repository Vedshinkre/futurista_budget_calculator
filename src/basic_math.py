import os
import re
from pathlib import Path
import pandas as pd
from transformers import pipeline

# Silence Windows symlink cache warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

LOCAL_AI_PIPELINE = None


# ==========================================
# 0. Helper Functions (Deterministic Math & Rules)
# ==========================================
def freq_words(monthly_count):
    """Converts a monthly transaction count into natural descriptive language."""
    if monthly_count >= 10:
        return "multiple times a week"
    if monthly_count >= 4:
        return "about weekly"
    if monthly_count >= 1.5:
        return "a few times a month"
    return "occasionally"


def build_targets(category_averages, frequent_merchants, deficit, total_months, top_n=3):
    """
    Selects top dynamic spending categories, calculates safe cuts rounded to $5 in Python,
    and attaches habitual merchant context without revealing raw frequency integers.
    """
    top_series = category_averages.head(top_n)
    total_top = max(top_series.sum(), 1.0)

    # Pre-map primary merchant per category to avoid raw frequency leakage
    merchant_map = {}
    for _, row in frequent_merchants.iterrows():
        cat = row["Category"]
        if cat not in merchant_map:
            rate = row["frequency"] / max(total_months, 1)
            merchant_map[cat] = f"frequent visits to {row['Description']} ({freq_words(rate)})"

    targets = []
    running_cut = 0
    num_categories = len(top_series)

    # Proportionally allocate cuts, rounded to nearest $5
    for i, (cat, amt) in enumerate(top_series.items()):
        if i == num_categories - 1:
            # Reconcile remainder on the final category so total cut matches deficit
            cut = max(5, round(deficit - running_cut))
        else:
            cut = int(round((deficit * amt / total_top) / 5.0) * 5)
            cut = max(5, cut)
            running_cut += cut

        habit_desc = merchant_map.get(cat, "regular discretionary purchases")
        targets.append({
            "category": cat,
            "cut": cut,
            "habit": habit_desc
        })

    return targets


def fallback_template(targets, deficit, goal_info):
    """Guaranteed deterministic output if the 0.5B model breaks formatting rules."""
    step1 = "\n".join([f"- **{t['category']}**: Cut ${t['cut']}/month" for t in targets])
    step2 = "\n".join([
        f"- **{t['category']}**: Reduce {t['habit']} by preparing meals or shopping essentials at home to save ~${t['cut']}/month."
        for t in targets
    ])
    return (
        f"### Step 1: Where to Trim\n{step1}\n\n"
        f"### Step 2: Habit Changes\n{step2}\n\n"
        f"### Step 3: Goal Impact\n"
        f"These small routine tweaks free up ${deficit:.2f}/month without drastic lifestyle changes. "
        f"Sticking with this plan keeps you fully funded for {goal_info}."
    )


# ==========================================
# 1. Local AI Coaching Engine
# ==========================================
def generate_ai_tradeoffs(deficit, category_averages, merchant_patterns, total_months, goal_price=None, months=None):
    """
    Passes pre-computed targets into Qwen2.5-0.5B using a one-shot chat prompt.
    Validates output safety and falls back deterministically if hallucinations occur.
    """
    global LOCAL_AI_PIPELINE

    if LOCAL_AI_PIPELINE is None:
        print("\nLoading local Qwen2.5-0.5B model...")
        LOCAL_AI_PIPELINE = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            dtype="auto",
            device_map="auto"
        )

    goal_info = f"${goal_price:,.2f} trip in {months} months" if goal_price and months else "your savings goal"
    targets = build_targets(category_averages, merchant_patterns, deficit, total_months, top_n=3)

    target_lines = "\n".join([
        f"- {t['category']}: cut ${t['cut']}/month (current habit: {t['habit']})"
        for t in targets
    ])

    system_prompt = (
        "You are a friendly, practical budgeting coach.\n"
        "STRICT RULES:\n"
        "- Use ONLY the exact dollar amounts provided. Never compute or invent any dollar amount.\n"
        "- Never write equations, multiplication signs (x or *), addition signs (+), or equals signs (=).\n"
        "- Never mention rent, utilities, insurance, or static bills.\n"
        "- Give one short, concrete habit change per category (under 25 words each)."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Goal: Save $150/month for vacation.\n"
                "Targets:\n"
                "- Restaurants: cut $90/month (current habit: takeout about weekly)\n"
                "- Coffee Shops: cut $60/month (current habit: frequent visits to Starbucks (multiple times a week))\n"
                "Format: Step 1, Step 2, Step 3."
            )
        },
        {
            "role": "assistant",
            "content": (
                "### Step 1: Where to Trim\n"
                "- Restaurants: cut $90/month\n"
                "- Coffee Shops: cut $60/month\n\n"
                "### Step 2: Habit Changes\n"
                "- Restaurants: Swap two monthly restaurant dinners with home-cooked dinners.\n"
                "- Coffee Shops: Make morning coffee at home on weekdays instead of buying daily drinks.\n\n"
                "### Step 3: Goal Impact\n"
                "Cutting these two habits safely frees up the $150/month needed. "
                "You will stay right on track for your vacation goal."
            )
        },
        {
            "role": "user",
            "content": (
                f"Goal: Fund {goal_info}. We need to free up ${deficit:.2f}/month from dynamic spending.\n"
                f"Here are the categories to trim with monthly cuts pre-decided in Python:\n"
                f"{target_lines}\n\n"
                "Write your answer in exactly this format:\n"
                "### Step 1: Where to Trim\n"
                "List the three categories above, each with its cut amount, one per line.\n\n"
                "### Step 2: Habit Changes\n"
                "For each category, provide one practical lifestyle habit change to achieve that cut.\n\n"
                "### Step 3: Goal Impact\n"
                f"Two encouraging sentences. Mention only the ${deficit:.2f}/month figure."
            )
        }
    ]

    output = LOCAL_AI_PIPELINE(
        messages,
        max_new_tokens=350,
        do_sample=False,
        repetition_penalty=1.1
    )

    raw_text = output[0]["generated_text"]
    response_text = raw_text[-1]["content"] if isinstance(raw_text, list) else raw_text

    # Whole-word regex checks to prevent false positives on words like "current"
    has_forbidden_words = bool(re.search(r"\b(rent|utility|utilities|insurance)\b", response_text, re.IGNORECASE))
    has_math_equation = bool(re.search(r"(\d+\s*[\*x×+=]\s*\d+|[=])", response_text))
    has_headers = ("### Step 1" in response_text) and ("### Step 2" in response_text)

    if has_forbidden_words or has_math_equation or not has_headers:
        return fallback_template(targets, deficit, goal_info)

    return response_text


# ==========================================
# 2. Path Management & Data Loading
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
# 3. Monthly Cash Flow & Sinking Fund Engine
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
# 4. Dynamic Aggregations for LLM Context
# ==========================================
total_months = len(monthly_summary)

category_monthly_avg = (
    (dynamic_df.groupby("Category")["Amount"].sum() / total_months)
    .round(2)
    .sort_values(ascending=False)
)

# Top 6 frequent merchants for context
frequent_merchants = (
    dynamic_df.groupby(["Category", "Description"])["Amount"]
    .agg(total_spent="sum", frequency="count")
    .sort_values(by="frequency", ascending=False)
    .head(6)
    .reset_index()
)

# ==========================================
# 5. User Goal & Historical Baselines
# ==========================================
trip_target_price = 18000.0
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
# 6. Feasibility Evaluation & LLM Call
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

    max_safe_monthly = avg_buffer + (0.35 * avg_dynamic)
    max_safe_trip_price = max_safe_monthly * months_until_trip

    print(f"\nFeasibility Score: {feasibility_score:.1f}%")
    print(f"Monthly Shortfall (Deficit): ${deficit:.2f}/month")

    if effort_ratio <= 0.35:
        print(
            f"Status: Achievable. Need to trim ${deficit:.2f}/mo from dynamic spending."
        )
        print("\n" + "=" * 55)
        print("GENERATING LOCAL AI TRADE-OFF RECOMMENDATIONS...")
        print("=" * 55)

        ai_recommendations = generate_ai_tradeoffs(
            deficit=deficit,
            category_averages=category_monthly_avg,
            merchant_patterns=frequent_merchants,
            total_months=total_months,
            goal_price=trip_target_price,
            months=months_until_trip
        )
        print(ai_recommendations)

    else:
        print(
            "Status: High Risk (Requires cutting > 35 percent of discretionary lifestyle)."
        )
        print(
            f"Suggested Alternative: Max safe trip budget is ${max_safe_trip_price:.2f}"
        )