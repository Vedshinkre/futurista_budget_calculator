from django.shortcuts import render
from .models import (
    Static_data,
    Dynamic_data,
    Salary_data,
    STATIC_CATEGORIES_CHOICES,
    DYNAMIC_CATEGORIES_CHOICES,
    INCOME_CATEGORIES_CHOICES,
    target_model
)
from src.generate import evaluate_financial_plan, get_live_financial_summary


def get_data(request):
    message = None

    if request.method == 'POST':
        # Identify which form the user submitted
        action = request.POST.get('form_action')

        if action == 'log_transaction':
            input_description = request.POST.get('description', '').strip()
            input_amount = request.POST.get('amount')
            input_date = request.POST.get('date')
            input_category = request.POST.get('category')
            input_month = request.POST.get('month')
            
            static_keys = [c[0] for c in STATIC_CATEGORIES_CHOICES]
            dynamic_keys = [c[0] for c in DYNAMIC_CATEGORIES_CHOICES]
            income_keys = [c[0] for c in INCOME_CATEGORIES_CHOICES]

            if input_description and input_amount and input_date and input_category:
                try:
                    amt = float(input_amount)
                    if input_category in static_keys:
                        Static_data.objects.create(description=input_description, amount=amt, date=input_date, category=input_category, month=input_month)
                    elif input_category in dynamic_keys:
                        Dynamic_data.objects.create(description=input_description, amount=amt, date=input_date, category=input_category, month=input_month)
                    elif input_category in income_keys:
                        Salary_data.objects.create(description=input_description, amount=amt, date=input_date, category=input_category, month=input_month)
                    message = "New transaction saved to database."
                except Exception as e:
                    message = f"Error saving transaction: {e}"
            # This drops down to return finance.html, keeping the user on the entry page.

        elif action == 'generate_plan':
            input_budget = request.POST.get('budget')
            input_timeline = request.POST.get('timeline_months') or "12"

            if input_budget:
                budget = float(input_budget)
                months = int(input_timeline)
                target_model.objects.create(budget=budget, timeline_months=months)
            
                plan_payload = evaluate_financial_plan(budget=budget, months=months)

                # This routes the user to the AI results page
                return render(request, 'finance/results.html', {
                    "result_data": plan_payload,
                    "baseline": plan_payload["baseline"],
                    "message": message
                })

    # Default GET rendering or post-transaction rendering
    current_metrics = get_live_financial_summary()
    context = {
        "message": message,
        "static_categories": STATIC_CATEGORIES_CHOICES,
        "dynamic_categories": DYNAMIC_CATEGORIES_CHOICES,
        "income_categories": INCOME_CATEGORIES_CHOICES,
        "baseline": {
            "income": round(current_metrics["avg_income"], 2),
            "static": round(current_metrics["avg_static"], 2),
            "dynamic": round(current_metrics["avg_dynamic"], 2),
            "sinking_fund": round(current_metrics["avg_sinking_fund"], 2),
            "buffer": round(current_metrics["avg_buffer"], 2)
        }
    }
    return render(request, 'finance/finance.html', context)