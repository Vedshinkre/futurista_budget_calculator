from django.shortcuts import render, HttpResponse
from .models import Static_data,Dynamic_data,Salary_data,STATIC_CATEGORIES_CHOICES,DYNAMIC_CATEGORIES_CHOICES,INCOME_CATEGORIES_CHOICES,target_model


def get_data(request):
    if request.method == 'POST':
        input_description = request.POST.get('description')
        input_amount = request.POST.get('amount')
        input_date = request.POST.get('date')
        input_category = request.POST.get('category')
        input_month = request.POST.get('month')
        input_budget = request.POST.get('budget')
        input_time  = request.POST.get('month')
    
        static_keys = [choice[0] for choice in STATIC_CATEGORIES_CHOICES]
        dynamic_keys = [choice[0] for choice in DYNAMIC_CATEGORIES_CHOICES]
        income_keys = [choice[0] for choice in INCOME_CATEGORIES_CHOICES]
        rflg = False
        
        if input_category in static_keys:
            data = Static_data(description=input_description, amount=input_amount, date=input_date, category=input_category, month=input_month)
            rflg = True
            data.save()

        elif input_category in dynamic_keys:
            data = Dynamic_data(description=input_description, amount=input_amount, date=input_date, category=input_category, month=input_month)
            rflg = True
            data.save()

        elif input_category in income_keys:
            data = Salary_data(description=input_description, amount=input_amount, date=input_date, category=input_category, month=input_month)
            rflg = True
            data.save()
            
        target_model_instance = target_model(budget=input_budget, month=input_time)
        target_model_instance.save()
        
        if rflg:
            message = "Data saved successfully."
        else:
            message = "Error in saving!! Oops Please try again."
            
        return render(request, 'finance/finance.html', {"message": message})


    return render(request, 'finance/finance.html')