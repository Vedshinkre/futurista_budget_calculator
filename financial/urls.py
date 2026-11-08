from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'), 
    path('finance/', views.get_data, name='finance_form'), 
    path('transaction/', views.log_transaction_view, name='log_transaction'), 
    path('input/', views.monthly_report_view, name='report_input'), 
]