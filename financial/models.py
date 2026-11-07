from django.db import models
STATIC_CATEGORIES_CHOICES = [
    ('Mortgage & Rent', 'Mortgage & Rent'),
    ('Utilities', 'Utilities'),
    ('Mobile Phone', 'Mobile Phone'),
    ('Internet', 'Internet'),
    ('Auto Insurance', 'Auto Insurance'),
    ('Music', 'Music'),
    ('Movies & Dvds', 'Movies & DVDs'), 
    ('Television', 'Television'),
]

# Create your models here.
class Static_data(models.Model):
    description = models.CharField(max_length=15)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    date = models.DateField()
    category = models.CharField(max_length=22,choices=STATIC_CATEGORIES_CHOICES)
    month = models.CharField(max_length=15)

    def __str__(self):
        return self.name
DYNAMIC_CATEGORIES_CHOICES = [
    ('Groceries', 'Groceries'),
    ('Restaurants', 'Restaurants'),
    ('Shopping', 'Shopping'),
    ('Gas & Fuel', 'Gas & Fuel'),
    ('Home Improvement', 'Home Improvement'),
    ('Coffee Shops', 'Coffee Shops'),
    ('Alcohol & Bars', 'Alcohol & Bars'),
    ('Fast Food', 'Fast Food'),
    ('Haircut', 'Haircut'),
    ('Electronics & Software', 'Electronics & Software'),
    ('Food & Dining', 'Food & Dining'),
    ('Entertainment', 'Entertainment')
]
class Dynamic_data(models.Model):
    description = models.CharField(max_length=15)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    date = models.DateField()
    category = models.CharField(max_length=30,choices=DYNAMIC_CATEGORIES_CHOICES)
    month = models.CharField(max_length=15)

    def __str__(self):
        return self.description
INCOME_CATEGORIES_CHOICES = [
    ('Income', 'Income')
]

class Salary_data(models.Model):
    description = models.CharField(max_length=15)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    date = models.DateField()
    category = models.CharField(max_length=30, choices=INCOME_CATEGORIES_CHOICES)
    month = models.CharField(max_length=15)

    def __str__(self):
        return self.description
class target_model(models.Model):
    budget = models.DecimalField(max_digits=10, decimal_places=2)
    month = models.IntegerField()
