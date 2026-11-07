import os
import django
import pandas as pd
from pathlib import Path

# Setup Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "hackathon_project.settings")
django.setup()

from financial.models import Static_data, Dynamic_data, Salary_data

base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"

dynamic_csv = data_dir / "table_dynamic.csv"
static_csv = data_dir / "table_static.csv"
paycheck_csv = data_dir / "table_paycheck.csv"

def seed_database():
    print("Clearing old records from database...")
    Dynamic_data.objects.all().delete()
    Static_data.objects.all().delete()
    Salary_data.objects.all().delete()

    # 1. Seed Dynamic Spending
    if dynamic_csv.exists():
        df_dyn = pd.read_csv(dynamic_csv)
        records = [
            Dynamic_data(
                date=row["Date"],
                description=str(row["Description"]),
                amount=float(row["Amount"]),
                category=str(row["Category"]),
                month=str(row["Month"])
            )
            for _, row in df_dyn.iterrows()
        ]
        Dynamic_data.objects.bulk_create(records)
        print(f"Loaded {len(records)} dynamic records into database.")

    # 2. Seed Static Bills
    if static_csv.exists():
        df_stat = pd.read_csv(static_csv)
        records = [
            Static_data(
                date=row["Date"],
                description=str(row["Description"]),
                amount=float(row["Amount"]),
                category=str(row["Category"]),
                month=str(row["Month"])
            )
            for _, row in df_stat.iterrows()
        ]
        Static_data.objects.bulk_create(records)
        print(f"Loaded {len(records)} static records into database.")

    # 3. Seed Income / Paychecks
    if paycheck_csv.exists():
        df_inc = pd.read_csv(paycheck_csv)
        records = [
            Salary_data(
                date=row["Date"],
                description=str(row["Description"]),
                amount=float(row["Amount"]),
                category=str(row["Category"]),
                month=str(row["Month"])
            )
            for _, row in df_inc.iterrows()
        ]
        Salary_data.objects.bulk_create(records)
        print(f"Loaded {len(records)} salary records into database.")

    print("\nDatabase seeded successfully!")

if __name__ == "__main__":
    seed_database()