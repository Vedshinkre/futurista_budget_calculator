# Futurista Budget Calculator

An intelligent personal finance web application built with **Django** and **Blablador (LLM)**. The platform bridges the gap between daily discretionary spending and long-term financial goals by providing forward-looking savings feasibility analysis and retrospective monthly cash flow audits.

---

## Key Features

### 1. Goal-Oriented Savings Planner

- **Dynamic Feasibility Engine:** Evaluates ambitious financial goals (e.g., vacations, sabbatical funds) against historical net cash buffers.
- **Tiered Feasibility Scoring:**
  - **Effortless (100%):** Existing cash buffers fully absorb the goal. Skips AI calls to prevent unnecessary latency and token costs.
  - **Achievable (65%–99%):** Prompts the LLM to analyze frequent merchant patterns and propose targeted, painless cuts to discretionary spending.
  - **Low Feasibility (<65%):** Automatically recalibrates the timeline and presents achievable budget milestones:
    - Zero-sacrifice option.
    - Balanced option (70%).
    - Extended timeline option.

### 2. Monthly Financial Health Audit

- **Cash Flow Breakdown:** Aggregates net income, fixed bills (static), and discretionary expenses (dynamic), filtered by exact year and month to prevent multi-year data collisions.
- **Intelligent Spending Diagnosis:** Uses a dedicated LLM prompt engine to compare the target month against historical spending baselines, grade financial health, detect lifestyle creep, and highlight specific category leaks.

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend | Python 3.12+, Django 5.x |
| Data Processing | Pandas, OpenPyXL |
| AI & LLM Integration | OpenAI Python SDK configured for the Blablador API (`alias-fast`) |
| Database | SQLite (Django ORM) |
| Frontend | Django Templates, HTML5, CSS3, Tailwind CSS |

---

## Project Structure

```text
├── financial/                  # Django core application
│   ├── models.py               # Salary_data, Static_data, Dynamic_data models
│   ├── views.py                # Application controllers and endpoints
│   ├── urls.py                 # Route mappings
│   └── templates/              # Dashboard, results, and audit forms
│
├── src/                        # Analytics and AI business logic
│   ├── generate.py             # Core evaluation engine and dual Blablador prompt engines
│   └── data_understanding.py   # Data ingestion and Excel inspection utilities
│
├── hackathon_project/          # Django project configuration
│   ├── settings.py
│   └── urls.py
│
├── manage.py
├── requirements.txt
└── .gitignore
```

---

## Getting Started

### 1. Prerequisites

Make sure you have the following installed:

- Python 3.10 or higher
- Git

### 2. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 3. Set Up a Virtual Environment

**Windows (Git Bash):**

```bash
python -m venv venv
source venv/Scripts/activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the project's root directory:

```ini
BLABLADOR_API_KEY=your_blablador_api_key_here
DEBUG=True
```

Replace the placeholder values with your actual API key and Django secret key.

### 6. Apply Database Migrations

Run the following commands to initialize the database:

```bash
python manage.py makemigrations
python manage.py migrate
```

### 7. Run the Development Server

Start the Django development server:

```bash
python manage.py runserver
```

Open your browser and visit:

```text
http://127.0.0.1:8000/
```

---

## Testing Core Logic Standalone

You can test the financial calculations and LLM integrations directly without starting the full Django web server.

Run:

```bash
python -m src.generate
```

This verifies:

1. Historical aggregation across different periods.
2. Savings feasibility calculations and trade-off generation.
3. Monthly cash flow diagnostics for the most recent month.

---

## Environment Variables

The application requires the following environment variables:

| Variable | Description |
|---|---|
| `BLABLADOR_API_KEY` | API key for accessing the Blablador LLM service |
| `DJANGO_SECRET_KEY` | Django application's secret key |
| `DEBUG` | Enables or disables Django debug mode |

