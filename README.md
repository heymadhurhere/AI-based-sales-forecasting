# 🤖 AI Sales Forecasting Agent — Microsoft Fabric

> An end-to-end AI-powered sales forecasting system built entirely on **Microsoft Fabric**.
> It ingests sales data, trains a machine-learning model, generates future forecasts, and
> exposes a natural-language AI agent so any business user can ask questions in plain English.

---

## 📌 What This Project Does

Most companies have sales history sitting in databases but no easy way to predict the future or let non-technical employees query the data. This project solves both problems:

1. **Forecasting** — Trains a [Facebook Prophet](https://facebook.github.io/prophet/) time-series model on your historical sales data and produces daily sales predictions up to 30 days ahead, complete with upper/lower confidence bounds.
2. **AI Agent** — Wraps that data in a conversational AI agent. You type a question in plain English ("Which region had the highest sales last quarter?") and the agent figures out the right SQL, runs it against your Lakehouse, and replies in plain English.

Everything lives inside **Microsoft Fabric** — no separate infrastructure needed.

---

## 🏗️ Architecture

```
Raw Sales Data (CSV or ERP)
        │
        ▼
┌─────────────────────────┐
│  01 · Data Ingestion    │  ──► Delta Table: SalesData
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  02 · EDA & Data Prep   │  ──► Delta Table: SalesData_Daily_Aggregated
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  03 · Model Training    │  ──► MLflow Registry: prophet_sales_forecaster
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  04 · Batch Scoring     │  ──► Delta Table: SalesForecasts
└─────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  05 · AI Agent  (choose one)                            │
│                                                         │
│  Option A — Fabric Data Agent  (no-code, built-in)      │
│  Option B — LangChain + Azure OpenAI  (code-based)      │
└─────────────────────────────────────────────────────────┘
```

Every step is a self-contained Fabric Notebook. You run them in order, **01 → 02 → 03 → 04**, and then pick your agent approach for step 05.

---

## 📁 File-by-File Breakdown

| File | Fabric Notebook Name | What It Does |
|------|---------------------|--------------|
| `01_data_ingestion.py` | `01_Data_Ingestion` | Generates realistic synthetic sales data (5 products × 4 regions × 3 years = ~21,900 rows) **or** loads your own CSV. Writes a Delta table named `SalesData`. |
| `02_eda_and_data_prep.py` | `02_EDA_and_Data_Prep` | Explores the data with charts (daily trend, sales by category/region, monthly bars), runs data-quality checks (nulls, negatives, duplicates), then creates a daily-aggregated table (`SalesData_Daily_Aggregated`) that the model needs. |
| `data_prep_and_train.py` | `03_Model_Training` | Trains a Prophet forecasting model with yearly + weekly seasonality. Logs all hyperparameters and accuracy metrics (MAE, RMSE, MAPE) to **MLflow**. Registers the final model in the Fabric **Model Registry** as `prophet_sales_forecaster`. Includes an optional cross-validation step for true out-of-sample evaluation. |
| `batch_scoring_pipeline.py` | `04_Batch_Scoring` | Loads the latest registered model from the Model Registry, creates a 30-day future date range, runs predictions, and saves the results to a Delta table named `SalesForecasts` with confidence intervals. Can be **scheduled** to run automatically (daily/weekly). |
| `sales_agent_langchain.py` | `05_Sales_Agent_LangChain` | *(Optional — code path)* A LangChain SQL agent backed by **Azure OpenAI GPT-4**. Connects to the Fabric Lakehouse SQL analytics endpoint, understands your table schemas, and translates natural-language questions into T-SQL, executes them, and returns readable answers. |
| `06_fabric_data_agent_instructions.py` | *(config file)* | *(Optional — no-code path)* Ready-to-paste system prompt, table descriptions, and example Q&A pairs for configuring the built-in **Fabric Data Agent** in the Fabric portal. No Azure OpenAI required. |
| `sales_data.csv` | *(upload to Lakehouse)* | Sample CSV file you can upload to the Lakehouse `Files` section and use with the "Option B" path in `01_data_ingestion.py` if you don't want to use the synthetic generator. |

---

## 🗄️ Data Schema

### `SalesData` — Historical Transactions

| Column | Type | Description |
|--------|------|-------------|
| `OrderDate` | Date | Date the sale occurred |
| `ProductID` | String | Unique product code (e.g., `P001`) |
| `ProductName` | String | Product name (e.g., `Laptop Pro`) |
| `Category` | String | Product category (`Electronics`, `Furniture`) |
| `Region` | String | Sales region (`North`, `South`, `East`, `West`) |
| `SalesAmount` | Double | Revenue in USD |
| `Quantity` | Integer | Units sold |

### `SalesData_Daily_Aggregated` — Prophet Training Input

| Column | Type | Description |
|--------|------|-------------|
| `ds` | Date | Date (Prophet's required column name) |
| `y` | Double | Total daily sales across all products and regions |

### `SalesForecasts` — AI-Generated Predictions

| Column | Type | Description |
|--------|------|-------------|
| `ForecastDate` | Date | The date being predicted |
| `PredictedSales` | Double | Model's best estimate of daily sales ($) |
| `PredictedSales_Lower` | Double | Lower bound of the prediction interval |
| `PredictedSales_Upper` | Double | Upper bound of the prediction interval |
| `ForecastGeneratedAt` | Timestamp | When this batch scoring run happened |
| `ModelName` | String | Name of the model used (`prophet_sales_forecaster`) |
| `HorizonDays` | Integer | How many days ahead this run was forecasting |

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| Platform | Microsoft Fabric (F2+ capacity or Trial) |
| Storage | OneLake Lakehouse — Delta / Parquet tables |
| Data Processing | Apache Spark (PySpark) |
| ML Modeling | [Prophet](https://facebook.github.io/prophet/) by Meta |
| Model Tracking | MLflow (built into Fabric) |
| AI Agent (option A) | Fabric Data Agent (native, no-code) |
| AI Agent (option B) | LangChain + Azure OpenAI GPT-4 |
| Visualization | Matplotlib (inside notebooks) |
| Language | Python 3 |

---

## ✅ Prerequisites

Before you start, make sure you have:

- [ ] A **Microsoft Fabric account** — trial (60 days free) or paid F2+ capacity
  - Sign up at [app.fabric.microsoft.com](https://app.fabric.microsoft.com/)
  - You need a Microsoft 365 work/school account (personal Outlook/Gmail won't work)
- [ ] **Tenant admin settings enabled** (ask your admin if you're not the admin):
  - "Users can create Fabric items" → On
  - "Copilot and Azure OpenAI Service" → On
  - "Fabric data agent" → On (only needed for Option A agent)
- [ ] *(Only for Option B LangChain agent)* An **Azure OpenAI** resource with a GPT-4 deployment

---

## 🚀 Quick Start — Step by Step

### Step 1 — Create a Workspace & Lakehouse
1. Log into [app.fabric.microsoft.com](https://app.fabric.microsoft.com/)
2. Click **Workspaces → + New workspace** → name it `SalesForecasting-WS` → **Apply**
3. Inside the workspace, click **+ New item → Lakehouse** → name it `Sales_Lakehouse` → **Create**

### Step 2 — Run Notebook 01: Data Ingestion

1. Inside the Lakehouse, click **Open notebook → New notebook**
2. Rename it `01_Data_Ingestion`
3. Open `01_data_ingestion.py` from this repo
4. Each `# --- CELL X ---` comment marks a **separate notebook cell** — paste each block into its own cell
5. **If using synthetic data:** Run Cell 1 (generates data) then Cell 2 with Option A
6. **If you have a CSV:** Upload it to Lakehouse → Files, then in Cell 2 switch to Option B
7. Run Cell 3 to verify the `SalesData` table was created

### Step 3 — Run Notebook 02: EDA & Data Prep

1. Create a new notebook → rename to `02_EDA_and_Data_Prep`
2. Attach `Sales_Lakehouse` (left panel → Add)
3. Paste each cell block from `02_eda_and_data_prep.py`
4. Run all 7 cells — you'll see charts and a quality report
5. Cell 7 creates the `SalesData_Daily_Aggregated` table needed for training

### Step 4 — Run Notebook 03: Model Training

1. Create a new notebook → rename to `03_Model_Training`
2. Attach `Sales_Lakehouse`
3. Paste cells from `data_prep_and_train.py`
4. **Cell 1:** Uncomment `%pip install prophet`, run it, then **restart the session**
5. Run Cells 2–4 (load data → train → visualize)
6. *(Optional)* Run Cell 5 for cross-validation metrics
7. Check **Workspace → ML models** to confirm `prophet_sales_forecaster` is registered

### Step 5 — Run Notebook 04: Batch Scoring

1. Create a new notebook → rename to `04_Batch_Scoring`
2. Attach `Sales_Lakehouse`
3. Paste cells from `batch_scoring_pipeline.py`
4. **Cell 1:** Uncomment `%pip install prophet`, run it, then **restart the session**
5. Run Cells 2–3 → forecasts are written to `SalesForecasts` table
6. *(Optional)* Schedule this notebook to run daily: Workspace → notebook → `...` → **Schedule**

### Step 6 — Set Up the AI Agent (Choose One)

#### Option A — Fabric Data Agent (Recommended, no code)
1. Go to workspace → **+ New item → Data Agent** → name it `Sales Forecasting Agent`
2. Connect data source: select `Sales_Lakehouse` (SQL endpoint) or your semantic model
3. In the **Instructions** field, paste the system prompt from `06_fabric_data_agent_instructions.py`
4. Add the example Q&A pairs from the same file to help the agent learn
5. Test with the sample questions at the bottom of that file
6. Click **Publish** to share with your team

#### Option B — LangChain Custom Agent (requires Azure OpenAI)
1. Create a new notebook → rename to `05_Sales_Agent_LangChain`
2. Attach `Sales_Lakehouse`
3. Paste cells from `sales_agent_langchain.py`
4. **Cell 1:** Uncomment the `%pip install` line, run it, restart session
5. **Cell 2:** Fill in your `SERVER_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and `AZURE_OPENAI_ENDPOINT`
6. Run Cells 3–4 to connect and build the agent
7. In Cell 5, change the `question` variable and run to get answers

---

## 💬 Example Questions to Ask the Agent

Once the agent is running, try these:

```
"What were the total sales in 2023?"
"Which product generated the most revenue last year?"
"Show me monthly sales trends for the Electronics category"
"Which region had the highest sales in Q3 2023?"
"What is the sales forecast for the next 2 weeks?"
"What is the total predicted revenue for next month?"
"Compare the best-case and worst-case forecast for next 30 days"
"How did Furniture sales compare to Electronics in 2023?"
"What was our average daily sales in Q4 2023?"
```

---

## 🔁 How the Model Works

The model used is **Facebook Prophet**, a decomposable time-series algorithm.

It breaks your sales data into three components:

| Component | What It Captures |
|-----------|-----------------|
| **Trend** | The overall long-term upward/downward direction |
| **Yearly Seasonality** | Patterns that repeat every year (e.g., peak mid-year) |
| **Weekly Seasonality** | Patterns within a week (e.g., weekday vs. weekend) |
| **Residual Noise** | Random variation unexplained by the above |

**Key hyperparameters** (you can tune in `data_prep_and_train.py`):

| Parameter | Default | Effect |
|-----------|---------|--------|
| `seasonality_mode` | `multiplicative` | Use `additive` if seasonality doesn't scale with the data level |
| `changepoint_prior_scale` | `0.05` | Higher = trend changes faster (risk of overfitting) |
| `yearly_seasonality` | `True` | Needs 1+ year of data to be useful |
| `weekly_seasonality` | `True` | Needs several weeks of data |

**Evaluation metrics** are automatically logged to MLflow:
- **MAE** — Average dollar error per day
- **RMSE** — Penalises large errors more heavily
- **MAPE** — Error as a percentage of actual sales (easier to interpret)

---

## 📊 Fabric Items Created by This Project

| Item Name | Type | Created By |
|-----------|------|-----------|
| `SalesForecasting-WS` | Workspace | You (manually) |
| `Sales_Lakehouse` | Lakehouse | You (manually) |
| `SalesData` | Delta Table | `01_data_ingestion.py` |
| `SalesData_Daily_Aggregated` | Delta Table | `02_eda_and_data_prep.py` |
| `SalesForecasts` | Delta Table | `batch_scoring_pipeline.py` |
| `sales-forecasting-experiment` | MLflow Experiment | `data_prep_and_train.py` |
| `prophet_sales_forecaster` | ML Model (Registry) | `data_prep_and_train.py` |
| `Sales Forecasting Agent` | Data Agent | You (via file 06 config) |

---

## 🛠️ Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: No module named 'prophet'` | Prophet not installed in session | Uncomment `%pip install prophet` in Cell 1, run it, then **restart the session** |
| Table not appearing in Lakehouse Explorer | UI cache | Click **⟳ Refresh** at the top of the Tables section |
| `Data Agent` option missing in "+ New item" | Feature not enabled in tenant | Ask your admin to enable "Fabric data agent" in Admin Portal → Tenant settings |
| Model loads but gives poor forecasts | Not enough training data | Ensure at least 2 full years of daily data; tune `changepoint_prior_scale` |
| SQL connection fails in LangChain agent | Wrong endpoint or auth | Double-check the SQL endpoint URL from Lakehouse → SQL analytics endpoint → Settings |
| `Authentication=ActiveDirectoryInteractive` fails | Non-interactive environment | Try `Authentication=ActiveDirectoryDefault` or use a service principal |
| Fabric trial expired | 60-day limit reached | Purchase F2+ Fabric capacity, or ask your org's admin for access |

---

## 📝 Notes for Developers

- **Each `.py` file maps to exactly one Fabric Notebook.** Each `# --- CELL X ---` section inside the file should be pasted as a separate cell in the Fabric notebook UI.
- **`%pip install` lines are commented out** in the source files to prevent errors when opening them locally. Uncomment them only inside Fabric.
- **Credentials in `sales_agent_langchain.py`** are placeholder strings. Never commit real API keys — use Fabric environment variables or Azure Key Vault in production.
- **The `SalesForecasts` table uses `mode("overwrite")`** by default, meaning each run of the scoring pipeline replaces the old forecasts with fresh ones. Change to `mode("append")` if you want to keep a history of all scoring runs.
- **The `spark-warehouse/` folder** in this repo is auto-generated by a local Spark session and can be ignored — it is not used in Fabric.
