# 📊 SalesForecast AI

SalesForecast AI is an intelligent data-driven solution designed to predict and visualize business sales trends. Using advanced machine learning time-series models (**Facebook Prophet**) wrapped in a modern, dynamic **Streamlit** user interface, this agent operates like a virtual "Editorial Analyst." It extrapolates complex patterns extending beyond the historical baseline to provide a granular, interactive view of your pipeline overview, historical trends, and expected financial growth.

---

## ✨ Key Features
- **Predictive Engine:** Leverages the robust `Prophet` framework to detect yearly and weekly seasonalities and process robust forward-looking analytics.
- **SaaS Aesthetic Dashboard:** A natively styled analytical Streamlit dashboard engineered with high-end, responsive custom CSS, metric cards, and drop indicators. 
- **Interactive Visualizations:** Employs `Plotly` to orchestrate smooth, completely interactive web-ready charting equipped with zoom functions and hover tooltips.
- **Parametric Filtering:** Toggle "Manual Entry" mode to independently search and slice custom category/region data parameters for localized granular breakdowns.
- **Export Ready:** Generate raw CSV documents instantly summarizing the extrapolated future revenue timelines on your local desktop directly through the application engine.

## 🛠️ Tech Stack
- **Python 3.9+**
- **Streamlit** (Frontend SaaS framework & interface)
- **Prophet** (Time Series Forecasting & Machine Learning generation)
- **Plotly & Pandas** (Interactive Data manipulation)
- **Docker** (Hugging Face Spaces deployment containerization)

---

## 🚀 Getting Started Locally

### 1. Clone & Set Up Environment
It is highly recommended to isolate the project inside a virtual environment.
```bash
git clone <your-repository-url>
cd "AI Based Sales Forecasting"
python -m venv venv
```

**Activate the VENV:**
- *Windows (PowerShell):* `.\venv\Scripts\Activate.ps1`
- *Mac/Linux:* `source venv/bin/activate`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Agent Application
Spin up the local Streamlit server to preview the dashboard:
```bash
streamlit run app.py
```
You can access your live forecast hub at `http://localhost:8501`.

---

## 🐳 Deployment to Hugging Face Spaces (Docker)

This repository is pre-configured with a custom containerized `Dockerfile` expressly formatted for the **Hugging Face Docker SDK** Spaces environment.

**Configuration included:**
- Pulls `python:3.9-slim` optimizing for server load limits over space restrictions.
- Installs all the precise required dependencies utilizing the included `requirements.txt`.
- Operates automatically on standard containerized port `7860`.

**Deployment Push Steps:**
1. Create a brand new "Docker" Space on Hugging Face.
2. Commit and push this directory via the terminal exactly as instructed in the HF repository.
```bash
git add .
git commit -m "Initial commit for SalesForecast AI Docker deployment"
git push
```
3. The Space will automatically transition to "Building" utilizing the `Dockerfile` definitions.

---

## 🗂️ Project Structure

```text
📁 AI Based Sales Forecasting
│
├── 📄 app.py                 # Core application layout, Streamlit UI, and inference logic
├── 📄 sales_forecasting.ipynb# EDA architecture drafting & notebook proofs  
├── 📄 Dockerfile             # Defines container instructions for HF SDK deployment
├── 📄 requirements.txt       # Production dependencies for UI / ML 
└── 📄 README.md              # Project documentation
```

*(Note: Currently, `app.py` utilizes smart dummy data simulation functions locally. Please replace the `load_data()` content with your primary backend CSV/SQL database pipelines before enterprise usage.)*
