# ============================================================
# Notebook: 03_Model_Training
# Purpose:  Train a Prophet time-series forecasting model on
#           daily aggregated sales data, log to MLflow, and 
#           register the model in the Fabric Model Registry.
#
# HOW TO USE IN FABRIC:
#   1. Create a new Notebook in your Fabric workspace
#   2. Rename it to "03_Model_Training"
#   3. Attach your "Sales_Lakehouse" from the left panel
#   4. Copy each "# --- CELL X ---" block into a separate notebook cell
#   5. Run cells in order (Shift+Enter)
#
# PREREQUISITE: You must have run 01 and 02 first (SalesData_Daily_Aggregated must exist).
# ============================================================


# --- CELL 1: Install Prophet ---
# Prophet is NOT pre-installed in Fabric. Run this cell first.
# IMPORTANT: After this cell finishes, click "Restart session" in the toolbar,
# then skip this cell and continue from CELL 2 onwards.

# %pip install prophet

# ^^^ UNCOMMENT the line above when running in Fabric.
# It is commented here to avoid errors when viewing locally.
# After running, RESTART THE SESSION, then continue from Cell 2.


# --- CELL 2: Load the Aggregated Training Data ---
import mlflow
import mlflow.prophet
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Set the MLflow experiment name
# Fabric creates the experiment automatically if it doesn't exist
EXPERIMENT_NAME = "sales-forecasting-experiment"
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"MLflow experiment set to: {EXPERIMENT_NAME}")

# Load the aggregated daily sales data
df_spark = spark.sql("SELECT ds, y FROM SalesData_Daily_Aggregated ORDER BY ds")
df = df_spark.toPandas()
df['ds'] = pd.to_datetime(df['ds'])

print(f"\nTraining data loaded:")
print(f"  Shape:      {df.shape}")
print(f"  Date range: {df['ds'].min().date()} to {df['ds'].max().date()}")
print(f"  Avg daily sales: ${df['y'].mean():,.2f}")
print(f"  Min daily sales: ${df['y'].min():,.2f}")
print(f"  Max daily sales: ${df['y'].max():,.2f}")
df.head()


# --- CELL 3: Train the Prophet Model ---
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from mlflow.models.signature import infer_signature

MODEL_NAME = "prophet_sales_forecaster"

with mlflow.start_run(run_name="prophet_v1_daily_sales") as run:

    # ------------------------------------------------------------------
    # Step 1: Define hyperparameters
    # ------------------------------------------------------------------
    params = {
        "yearly_seasonality":     True,
        "weekly_seasonality":     True,
        "daily_seasonality":      False,
        "seasonality_mode":       "multiplicative",  # "additive" or "multiplicative"
        "changepoint_prior_scale": 0.05,             # flexibility of trend changes
        # NOTE: You can tune these values to improve accuracy.
        # - Higher changepoint_prior_scale = more flexible trend (risk of overfitting)
        # - Lower  changepoint_prior_scale = smoother trend (risk of underfitting)
        # - "multiplicative" seasonality_mode works better when seasonal swings
        #   grow with the overall level of the time series
    }

    # ------------------------------------------------------------------
    # Step 2: Initialize and fit the model
    # ------------------------------------------------------------------
    print("Training Prophet model...")
    model = Prophet(
        yearly_seasonality=params["yearly_seasonality"],
        weekly_seasonality=params["weekly_seasonality"],
        daily_seasonality=params["daily_seasonality"],
        seasonality_mode=params["seasonality_mode"],
        changepoint_prior_scale=params["changepoint_prior_scale"],
    )
    model.fit(df)
    print("✅ Model training complete.")

    # ------------------------------------------------------------------
    # Step 3: Evaluate in-sample accuracy
    # ------------------------------------------------------------------
    print("\nCalculating evaluation metrics...")
    forecast_train = model.predict(df[['ds']])

    mae  = mean_absolute_error(df['y'], forecast_train['yhat'])
    rmse = np.sqrt(mean_squared_error(df['y'], forecast_train['yhat']))
    mape = np.mean(np.abs((df['y'] - forecast_train['yhat']) / df['y'])) * 100

    print(f"  MAE  (Mean Absolute Error):       ${mae:,.2f}")
    print(f"  RMSE (Root Mean Squared Error):    ${rmse:,.2f}")
    print(f"  MAPE (Mean Abs. Percentage Error): {mape:.2f}%")

    # ------------------------------------------------------------------
    # Step 4: Log everything to MLflow
    # ------------------------------------------------------------------
    print("\nLogging to MLflow...")
    mlflow.log_params(params)
    mlflow.log_metrics({
        "mae":  mae,
        "rmse": rmse,
        "mape": mape,
        "training_rows": len(df),
    })

    # ------------------------------------------------------------------
    # Step 5: Create model signature and register
    # ------------------------------------------------------------------
    future_sample = model.make_future_dataframe(periods=30, freq='D')
    forecast_sample = model.predict(future_sample)
    signature = infer_signature(future_sample, forecast_sample)

    model_info = mlflow.prophet.log_model(
        pr_model=model,
        artifact_path="prophet-sales-model",
        registered_model_name=MODEL_NAME,
        signature=signature,
    )

    print(f"\n{'='*50}")
    print(f"✅ MODEL REGISTERED SUCCESSFULLY")
    print(f"{'='*50}")
    print(f"  Model name:  {MODEL_NAME}")
    print(f"  MLflow Run:  {run.info.run_id}")
    print(f"  Model URI:   {model_info.model_uri}")
    print(f"  Experiment:  {EXPERIMENT_NAME}")


# --- CELL 4: Visualize Model Fit ---
import matplotlib.pyplot as plt

# Generate forecast: all historical dates + 90 days into the future
future = model.make_future_dataframe(periods=90, freq='D')
forecast = model.predict(future)

# Plot 1: Forecast with confidence intervals
fig1 = model.plot(forecast)
ax1 = fig1.gca()
ax1.set_title("Sales Forecast: Historical Fit + 90-Day Prediction", fontsize=14, fontweight='bold')
ax1.set_xlabel("Date")
ax1.set_ylabel("Daily Sales ($)")
plt.tight_layout()
plt.show()

# Plot 2: Components (trend, weekly, yearly)
fig2 = model.plot_components(forecast)
plt.tight_layout()
plt.show()

print("📊 Blue dots = actual data, Blue line = forecast, Shaded area = prediction interval")


# --- CELL 5: Cross-Validation (Optional but Recommended) ---
# This takes several minutes to run. Skip if you're in a hurry.
# It performs true out-of-sample evaluation by training on subsets of your data.

from prophet.diagnostics import cross_validation, performance_metrics

print("Running cross-validation (this may take a few minutes)...")
# initial: train on first 365 days, forecast next 30, slide window by 90 days
df_cv = cross_validation(
    model,
    initial='365 days',
    period='90 days',
    horizon='30 days',
)

# Calculate performance metrics
df_perf = performance_metrics(df_cv)
print("\nCross-Validation Performance Metrics:")
print(df_perf[['horizon', 'mae', 'rmse', 'mape']].to_string(index=False))

# Log the best CV metrics to MLflow as well
with mlflow.start_run(run_id=run.info.run_id):
    avg_cv_mape = df_perf['mape'].mean() * 100
    mlflow.log_metric("cv_avg_mape", avg_cv_mape)
    print(f"\n✅ Avg cross-validated MAPE: {avg_cv_mape:.2f}%")

print("\n✅ Model training complete. Proceed to notebook 04_Batch_Scoring.")
