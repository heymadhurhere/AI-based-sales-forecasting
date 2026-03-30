# ============================================================
# Notebook: 04_Batch_Scoring
# Purpose:  Load the registered Prophet model from MLflow,
#           generate future forecasts, and save the results
#           back to the Fabric Lakehouse as a Delta table.
#
# HOW TO USE IN FABRIC:
#   1. Create a new Notebook in your Fabric workspace
#   2. Rename it to "04_Batch_Scoring"
#   3. Attach your "Sales_Lakehouse" from the left panel
#   4. Copy each "# --- CELL X ---" block into a separate notebook cell
#   5. Run cells in order (Shift+Enter)
#
# PREREQUISITE: You must have run 03_Model_Training first
#               (so the model is registered in MLflow).
#
# SCHEDULING: After testing, you can schedule this notebook to run
#             automatically (daily/weekly) via Workspace > notebook 
#             > ... > Schedule.
# ============================================================


# --- CELL 1: Install Prophet ---
# Required to deserialize the Prophet model from MLflow.
# IMPORTANT: After running, click "Restart session", then skip to Cell 2.

# %pip install prophet

# ^^^ UNCOMMENT the line above when running in Fabric.
# After running, RESTART THE SESSION, then continue from Cell 2.


# --- CELL 2: Load Model and Generate Forecasts ---
import mlflow
import mlflow.prophet
import pandas as pd
from datetime import datetime
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# ------------------------------------------------------------------
# Configuration — Adjust these values as needed
# ------------------------------------------------------------------
MODEL_NAME           = "prophet_sales_forecaster"  # Must match the registered model name
FORECAST_HORIZON_DAYS = 30                          # How many days to forecast
TARGET_TABLE         = "SalesForecasts"             # Output table name in Lakehouse

# ------------------------------------------------------------------
# Step 1: Load the latest version of the registered model
# ------------------------------------------------------------------
print(f"Loading registered model: {MODEL_NAME}...")
model_uri = f"models:/{MODEL_NAME}/latest"
loaded_model = mlflow.prophet.load_model(model_uri)
print("✅ Model loaded successfully.")

# ------------------------------------------------------------------
# Step 2: Generate future dates
# ------------------------------------------------------------------
print(f"Creating future dataframe for {FORECAST_HORIZON_DAYS} days...")
future_df = loaded_model.make_future_dataframe(periods=FORECAST_HORIZON_DAYS, freq='D')
print(f"   Total dates (historical + future): {len(future_df)}")

# ------------------------------------------------------------------
# Step 3: Generate predictions
# ------------------------------------------------------------------
print("Generating predictions...")
forecast = loaded_model.predict(future_df)

# Keep only relevant columns and rename for business clarity
forecast_results = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
forecast_results = forecast_results.rename(columns={
    'ds':         'ForecastDate',
    'yhat':       'PredictedSales',
    'yhat_lower': 'PredictedSales_Lower',
    'yhat_upper': 'PredictedSales_Upper',
})

# Add metadata columns
forecast_results['ForecastGeneratedAt'] = datetime.now()
forecast_results['ModelName']           = MODEL_NAME
forecast_results['HorizonDays']         = FORECAST_HORIZON_DAYS

# Round financial numbers to 2 decimal places
forecast_results['PredictedSales']       = forecast_results['PredictedSales'].round(2)
forecast_results['PredictedSales_Lower'] = forecast_results['PredictedSales_Lower'].round(2)
forecast_results['PredictedSales_Upper'] = forecast_results['PredictedSales_Upper'].round(2)

print(f"   Predictions generated: {len(forecast_results)} rows")

# ------------------------------------------------------------------
# Step 4: Save results to Lakehouse Delta table
# ------------------------------------------------------------------
print(f"Writing results to Lakehouse table '{TARGET_TABLE}'...")
spark_forecast = spark.createDataFrame(forecast_results)

# Using "overwrite" so we always have the latest full forecast.
# Change to "append" if you want to keep a history of all forecasts.
spark_forecast.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(TARGET_TABLE)

# ------------------------------------------------------------------
# Step 5: Verify and preview
# ------------------------------------------------------------------
row_count = spark.sql(f"SELECT COUNT(*) as cnt FROM {TARGET_TABLE}").collect()[0]['cnt']
print(f"\n{'='*50}")
print(f"✅ BATCH SCORING COMPLETE")
print(f"{'='*50}")
print(f"   Table:       {TARGET_TABLE}")
print(f"   Rows saved:  {row_count:,}")
print(f"   Model used:  {MODEL_NAME}")
print(f"   Generated:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print(f"\nFuture forecasts (upcoming dates only):")
spark.sql(f"""
    SELECT 
        ForecastDate,
        PredictedSales,
        PredictedSales_Lower,
        PredictedSales_Upper
    FROM {TARGET_TABLE}
    WHERE ForecastDate > current_date()
    ORDER BY ForecastDate
    LIMIT 15
""").show()


# --- CELL 3: Visualize the Forecast ---
import matplotlib.pyplot as plt

df_plot = spark.sql(f"""
    SELECT ForecastDate, PredictedSales, PredictedSales_Lower, PredictedSales_Upper
    FROM {TARGET_TABLE}
    ORDER BY ForecastDate
""").toPandas()

df_plot['ForecastDate'] = pd.to_datetime(df_plot['ForecastDate'])

# Only plot the last 180 days for clarity
cutoff = df_plot['ForecastDate'].max() - pd.Timedelta(days=180)
df_plot = df_plot[df_plot['ForecastDate'] >= cutoff]

fig, ax = plt.subplots(figsize=(14, 6))

# Today line
today = pd.Timestamp.now().normalize()
ax.axvline(x=today, color='red', linestyle='--', linewidth=1.5, label='Today', alpha=0.7)

# Forecast line
ax.plot(df_plot['ForecastDate'], df_plot['PredictedSales'], 
        color='#2196F3', linewidth=2, label='Predicted Sales')

# Confidence interval
ax.fill_between(
    df_plot['ForecastDate'],
    df_plot['PredictedSales_Lower'],
    df_plot['PredictedSales_Upper'],
    alpha=0.2, color='#2196F3', label='Prediction Interval'
)

ax.set_title('Sales Forecast — Last 180 Days + Future', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Daily Sales ($)')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("✅ Batch scoring notebook complete.")
print("   Next step: Create a Fabric Data Agent (Phase 7) or schedule this notebook.")
