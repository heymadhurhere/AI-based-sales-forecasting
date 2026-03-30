# ============================================================
# Notebook: 02_EDA_and_Data_Prep
# Purpose:  Exploratory Data Analysis, data quality checks,
#           and prepare aggregated data for Prophet model training.
#
# HOW TO USE IN FABRIC:
#   1. Create a new Notebook in your Fabric workspace
#   2. Rename it to "02_EDA_and_Data_Prep"
#   3. Attach your "Sales_Lakehouse" from the left panel
#   4. Copy each "# --- CELL X ---" block into a separate notebook cell
#   5. Run cells in order (Shift+Enter)
#
# PREREQUISITE: You must have run 01_Data_Ingestion first.
# ============================================================


# --- CELL 1: Load and Explore the Data ---
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Load from the Delta table created in notebook 01
df = spark.sql("SELECT * FROM SalesData")

print(f"Total rows:    {df.count():,}")
print(f"Columns:       {df.columns}")
print(f"\nSchema:")
df.printSchema()
print("\nBasic Statistics:")
df.describe().show()


# --- CELL 2: Visualize Daily Sales Trend ---
import matplotlib.pyplot as plt
import pandas as pd

# Aggregate to daily total sales across all products/regions
df_daily = spark.sql("""
    SELECT 
        OrderDate, 
        SUM(SalesAmount) as TotalSales
    FROM SalesData
    GROUP BY OrderDate
    ORDER BY OrderDate
""").toPandas()

df_daily['OrderDate'] = pd.to_datetime(df_daily['OrderDate'])

# Plot the time series
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df_daily['OrderDate'], df_daily['TotalSales'], linewidth=0.8, color='#2196F3')
ax.set_title('Daily Total Sales Over Time', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Sales Amount ($)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# --- CELL 3: Sales by Category ---
import matplotlib.pyplot as plt

df_category = spark.sql("""
    SELECT 
        Category,
        ROUND(SUM(SalesAmount), 2) as TotalSales,
        SUM(Quantity) as TotalQuantity
    FROM SalesData
    GROUP BY Category
    ORDER BY TotalSales DESC
""").toPandas()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart: Sales by Category
axes[0].barh(df_category['Category'], df_category['TotalSales'], color=['#2196F3', '#FF9800', '#4CAF50'])
axes[0].set_title('Total Sales by Category', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Sales Amount ($)')

# Bar chart: Quantity by Category
axes[1].barh(df_category['Category'], df_category['TotalQuantity'], color=['#2196F3', '#FF9800', '#4CAF50'])
axes[1].set_title('Total Quantity by Category', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Quantity')

plt.tight_layout()
plt.show()

print("Sales by Category:")
print(df_category.to_string(index=False))


# --- CELL 4: Sales by Region ---
df_region = spark.sql("""
    SELECT 
        Region,
        ROUND(SUM(SalesAmount), 2) as TotalSales,
        COUNT(*) as NumTransactions
    FROM SalesData
    GROUP BY Region
    ORDER BY TotalSales DESC
""").toPandas()

fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
ax.bar(df_region['Region'], df_region['TotalSales'], color=colors)
ax.set_title('Total Sales by Region', fontsize=13, fontweight='bold')
ax.set_ylabel('Sales Amount ($)')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print("Sales by Region:")
print(df_region.to_string(index=False))


# --- CELL 5: Monthly Sales Trend ---
df_monthly = spark.sql("""
    SELECT 
        DATE_TRUNC('month', OrderDate) as Month,
        ROUND(SUM(SalesAmount), 2) as MonthlySales
    FROM SalesData
    GROUP BY DATE_TRUNC('month', OrderDate)
    ORDER BY Month
""").toPandas()

df_monthly['Month'] = pd.to_datetime(df_monthly['Month'])

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(df_monthly['Month'], df_monthly['MonthlySales'], width=20, color='#673AB7', alpha=0.8)
ax.plot(df_monthly['Month'], df_monthly['MonthlySales'], color='#E91E63', linewidth=2, marker='o', markersize=4)
ax.set_title('Monthly Sales Trend', fontsize=14, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Sales Amount ($)')
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# --- CELL 6: Data Quality Checks ---
from pyspark.sql.functions import col, count, when, isnan, isnull

print("=" * 50)
print("DATA QUALITY REPORT")
print("=" * 50)

# Check for nulls
print("\n1. NULL / Missing Values:")
null_counts = df.select([
    count(when(col(c).isNull(), c)).alias(c) for c in df.columns
])
null_counts.show()

# Check date range
print("2. Date Range:")
spark.sql("""
    SELECT 
        MIN(OrderDate) as earliest_date, 
        MAX(OrderDate) as latest_date,
        DATEDIFF(MAX(OrderDate), MIN(OrderDate)) as total_days
    FROM SalesData
""").show()

# Check for negative sales
print("3. Negative Sales Check:")
neg_count = df.filter(col("SalesAmount") < 0).count()
print(f"   Rows with negative SalesAmount: {neg_count}")

# Check for duplicates
total = df.count()
distinct = df.distinct().count()
print(f"\n4. Duplicate Check:")
print(f"   Total rows:    {total:,}")
print(f"   Distinct rows: {distinct:,}")
print(f"   Duplicates:    {total - distinct:,}")

# Check products and regions
print("\n5. Unique Values:")
spark.sql("""
    SELECT 
        COUNT(DISTINCT ProductID) as products,
        COUNT(DISTINCT Category) as categories,
        COUNT(DISTINCT Region) as regions
    FROM SalesData
""").show()

print("✅ Data quality checks complete.")


# --- CELL 7: Prepare Prophet-Ready Aggregated Data ---
# Prophet needs exactly two columns: 'ds' (date) and 'y' (target value)

df_prophet_ready = spark.sql("""
    SELECT 
        OrderDate as ds, 
        SUM(SalesAmount) as y
    FROM SalesData
    GROUP BY OrderDate
    ORDER BY OrderDate
""")

# Save as a separate Delta table for easy access in the training notebook
AGGREGATED_TABLE = "SalesData_Daily_Aggregated"
df_prophet_ready.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(AGGREGATED_TABLE)

# Verify
agg_count = spark.sql(f"SELECT COUNT(*) as cnt FROM {AGGREGATED_TABLE}").collect()[0]["cnt"]
print(f"✅ Created aggregated table '{AGGREGATED_TABLE}' with {agg_count:,} daily records")
print("\nSample rows (Prophet format — ds and y):")
spark.sql(f"SELECT * FROM {AGGREGATED_TABLE} LIMIT 5").show()

print("\n✅ EDA and Data Prep complete. Proceed to notebook 03_Model_Training.")
