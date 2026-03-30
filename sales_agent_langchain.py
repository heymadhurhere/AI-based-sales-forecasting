# ============================================================
# Notebook: 05_Sales_Agent_LangChain
# Purpose:  A conversational AI agent that connects to the
#           Microsoft Fabric Lakehouse via SQL and lets users
#           ask natural language questions about sales and forecasts.
#
# HOW TO USE IN FABRIC:
#   1. Create a new Notebook in your Fabric workspace
#   2. Rename it to "05_Sales_Agent_LangChain"
#   3. Attach your "Sales_Lakehouse" from the left panel
#   4. Copy each "# --- CELL X ---" block into a separate notebook cell
#   5. Run cells in order (Shift+Enter)
#
# PREREQUISITES:
#   - Notebooks 01-04 must have been run (SalesData + SalesForecasts tables exist)
#   - Azure OpenAI resource deployed with a GPT-4 (or GPT-4o) model
#   - ODBC Driver 18 for SQL Server installed (comes pre-installed in Fabric)
#   - Your Fabric SQL analytics endpoint connection string
# ============================================================


# --- CELL 1: Install Dependencies ---
# IMPORTANT: After running, click "Restart session", then skip to Cell 2.

# %pip install langchain langchain-openai langchain-community pyodbc sqlalchemy
 
# ^^^ UNCOMMENT the line above when running in Fabric.
# After running, RESTART THE SESSION, then continue from Cell 2.


# --- CELL 2: Configuration ---
# Fill in YOUR values in the section below.
# DO NOT share this notebook with credentials in it — use environment variables in production.

import os

# ============================================================
# 🔧 EDIT THESE VALUES WITH YOUR ACTUAL CREDENTIALS
# ============================================================

# --- Fabric SQL Endpoint ---
# How to find: Go to your Lakehouse > switch to "SQL analytics endpoint" view > Settings
# Copy the "SQL connection string" — it looks like: xxxx.datawarehouse.fabric.microsoft.com
SERVER_ENDPOINT = "YOUR-ENDPOINT.datawarehouse.fabric.microsoft.com"
DATABASE_NAME   = "Sales_Lakehouse"

# --- Azure OpenAI ---
# How to find: Azure Portal > your Azure OpenAI resource > Keys and Endpoint
AZURE_OPENAI_API_KEY    = "YOUR-API-KEY-HERE"
AZURE_OPENAI_ENDPOINT   = "https://YOUR-RESOURCE-NAME.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT = "gpt-4"  # The deployment name you created in Azure OpenAI Studio

# ============================================================
# Set environment variables (LangChain reads these automatically)
# ============================================================
os.environ["AZURE_OPENAI_API_KEY"]     = AZURE_OPENAI_API_KEY
os.environ["AZURE_OPENAI_ENDPOINT"]    = AZURE_OPENAI_ENDPOINT
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-08-01-preview"

print("✅ Configuration set.")
print(f"   SQL Endpoint:  {SERVER_ENDPOINT}")
print(f"   Database:      {DATABASE_NAME}")
print(f"   OpenAI Model:  {AZURE_OPENAI_DEPLOYMENT}")


# --- CELL 3: Connect to Fabric Lakehouse SQL Endpoint ---
import urllib
from langchain_community.utilities import SQLDatabase

# Build the SQLAlchemy connection string
connection_string = (
    f"Driver={{ODBC Driver 18 for SQL Server}};"
    f"Server={SERVER_ENDPOINT},1433;"
    f"Database={DATABASE_NAME};"
    f"Authentication=ActiveDirectoryInteractive;"
    f"Encrypt=yes;"
    f"TrustServerCertificate=no;"
    f"Connection Timeout=30;"
)
params = urllib.parse.quote_plus(connection_string)
sqlalchemy_url = f"mssql+pyodbc:///?odbc_connect={params}"

print(f"Connecting to Fabric Lakehouse: {DATABASE_NAME}...")
try:
    db = SQLDatabase.from_uri(
        sqlalchemy_url,
        include_tables=["SalesData", "SalesForecasts"],
        sample_rows_in_table_info=3,  # Show 3 sample rows when describing tables
    )
    print(f"✅ Connected successfully!")
    print(f"   Available tables: {db.get_usable_table_names()}")
    
    # Show table schema so you can verify the connection
    print(f"\nTable Info Preview:")
    print(db.get_table_info())
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\nTroubleshooting steps:")
    print("1. Verify your SQL endpoint URL is correct (check Lakehouse > SQL analytics endpoint > Settings)")
    print("2. Make sure ODBC Driver 18 for SQL Server is installed")
    print("3. Ensure you have access to the Lakehouse in your organization")
    print("4. Try 'Authentication=ActiveDirectoryDefault' if Interactive doesn't work")


# --- CELL 4: Create the LangChain SQL Agent ---
from langchain_openai import AzureChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent

# Initialize the Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    temperature=0.0,  # 0 = deterministic; higher = more creative
)

# Create the SQL Agent
# The agent will:
#   1. Receive your natural language question
#   2. Inspect the table schemas
#   3. Generate a T-SQL query
#   4. Execute it against the Fabric Lakehouse
#   5. Summarize the results in plain English
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="openai-tools",
    verbose=True,        # Set to False in production to hide intermediate steps
    handle_parsing_errors=True,
    prefix="""You are a Sales Forecasting AI Agent connected to a Microsoft Fabric Lakehouse.

You have access to two tables:

1. **SalesData** — Contains historical sales records.
   Columns: OrderDate (date), ProductID (string), ProductName (string), 
            Category (string), Region (string), SalesAmount (float), Quantity (int)

2. **SalesForecasts** — Contains AI-generated future sales predictions.
   Columns: ForecastDate (date), PredictedSales (float), 
            PredictedSales_Lower (float), PredictedSales_Upper (float),
            ForecastGeneratedAt (datetime), ModelName (string), HorizonDays (int)

RULES:
- For questions about past/historical data, use the SalesData table.
- For questions about future/predicted/expected/forecasted data, use the SalesForecasts table.
- Always use T-SQL syntax (this is Microsoft SQL Server / Fabric).
- Format monetary values as currency with 2 decimal places.
- If asked to compare actual vs. forecast, join SalesData and SalesForecasts on date.
- Be precise. Double-check your SQL before executing.
- If you're unsure which table to use, explain your assumption.
""",
)

print("\n" + "=" * 60)
print("🤖 AI Sales Forecasting Agent is Ready!")
print("=" * 60)
print(f"Connected to: {DATABASE_NAME}")
print(f"LLM:          Azure OpenAI ({AZURE_OPENAI_DEPLOYMENT})")
print("=" * 60)


# --- CELL 5: Ask Questions (Run this cell repeatedly) ---
# Change the 'question' variable below to whatever you want to ask,
# then re-run this cell to get an answer.

question = "What were the total sales by category in 2023?"

# --- You can replace the question above with any of these examples:
# question = "What were total sales in 2023?"
# question = "Show me monthly sales trend for Electronics category in 2023"
# question = "Which region had the highest sales last year?"
# question = "What is the sales forecast for the next 2 weeks?"
# question = "Compare the predicted sales upper and lower bounds for next month"
# question = "What was the best-selling product in Q4 2023?"
# question = "Show average daily sales by day of week"
# question = "What is the total predicted sales for the next 30 days?"

print(f"\n❓ Question: {question}\n")
try:
    response = agent_executor.invoke({"input": question})
    print("\n📊 Answer:")
    print(response["output"])
except Exception as e:
    print(f"❌ Error: {e}")
    print("Try rephrasing your question or check the SQL endpoint connection.")


# --- CELL 6: Interactive Chat Loop (Optional) ---
# Uncomment and run this cell for a continuous chat experience.
# Type 'exit' or 'quit' to stop.

# print("\n" + "=" * 60)
# print("🤖 Interactive Chat Mode — type 'exit' to quit")
# print("=" * 60 + "\n")
#
# while True:
#     user_input = input("You > ")
#     if user_input.lower().strip() in ("exit", "quit", "q"):
#         print("Goodbye! 👋")
#         break
#     if not user_input.strip():
#         continue
#     try:
#         response = agent_executor.invoke({"input": user_input})
#         print(f"\n📊 Agent: {response['output']}\n")
#     except Exception as e:
#         print(f"❌ Error: {e}\n")
