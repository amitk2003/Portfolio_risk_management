import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Streamlit Page Settings ---
st.set_page_config(
    page_title="Portfolio Risk Management",
    layout="wide"
)

# Sidebar
st.sidebar.title("Portfolio Optimization Tool")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

# ---- Safe dataset path handling (important for deployment) ----
current_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(current_dir, "all_stocks_5yr.csv")

st.sidebar.markdown("---")
st.sidebar.write("If you don't upload a file, the default dataset will be used.")

# ---- Load dataset safely ----
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Using uploaded dataset.")
elif os.path.exists(DEFAULT_DATASET):
    df = pd.read_csv(DEFAULT_DATASET)
    st.sidebar.info("Using default dataset.")
else:
    st.error("⚠ Default dataset not found on server. Please upload a CSV file to continue.")
    st.stop()

# --- Main Title ---
st.title("Portfolio Risk Management & Optimization Dashboard")

# --- Dataset Preview ---
st.subheader("Dataset Overview")
st.write("The table below displays the first rows of the dataset:")
st.dataframe(df.head())

# --- Data Preparation ---
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Pivot to wide format
price_df = df.pivot(index="date", columns="Name", values="close").ffill().dropna()

# Calculate daily returns
returns = price_df.pct_change().dropna()

# --- Optimization Computation ---
num_portfolios = 3000
results = []
weights_log = []

mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

for _ in range(num_portfolios):
    weights = np.random.random(len(price_df.columns))
    weights /= weights.sum()

    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility

    results.append([portfolio_volatility, portfolio_return, sharpe_ratio])
    weights_log.append(weights)

results_df = pd.DataFrame(results, columns=["Risk", "Return", "Sharpe"])

# Best portfolios
max_sharpe_idx = results_df["Sharpe"].idxmax()
min_risk_idx = results_df["Risk"].idxmin()

max_sharpe = results_df.loc[max_sharpe_idx]
min_risk = results_df.loc[min_risk_idx]

# --- Portfolio Weights Display ---
st.subheader("Optimized Portfolio Allocation")

col1, col2 = st.columns(2)

with col1:
    st.write("Maximum Sharpe Ratio Portfolio")
    st.dataframe(
        pd.Series(weights_log[max_sharpe_idx], index=price_df.columns)
        .sort_values(ascending=False)
    )

with col2:
    st.write("Minimum Variance Portfolio")
    st.dataframe(
        pd.Series(weights_log[min_risk_idx], index=price_df.columns)
        .sort_values(ascending=False)
    )

# --- Efficient Frontier Plot ---
st.subheader("Efficient Frontier")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(results_df["Risk"], results_df["Return"], alpha=0.4, s=10)
ax.scatter(max_sharpe["Risk"], max_sharpe["Return"], color="red", marker="*", s=200, label="Max Sharpe")
ax.scatter(min_risk["Risk"], min_risk["Return"], color="blue", marker="D", s=120, label="Min Variance")

ax.set_title("Risk vs Return")
ax.set_xlabel("Annualized Volatility")
ax.set_ylabel("Expected Annual Return")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# --- Heatmap ---
st.subheader("Correlation Map of Stock Returns")

fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.heatmap(returns.corr(), cmap="coolwarm", center=0, annot=False, ax=ax2)
st.pyplot(fig2)

st.success("Analysis completed.")
