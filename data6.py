import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""EX ANTE ANALYYSI"""

# Kaikissa skenaarioissa tiputetaan alle 10 volyymiset havainnot
# Koska niitä ei voi treidata


#1A: Agentti ei tiedä osinkoja, ei kuluja
#1B: agentti tietää osingot, ei kuluja
#2A: agentti ei tiedä osingot, kuluja
#2B: Agentti tietää osinkoja, kuluja
#3A: Agentti ei tiedä osingot, lagilla ja kuluja
#3B: Agentti tietää osinkoja, lagilla ja kuluja
#4A: agentti ei tiedä osinkoja, isommat kulut
#4B: agentti tietää osingot, isommat kulut
#5A: agentti ei tiedä osinkoja, isommat kulut ja lagilla
#5B: agentti tietää osingot, isommat kulut ja lagilla
data = 'processed_data/no_processed_data.csv'

def compute_lagged_profit(row, fees):
    x_t = row['x']
    y_t = row['y']
    x_next = row['x_next']
    y_next = row['y_next']

    # Trigger only if the absolute difference exceeds fees
    if abs(y_t - x_t) <= fees:
        return 0

    if (y_t - x_t) > 0:
        # Strategy: long x at time t
        # Option 1: Complete hedge by shorting y next day:
        profit_hedge = (y_next - x_t) - fees
        # Option 2: Exit long position by selling x next day:
        profit_close = (x_next - x_t) - fees
    else:  # y_t - x_t < 0
        # Strategy: long y at time t
        # Option 1: Complete hedge by shorting x next day:
        profit_hedge = (x_next - y_t) - fees
        # Option 2: Exit long position by selling y next day:
        profit_close = (y_next - y_t) - fees

    # Choose the alternative with the higher profit, if positive; otherwise, no trade.
    return max(profit_hedge, profit_close)

# ----------------------------
# 1A: Agentti ei tiedä osinkoja, ei kuluja
# ----------------------------
A1 = pd.read_csv(data)
A1 = A1.drop(columns=['Date', 'put_moneyness', 'call_moneyness',
    'put_log_moneyness', 'call_log_moneyness', 'call_tv', 'put_tv',
    'Ivalue_put', 'Ivalue_call', 'IV_put', 'IV_call', 'put_delta', 'call_delta',
    'gamma', 'vega', 'call_theta', 'put_theta', 'vega', 'call_rho',
    'put_rho', 'country', 'underlying_return', 'underlying_log_return',
    'underlying_volatility'])
A1['x'] = A1['x'] + A1['PV_alldivs']
A1['error'] = A1['y'] - A1['x']

A1 = A1.query("call_v > 10 & put_v > 10 & ulying_volume > 0.01")
A1 = A1.reset_index(drop=True)

A1['profit'] = A1['error'].abs()
threshold = 0
obs_count = A1[A1['error'].abs() > threshold].shape[0]
print("A1 count of observations with abs(error) > {}: {}".format(threshold, obs_count))
A1['max_trade_count'] = A1[['call_v', 'put_v', 'ulying_volume']].min(axis=1) * 0.1
A1['total_profit'] = A1['profit'] * A1['max_trade_count']
A1['capital_per_trade'] = A1['x'].abs() + A1['y'].abs()

A1['trade_count'] = A1['max_trade_count'].astype(int)
# Compute returns only on rows where a trade occurs (error.abs() > threshold)
trades_A1 = A1[A1['trade_count'] > 0].copy()
trades_A1['returns'] = trades_A1['profit'] / trades_A1['capital_per_trade']

print(A1['total_profit'].describe())
print("A1 total profit:", A1['total_profit'].sum())
print(trades_A1['returns'].describe())

# ----------------------------
# 1B: Agentti tietää osingot, ei kuluja
# ----------------------------
B1 = pd.read_csv(data)
B1 = B1.drop(columns=['Date', 'put_moneyness', 'call_moneyness',
    'put_log_moneyness', 'call_log_moneyness', 'call_tv', 'put_tv',
    'Ivalue_put', 'Ivalue_call', 'IV_put', 'IV_call', 'put_delta', 'call_delta',
    'gamma', 'vega', 'call_theta', 'put_theta', 'vega', 'call_rho',
    'put_rho', 'country', 'underlying_return', 'underlying_log_return',
    'underlying_volatility'])

B1['error'] = B1['y'] - B1['x']
B1 = B1.query("call_v > 10 & put_v > 10 & ulying_volume > 0.01")
B1 = B1.reset_index(drop=True)

B1['profit'] = B1['error'].abs()
B1['max_trade_count'] = B1[['call_v', 'put_v', 'ulying_volume']].min(axis=1) * 0.1
B1['total_profit'] = B1['profit'] * B1['max_trade_count']
B1['trade_count'] = B1['max_trade_count'].astype(int)

threshold = 0
obs_count = B1[B1['error'].abs() > threshold].shape[0]
print("B1 count of observations with abs(error) > {}: {}".format(threshold, obs_count))
B1['capital_per_trade'] = B1['x'].abs() + B1['y'].abs()

trades_B1 = B1[B1['trade_count'] > 0].copy()
trades_B1['returns'] = trades_B1['profit'] / trades_B1['capital_per_trade']

print(B1['total_profit'].describe())
print("B1 total profit:", B1['total_profit'].sum())
print(trades_B1['returns'].describe())

# ----------------------------
# 2A: Agentti ei tiedä osinkoja, kuluja
# ----------------------------
A2 = pd.read_csv(data)
direct_fees = 31.36  # SEKeissä

A2 = A2.drop(columns=['Date', 'put_moneyness', 'call_moneyness',
    'put_log_moneyness', 'call_log_moneyness', 'call_tv', 'put_tv',
    'Ivalue_put', 'Ivalue_call', 'IV_put', 'IV_call', 'put_delta', 'call_delta',
    'gamma', 'vega', 'call_theta', 'put_theta', 'vega', 'call_rho',
    'put_rho', 'country', 'underlying_return', 'underlying_log_return',
    'underlying_volatility'])
A2['x'] = A2['x'] + A2['PV_alldivs']
A2['error'] = A2['y'] - A2['x']

A2 = A2.query("call_v > 10 & put_v > 10 & ulying_volume > 0.01")
A2 = A2.query("error.abs() > 31.36")  # direct fees
A2 = A2.reset_index(drop=True)

A2['profit'] = A2['error'].abs() - direct_fees
A2['max_trade_count'] = A2[['call_v', 'put_v', 'ulying_volume']].min(axis=1) * 0.1
A2['trade_count'] = A2['max_trade_count'].astype(int).where(A2['error'].abs() > direct_fees, 0)
A2['total_profit'] = A2['profit'] * A2['trade_count']
threshold = direct_fees
obs_count = A2[A2['profit'] > threshold].shape[0]
print("A2 count of observations with abs(error) > {}: {}".format(threshold, obs_count))

A2['capital_per_trade'] = A2['x'].abs() + A2['y'].abs()
# A2 is already filtered to trades so we compute returns directly
trades_A2 = A2[A2['trade_count'] > 0]
trades_A2['returns'] = trades_A2['total_profit'] / trades_A2['capital_per_trade']

print(A2['total_profit'].describe())
print("A2 total profit:", A2['total_profit'].sum())
print(trades_A2['returns'].describe())

# ----------------------------
# 2B: Agentti tietää osingot, kuluja
# ----------------------------
B2 = pd.read_csv(data)
direct_fees = 31.36  # SEKeissä

B2 = B2.drop(columns=['Date', 'put_moneyness', 'call_moneyness',
    'put_log_moneyness', 'call_log_moneyness', 'call_tv', 'put_tv',
    'Ivalue_put', 'Ivalue_call', 'IV_put', 'IV_call', 'put_delta', 'call_delta',
    'gamma', 'vega', 'call_theta', 'put_theta', 'vega', 'call_rho',
    'put_rho', 'country', 'underlying_return', 'underlying_log_return',
    'underlying_volatility'])
B2['error'] = B2['y'] - B2['x']

B2 = B2.query("call_v > 10 & put_v > 10 & ulying_volume > 0.01")
B2 = B2.query("error.abs() > 31.36")
B2 = B2.reset_index(drop=True)

B2['profit'] = B2['error'].abs() - direct_fees
B2['max_trade_count'] = B2[['call_v', 'put_v', 'ulying_volume']].min(axis=1) * 0.1
B2['total_profit'] = B2['profit'] * B2['max_trade_count']
B2['trade_count'] = B2['max_trade_count'].astype(int).where(B2['profit'] > 0, 0)

threshold = direct_fees
obs_count = B2[B2['profit'] > threshold].shape[0]
print("B2 count of observations with abs(error) > {}: {}".format(threshold, obs_count))

B2['capital_per_trade'] = B2['x'].abs() + B2['y'].abs()
trades_B2 = B2[B2['trade_count'] > 0].copy()
trades_B2['returns'] = trades_B2['profit'] / trades_B2['capital_per_trade']
print(B2['total_profit'].describe())
print("B2 total profit:", B2['total_profit'].sum())
print(trades_B2['returns'].describe())

# ----------------------------
# 3A: Agentti ei tiedä osinkoja, kuluilla ja lagilla
# ----------------------------
A3 = pd.read_csv(data)
direct_fees = 31.36  # SEKeissä

A3 = A3.drop(columns=['Date', 'put_moneyness', 'call_moneyness',
    'put_log_moneyness', 'call_log_moneyness', 'call_tv', 'put_tv',
    'Ivalue_put', 'Ivalue_call', 'IV_put', 'IV_call', 'put_delta', 'call_delta',
    'gamma', 'vega', 'call_theta', 'put_theta', 'vega', 'call_rho',
    'put_rho', 'country', 'underlying_return', 'underlying_log_return',
    'underlying_volatility'])
A3['x'] = A3['x'] + A3['PV_alldivs']
A3['error'] = A3['y'] - A3['x']
A3['x_next'] = A3['x'].shift(-1)
A3['y_next'] = A3['y'].shift(-1)

# Apply the lagged profit function row-wise:
A3['lagged_profit'] = A3.apply(lambda row: compute_lagged_profit(row, direct_fees), axis=1)

vol_columns = ['call_v', 'put_v', 'ulying_volume']
A3['max_lagged_trade_count'] = A3[vol_columns].min(axis=1) * 0.1
A3.drop(columns=['x_next', 'y_next'], inplace=True)

A3['total_profit'] = A3['lagged_profit'] * A3['max_lagged_trade_count']
A3['capital_per_trade'] = A3['x'].abs() + A3['y'].abs()
A3['returns'] = (A3['lagged_profit'] / A3['capital_per_trade']).where(A3['lagged_profit'] != 0)
threshold = direct_fees
obs_count = A3[A3['error'].abs() > threshold].shape[0]
print("A3 count of observations with abs(error) > {}: {}".format(threshold, obs_count))
print(A3['total_profit'].describe())
print("A3 total profit:", A3['total_profit'].sum())
print(A3['returns'].describe())

# ----------------------------
# 3B: Agentti tietää osingot, kuluilla ja lagilla
# ----------------------------
B3 = pd.read_csv(data)
direct_fees = 31.36  # SEKeissä

B3 = B3.drop(columns=['Date', 'put_moneyness', 'call_moneyness',
    'put_log_moneyness', 'call_log_moneyness', 'call_tv', 'put_tv',
    'Ivalue_put', 'Ivalue_call', 'IV_put', 'IV_call', 'put_delta', 'call_delta',
    'gamma', 'vega', 'call_theta', 'put_theta', 'vega', 'call_rho',
    'put_rho', 'country', 'underlying_return', 'underlying_log_return',
    'underlying_volatility'])
B3['error'] = B3['y'] - B3['x']
B3['x_next'] = B3['x'].shift(-1)
B3['y_next'] = B3['y'].shift(-1)

B3['lagged_profit'] = B3.apply(lambda row: compute_lagged_profit(row, direct_fees), axis=1)

vol_columns = ['call_v', 'put_v', 'ulying_volume']
B3['max_lagged_trade_count'] = B3[vol_columns].min(axis=1) * 0.1

B3.drop(columns=['x_next', 'y_next'], inplace=True)

B3['total_profit'] = B3['lagged_profit'] * B3['max_lagged_trade_count']
B3['capital_per_trade'] = B3['x'].abs() + B3['y'].abs()
B3['returns'] = (B3['lagged_profit'] / B3['capital_per_trade']).where(B3['lagged_profit'] != 0)

threshold = direct_fees
obs_count = B3[B3['error'].abs() > threshold].shape[0]
print("B3 count of observations with abs(error) > {}: {}".format(threshold, obs_count))
print(B3['total_profit'].describe())
print("B3 total profit:", B3['total_profit'].sum())
print(B3['returns'].describe())

# ----------------------------
# 4A: Agentti ei tiedä osinkoja, isommat kulut
# ----------------------------
A4 = pd.read_csv(data)
direct_fees = 62.73  # SEKeissä

A4 = A4.drop(columns=['Date', 'put_moneyness', 'call_moneyness',
    'put_log_moneyness', 'call_log_moneyness', 'call_tv', 'put_tv',
    'Ivalue_put', 'Ivalue_call', 'IV_put', 'IV_call', 'put_delta', 'call_delta',
    'gamma', 'vega', 'call_theta', 'put_theta', 'vega', 'call_rho',
    'put_rho', 'country', 'underlying_return', 'underlying_log_return',
    'underlying_volatility'])
A4['x'] = A4['x'] + A4['PV_alldivs']
A4['error'] = A4['y'] - A4['x']

A4 = A4.query("call_v > 10 & put_v > 10 & ulying_volume > 0.01")
A4 = A4.query("error.abs() > 62.73")  # direct fees
A4 = A4.reset_index(drop=True)

A4['profit'] = A4['error'].abs() - direct_fees
A4['max_trade_count'] = A4[['call_v', 'put_v', 'ulying_volume']].min(axis=1) * 0.1
A4['trades'] = A4['max_trade_count'].astype(int).where(A4['max_trade_count'] > 0, 0)
A4['total_profit'] = A4['profit'] * A4['max_trade_count']
A4['capital_per_trade'] = A4['x'].abs() + A4['y'].abs()
trades_A4 = A4.copy()
trades_A4['returns'] = (trades_A4['profit'] / trades_A4['capital_per_trade']).where(A4['trades'] > 0)
threshold = direct_fees
obs_count = A4[A4['error'].abs() > threshold].shape[0]
print("A4 count of observations with abs(error) > {}: {}".format(threshold, obs_count))
print(A4['total_profit'].describe())
print("A4 total profit:", A4['total_profit'].sum())
print(trades_A4['returns'].describe())

# ----------------------------
# 4B: Agentti tietää osingot, isommat kulut
# ----------------------------
B4 = pd.read_csv(data)
direct_fees = 62.73  # SEKeissä

B4 = B4.drop(columns=['Date', 'put_moneyness', 'call_moneyness',
    'put_log_moneyness', 'call_log_moneyness', 'call_tv', 'put_tv',
    'Ivalue_put', 'Ivalue_call', 'IV_put', 'IV_call', 'put_delta', 'call_delta',
    'gamma', 'vega', 'call_theta', 'put_theta', 'vega', 'call_rho',
    'put_rho', 'country', 'underlying_return', 'underlying_log_return',
    'underlying_volatility'])

B4['error'] = B4['y'] - B4['x']

B4 = B4.query("call_v > 10 & put_v > 10 & ulying_volume > 0.01")
B4 = B4.query("error.abs() > 62.73")
B4 = B4.reset_index(drop=True)

B4['profit'] = B4['error'].abs() - direct_fees
B4['max_trade_count'] = B4[['call_v', 'put_v', 'ulying_volume']].min(axis=1) * 0.1
B4['trades'] = (B4['profit'] > 0) * B4['max_trade_count']
B4['total_profit'] = B4['profit'] * B4['max_trade_count']
B4['capital_per_trade'] = B4['x'].abs() + B4['y'].abs()
trades_B4 = B4.copy()
trades_B4['returns'] = (trades_B4['profit'] / trades_B4['capital_per_trade']).where(B4['trades'] > 0)
threshold = direct_fees
obs_count = B4[B4['error'].abs() > threshold].shape[0]
print("B4 count of observations with abs(error) > {}: {}".format(threshold, obs_count))
print(B4['total_profit'].describe())
print("B4 total profit:", B4['total_profit'].sum())
print(trades_B4['returns'].describe())

# ----------------------------
# 5A: Agentti ei tiedä osinkoja, isommat kulut ja lagilla
# ----------------------------
A5 = pd.read_csv(data)
direct_fees = 62.73  # SEKeissä

A5 = A5.drop(columns=['Date', 'put_moneyness', 'call_moneyness',
    'put_log_moneyness', 'call_log_moneyness', 'call_tv', 'put_tv',
    'Ivalue_put', 'Ivalue_call', 'IV_put', 'IV_call', 'put_delta', 'call_delta',
    'gamma', 'vega', 'call_theta', 'put_theta', 'vega', 'call_rho',
    'put_rho', 'country', 'underlying_return', 'underlying_log_return',
    'underlying_volatility'])
A5['x'] = A5['x'] + A5['PV_alldivs']
A5['error'] = A5['y'] - A5['x']
A5['x_next'] = A5['x'].shift(-1)
A5['y_next'] = A5['y'].shift(-1)

A5['lagged_profit'] = A5.apply(lambda row: compute_lagged_profit(row, direct_fees), axis=1)

vol_columns = ['call_v', 'put_v', 'ulying_volume']
A5['max_lagged_trade_count'] = A5[vol_columns].min(axis=1) * 0.1

A5.drop(columns=['x_next', 'y_next'], inplace=True)

A5['total_profit'] = A5['lagged_profit'] * A5['max_lagged_trade_count'] - direct_fees
A5['capital_per_trade'] = A5['x'].abs() + A5['y'].abs()
A5['returns'] = (A5['lagged_profit'] / A5['capital_per_trade']).where(A5['lagged_profit'] != 0)
threshold = direct_fees
obs_count = A5[A5['error'].abs() > threshold].shape[0]
print("A5 count of observations with abs(error) > {}: {}".format(threshold, obs_count))
print(A5['total_profit'].describe())
print("A5 total profit:", A5['total_profit'].sum())
print(A5['returns'].describe())

# ----------------------------
# 5B: Agentti tietää osingot, isommat kulut ja lagilla
# ----------------------------
B5 = pd.read_csv(data)
direct_fees = 62.73  # SEKeissä

B5 = B5.drop(columns=['Date', 'put_moneyness', 'call_moneyness',
    'put_log_moneyness', 'call_log_moneyness', 'call_tv', 'put_tv',
    'Ivalue_put', 'Ivalue_call', 'IV_put', 'IV_call', 'put_delta', 'call_delta',
    'gamma', 'vega', 'call_theta', 'put_theta', 'vega', 'call_rho',
    'put_rho', 'country', 'underlying_return', 'underlying_log_return',
    'underlying_volatility'])

B5['error'] = B5['y'] - B5['x']
B5['x_next'] = B5['x'].shift(-1)
B5['y_next'] = B5['y'].shift(-1)

B5['lagged_profit'] = B5.apply(lambda row: compute_lagged_profit(row, direct_fees), axis=1)

vol_columns = ['call_v', 'put_v', 'ulying_volume']
B5['max_lagged_trade_count'] = B5[vol_columns].min(axis=1) * 0.1

B5.drop(columns=['x_next', 'y_next'], inplace=True)

B5['total_profit'] = B5['lagged_profit'] * B5['max_lagged_trade_count']
B5['capital_per_trade'] = B5['x'].abs() + B5['y'].abs()
B5['returns'] = (B5['lagged_profit'] / B5['capital_per_trade']).where(B5['lagged_profit'] != 0)
threshold = direct_fees
obs_count = B5[B5['error'].abs() > threshold].shape[0]
print("B5 count of observations with abs(error) > {}: {}".format(threshold, obs_count))
print(B5['total_profit'].describe())
print("B5 total profit:", B5['total_profit'].sum())
print(B5['returns'].describe())


# ----------------------------
# ex POST analyysit loppuu
# ----------------------------
hist_df = pd.read_csv(data, parse_dates=['Date'])

# Filter observations where both call and put volumes exceed 10
hist_df_valid = hist_df[(hist_df["call_v"] > 10) & (hist_df["put_v"] > 10)].copy()

# Define the transaction cost parameter (fee) to check against violations
fee = 30

# Compute error and violation (PCP violation if absolute error exceeds fee)
hist_df_valid["error"] = hist_df_valid["y"] - hist_df_valid["x"]
hist_df_valid["violation"] = hist_df_valid["error"].abs() > fee

# Create a new column for year-month extraction from Date for monthly grouping
hist_df_valid["year_month"] = hist_df_valid["Date"].dt.to_period("M").astype(str)

# -------------------------------
# Compute Arbitrage Opportunity Columns
# -------------------------------
max_trading_volume = 0.1  # 10% of the smaller volume per observation
hist_df_valid["min_vol"] = hist_df_valid[["call_v", "put_v"]].min(axis=1)
hist_df_valid["max_trading_vol"] = hist_df_valid["min_vol"] * max_trading_volume
hist_df_valid["vol_witherror"] = hist_df_valid["min_vol"] * hist_df_valid["violation"]
# If a violation occurs, arbitrage volume equals max_trading_vol; otherwise, 0.
hist_df_valid["arbitrage_vol"] = hist_df_valid["max_trading_vol"] * hist_df_valid["violation"].astype(int)

grouped_month = hist_df_valid.groupby(["year_month", "country"]).agg(
    total_obs=("min_vol", "sum"),
    violations=("violation", "count")
).reset_index()
grouped_month["portion"] = grouped_month["violations"] / grouped_month["total_obs"]
countries = grouped_month["country"].unique()

monthly_arbitrage = hist_df_valid.groupby(["year_month", "country"]).agg(
    total_possible=("min_vol", "sum"),
    available=("arbitrage_vol", "sum")
).reset_index()
monthly_arbitrage["percentage_available"] = monthly_arbitrage["available"] / monthly_arbitrage["total_possible"]

for c in countries:
    subset = monthly_arbitrage[monthly_arbitrage["country"] == c]
    tick_positions = np.arange(0, len(subset), 12)
    tick_labels = [str(2011+i) for i in range(len(tick_positions))]
    plt.figure(figsize=(10,6))
    plt.xticks(tick_positions, tick_labels)
    plt.bar(range(len(subset)), subset["percentage_available"])
    new_labels = [ym.split('-')[0] for ym in subset["year_month"]]
    plt.title(f"How much of monthly volume has potential for arbitrage in {c} when fee is {fee}")
    plt.xlabel("Year")
    plt.ylabel("Percentage of Available Volume")
    plt.tight_layout()
    #plt.savefig("käyrät/trading/1_" + c + ".png", dpi=600)
    #plt.show()


# -------------------------------
# SECTION 2: Monthly Arbitrage Opportunities (fee = 60)
# -------------------------------
fee = 60
hist_df_valid["violation"] = hist_df_valid["error"].abs() > fee
hist_df_valid["vol_witherror"] = hist_df_valid["min_vol"] * hist_df_valid["violation"]
hist_df_valid["arbitrage_vol"] = hist_df_valid["max_trading_vol"] * hist_df_valid["violation"].astype(int)

grouped_month = hist_df_valid.groupby(["year_month", "country"]).agg(
    total_obs=("min_vol", "sum"),
    violations=("violation", "count")
).reset_index()
grouped_month["portion"] = grouped_month["violations"] / grouped_month["total_obs"]
countries = grouped_month["country"].unique()

monthly_arbitrage = hist_df_valid.groupby(["year_month", "country"]).agg(
    total_possible=("min_vol", "sum"),
    available=("arbitrage_vol", "sum")
).reset_index()
monthly_arbitrage["percentage_available"] = monthly_arbitrage["available"] / monthly_arbitrage["total_possible"]

for c in countries:
    subset = monthly_arbitrage[monthly_arbitrage["country"] == c]
    tick_positions = np.arange(0, len(subset), 12)
    tick_labels = [str(2011+i) for i in range(len(tick_positions))]
    plt.figure(figsize=(10,6))
    plt.xticks(tick_positions, tick_labels)
    plt.bar(range(len(subset)), subset["percentage_available"])
    new_labels = [ym.split('-')[0] for ym in subset["year_month"]]
    plt.title(f"How much of monthly volume has potential for arbitrage in {c} when fee is {fee}")
    plt.xlabel("Year")
    plt.ylabel("Percentage of Available Volume")
    plt.tight_layout()
    #plt.savefig("käyrät/trading/2_" + c + ".png", dpi=600)
    #plt.show()
