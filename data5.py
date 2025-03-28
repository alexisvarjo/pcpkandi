import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""EX ANTE ANALYYSI"""

# Kaikissa skenaarioissa tiputetaan alle 10 volyymiset havainnot
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

def simulate_trade(data:pd.dataframe, divs:bool, fees:float, lag:bool):
    if not divs:
        data['x'] = data['x'] + data['PV_alldivs']
    data = data.query("call_v > 10 & put_v > 10 & ulying_volume > 0.01")
    data['error'] = data['y']-data['x']

    if not lag:
        data['profit'] = data['profit'].abs() - fees
        data = data[data['profit'] > 0]
        data['max_trade_count'] = data[['call_v', 'put_v', 'ulying_volume']].min(axis=1) * 0.1
        data['total_profit'] = data['profit'] * data['max_trade_count']
        data['capital_per_trade'] = data['x'].abs() + data['y'].abs()
        data['trade_count'] = data['max_trade_count'].astype(int)
        data = data[data['trade_count'] > 0].copy()
        data['returns'] = data['profit'] / data['capital_per_trade']
    else:
        data['error'] = data['y'] - data['x']
        data['x_next'] = data['x'].shift(-1)
        data['y_next'] = data['y'].shift(-1)
        data['lagged_profit'] = data.apply(lambda row: compute_lagged_profit(row, fees), axis=1)
        vol_columns = ['call_v', 'put_v', 'ulying_volume']
        data['max_lagged_trade_count'] = data[vol_columns].min(axis=1) * 0.1
        data.drop(columns=['x_next', 'y_next'], inplace=True)
        data['trade_count'] = data['max_lagged_trade_count'].astype(int).where(data['lagged_profit'] != 0, 0)
        data['total_profit'] = (data['lagged_profit'] * data['max_lagged_trade_count']).where(data['lagged_profit'] != 0)
        data['capital_per_trade'] = data['x'].abs() + data['y'].abs()
        data['returns'] = (data['lagged_profit'] / data['capital_per_trade']).where(data['lagged_profit'] != 0)
    return data

def wf(data, filename, country, id):
    with open(filename, 'a') as f:
        f.write(f"Country: {country}\n")
        f.write(f"Scenario {id}\n")
        f.write((data['total_profit'].describe()).to_string())
        f.write("\n")
        f.write(f"{id} total profit: {data['total_profit'].sum()}")
        f.write((data['returns'].describe()).to_string())
        f.write("\n")
        f.write("\n")

def wrapper(datas:list, low_fees:list, high_fees:list, countries:list):
    filename = "output.txt"
    for csv, low_fee, high_fee, country in zip(datas, low_fees, high_fees, countries):
        #1A: Agentti ei tiedä osinkoja, ei kuluja
        A1 = pd.read_csv(csv)
        A1 = A1.drop(columns=['Date', 'put_moneyness', 'call_moneyness',
            'IV_put', 'IV_call', 'country', 'underlying_return', 'underlying_log_return',
            'underlying_volatility'])
        A1 = simulate_trade(A1, False, 0.0, False)
        wf(A1, filename, country, "1A")

        #1B: Agentti tietää osingot, ei kuluja
        B1 = pd.read_csv(csv)
        B1 = B1.drop(columns=['Date', 'put_moneyness', 'call_moneyness',
            'IV_put', 'IV_call', 'country', 'underlying_return', 'underlying_log_return',
            'underlying_volatility'])
        B1 = simulate_trade(B1, True, 0.0, False)
        wf(B1, filename, country, "1B")

        # 2A: Agentti ei tiedä osinkoja, kuluja
        A2 = pd.read_csv(csv)
        A2 = A2.drop(columns=['Date', 'put_moneyness', 'call_moneyness',
            'IV_put', 'IV_call'])
        A2 = simulate_trade(A2, False, low_fee, False)
        wf(A2, filename, country, "2A")

        # 2B: Agentti tietää osingot, kuluja
        B2 = pd.read_csv(csv)

        B2 = B2.drop(columns=['Date', 'put_moneyness', 'call_moneyness',
            'IV_put', 'IV_call', 'country', 'underlying_return', 'underlying_log_return',
            'underlying_volatility'])
        B2 = simulate_trade(B2, False, low_fee, False)
        wf(B2, filename, country, "2B")

        # 3A: Agentti ei tiedä osinkoja, kuluilla ja lagilla
        A3 = pd.read_csv(csv)
        A3 = A3.drop(columns=['Date', 'put_moneyness', 'call_moneyness',
            'country', 'underlying_return', 'underlying_log_return',
            'underlying_volatility'])
        A3 = simulate_trade(A3, False, low_fee, True)
        wf(A3, filename, country, "3A")

        # 3B: Agentti tietää osingot, kuluilla ja lagilla
        B3 = pd.read_csv(csv)
        B3 = B3.drop(columns=['Date', 'put_moneyness', 'call_moneyness',
            'country', 'underlying_return', 'underlying_log_return',
            'underlying_volatility'])
        B3 = simulate_trade(B3, True, low_fee, True)
        wf(B3, filename, country, "3B")

        # 4A: Agentti ei tiedä osinkoja, isommat kulut
        # ----------------------------
        A4 = pd.read_csv(csv)
        A4 = A4.drop(columns=['Date', 'put_moneyness', 'call_moneyness',
            'IV_put', 'IV_call', 'country', 'underlying_return', 'underlying_log_return',
            'underlying_volatility'])
        A4 = simulate_trade(A4, False, high_fee, False)
        wf(A4, filename, country, "4A")

        # 4B: Agentti tietää osingot, isommat kulut
        B4 = pd.read_csv(csv)
        B4 = B4.drop(columns=['Date', 'put_moneyness', 'call_moneyness',
            'IV_put', 'IV_call', 'country','underlying_return', 'underlying_log_return',
            'underlying_volatility'])
        B4 = simulate_trade(B4, True, high_fee, False)
        wf(B4, filename, country, "4B")

        # 5A: Agentti ei tiedä osinkoja, isommat kulut ja lagilla
        A5 = pd.read_csv(csv)
        A5 = A5.drop(columns=['Date', 'put_moneyness', 'call_moneyness',
            'country', 'underlying_return', 'underlying_log_return',
            'underlying_volatility'])
        A5 = simulate_trade(A5, False, high_fee, True)
        wf(A5, filename, country, "5A")

        # 5B: agentti tietää osingot, isompi kulu ja lagi
        B5 = pd.read_csv(csv)
        B5 = B5.drop(columns=['Date', 'put_moneyness', 'call_moneyness',
            'country', 'underlying_return', 'underlying_log_return',
            'underlying_volatility'])
        B5 = simulate_trade(B5, True, high_fee, True)
        wf(B5, filename, country, "5B")

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


if __name__ == "__main__":
    datas_list = ['processed_data/dk_processed_data.csv','processed_data/no_processed_data.csv',
        'processed_data/se_processed_data.csv']
    low_fees = [20, 30, 20]
    high_fees = [40, 40, 40]
    countries = ['Denmark', 'Norway', 'Sweden']
    wrapper(datas_list, low_fees, high_fees, countries)



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
