import pandas as pd
import numpy as np

"""EX ANTE ANALYYSI"""

# Kaikissa skenaarioissa tiputetaan alle 10 volyymiset havainnot
# Koska niitä ei voi treidata


#1A: Agentti ei tiedä osinkoja, ei kuluja
#1B: agentti tietää osingot, ei kuluja
#2A: agentti tietää osingot, kuluja
#2B: Agentti ei tiedä osinkoja, kuluja
#3A: Agentti tietää osingot, lagilla ja kuluja
#3B: Agentti ei tiedä osinkoja, lagilla ja kuluja

# Ex ante analyysissä: epäsuorienkulujen arvioiminen on vaikeaa (bid ask spread)
# Mutta ATM optioilla spreadin pitäisi olla pienin
# Jos silläkään ei onnistu saamaan arbitraasia, niin ei varmaan kauempanakaan ATM:stä onnistu
# LAG -> immediacy risk
# Likviditeettiriski: likviditeetin vaihtelu. Jos likviditeetti pysyy samana
# niin treidaaja pystyy arvioimaan tulevaisuuden epäsuoria transaktiokuluja

def compute_lagged_profit(row, fees):
    x_t = row['x']
    y_t = row['y']
    x_next = row['x_next']
    y_next = row['y_next']

    # Trigger only if the absolute difference exceeds fees
    if abs(y_t - x_t) <= fees:
        return 0.0

    if (y_t - x_t) > 0:
        # Strategy: long x at time t
        # Option 1: Complete hedge by shorting y next day:
        profit_hedge = (y_next - x_t) - fees
        # Option 2: Exit long position by selling x next day:
        profit_close = (x_next - x_t) - fees
    else:  # y_t - x_t < 0
        # Strategy: long y at time t
        # Option 1: Complete hedge by shorting x next day:
        profit_hedge = (y_t - x_next) - fees
        # Option 2: Exit long position by selling y next day:
        profit_close = (y_next - y_t) - fees

    # Choose the alternative with the higher profit, if positive; otherwise, no trade.
    return max(profit_hedge, profit_close, 0.0)



#1A agentti tietää osingot, ei kuluja
A1 = pd.read_csv('processed_data/gen_processed_data.csv')

A1 = A1.drop(columns=['Date', 'put_moneyness', 'call_moneyness',
    'put_log_moneyness', 'call_log_moneyness', 'call_tv', 'put_tv',
    'Ivalue_put', 'Ivalue_call', 'IV_put', 'IV_call', 'put_delta', 'call_delta',
    'gamma', 'vega', 'call_theta', 'put_theta', 'vega', 'call_rho',
    'put_rho', 'country', 'underlying_return', 'underlying_log_return',
    'underlying_volatility'])

A1['error'] = A1['y'] - A1['x']

A1 = A1.query("call_v > 10 & put_v > 10 & ulying_volume > 0.01")
A1 = A1.reset_index(drop=True)

A1['profit'] = A1['error'].abs()
A1['max_trade_count'] = A1[['call_v', 'put_v', 'ulying_volume']].min(axis=1)*0.1
A1['total_profit'] = A1['profit']*A1['max_trade_count']

A1['capital_per_trade'] = A1['x'].abs() + A1['y'].abs()
# Calculate returns for each trade scenario (profit relative to capital required)
A1['returns'] = A1['profit'] / A1['capital_per_trade']

print(A1['total_profit'].describe())
print("A1 total profit:", A1['total_profit'].sum())
print(A1['returns'].describe())

# 1B Agentti ei tiedä osinkoja, ei kuluja
B1 = pd.read_csv('processed_data/gen_processed_data.csv')

B1 = B1.drop(columns=['Date', 'put_moneyness', 'call_moneyness',
    'put_log_moneyness', 'call_log_moneyness', 'call_tv', 'put_tv',
    'Ivalue_put', 'Ivalue_call', 'IV_put', 'IV_call', 'put_delta', 'call_delta',
    'gamma', 'vega', 'call_theta', 'put_theta', 'vega', 'call_rho',
    'put_rho', 'country', 'underlying_return', 'underlying_log_return',
    'underlying_volatility'])

B1['x'] = B1['x'] + B1['PV_alldivs']
B1['error'] = B1['y'] - B1['x']

B1 = B1.query("call_v > 10 & put_v > 10 & ulying_volume > 0.01")
B1 = B1.reset_index(drop=True)

B1['profit'] = B1['error'].abs()
B1['max_trade_count'] = B1[['call_v', 'put_v', 'ulying_volume']].min(axis=1)*0.1
B1['total_profit'] = B1['profit']*B1['max_trade_count']

B1['capital_per_trade'] = B1['x'].abs() + B1['y'].abs()
# Calculate returns for each trade scenario (profit relative to capital required)
B1['returns'] = B1['profit'] / B1['capital_per_trade']

print(B1['total_profit'].describe())
print("B1 total profit:", B1['total_profit'].sum())
print(B1['returns'].describe())

###2A: agentti tietää osingot, kuluja
A2 = pd.read_csv('processed_data/gen_processed_data.csv')
direct_fees = 15 #SEKeissä

A2 = A2.drop(columns=['Date', 'put_moneyness', 'call_moneyness',
    'put_log_moneyness', 'call_log_moneyness', 'call_tv', 'put_tv',
    'Ivalue_put', 'Ivalue_call', 'IV_put', 'IV_call', 'put_delta', 'call_delta',
    'gamma', 'vega', 'call_theta', 'put_theta', 'vega', 'call_rho',
    'put_rho', 'country', 'underlying_return', 'underlying_log_return',
    'underlying_volatility'])

A2['error'] = A2['y'] - A2['x']

A2 = A2.query("call_v > 10 & put_v > 10 & ulying_volume > 0.01")
A2 = A2.query("error.abs() > 15") #direct fees

A2 = A2.reset_index(drop=True)

A2['profit'] = A2['error'].abs()
A2['max_trade_count'] = A2[['call_v', 'put_v', 'ulying_volume']].min(axis=1)*0.1
A2['total_profit'] = A2['profit']*A2['max_trade_count']

A2['capital_per_trade'] = A2['x'].abs() + A2['y'].abs()
# Calculate returns for each trade scenario (profit relative to capital required)
A2['returns'] = A2['profit'] / A2['capital_per_trade']

print(A2['total_profit'].describe())
print("A2 total profit:", A2['total_profit'].sum())
print(A2['returns'].describe())

##2B ei tiedä osinkoja, kuluja
B2 = pd.read_csv('processed_data/gen_processed_data.csv')
direct_fees = 15 #SEKeissä

B2 = B2.drop(columns=['Date', 'put_moneyness', 'call_moneyness',
    'put_log_moneyness', 'call_log_moneyness', 'call_tv', 'put_tv',
    'Ivalue_put', 'Ivalue_call', 'IV_put', 'IV_call', 'put_delta', 'call_delta',
    'gamma', 'vega', 'call_theta', 'put_theta', 'vega', 'call_rho',
    'put_rho', 'country', 'underlying_return', 'underlying_log_return',
    'underlying_volatility'])

B2['x'] = B2['x'] + B2['PV_alldivs']
B2['error'] = B2['y'] - B2['x']

B2 = B2.query("call_v > 10 & put_v > 10 & ulying_volume > 0.01")
A2 = A2.query("error.abs() > 15")
B2 = B2.reset_index(drop=True)

B2['profit'] = B2['error'].abs()
B2['max_trade_count'] = B2[['call_v', 'put_v', 'ulying_volume']].min(axis=1)*0.1
B2['total_profit'] = B2['profit']*B2['max_trade_count']

B2['capital_per_trade'] = B2['x'].abs() + B2['y'].abs()
# Calculate returns for each trade scenario (profit relative to capital required)
B2['returns'] = B2['profit'] / B2['capital_per_trade']

print(B2['total_profit'].describe())
print("B2 total profit:", B2['total_profit'].sum())
print(B2['returns'].describe())

#3A tietää osingot, kuluilla ja lagilla
A3 = pd.read_csv('processed_data/gen_processed_data.csv')
direct_fees = 15 #SEKeissä

A3 = A3.drop(columns=['Date', 'put_moneyness', 'call_moneyness',
    'put_log_moneyness', 'call_log_moneyness', 'call_tv', 'put_tv',
    'Ivalue_put', 'Ivalue_call', 'IV_put', 'IV_call', 'put_delta', 'call_delta',
    'gamma', 'vega', 'call_theta', 'put_theta', 'vega', 'call_rho',
    'put_rho', 'country', 'underlying_return', 'underlying_log_return',
    'underlying_volatility'])

A3['error'] = A3['y'] - A3['x']
A3['x_next'] = A3['x'].shift(-1)
A3['y_next'] = A3['y'].shift(-1)

# Apply the function row-wise:
A3['lagged_profit'] = A3.apply(lambda row: compute_lagged_profit(row, direct_fees), axis=1)

# Optionally, if you still want to compute a "max lagged trade count" (e.g., a trade size limiter),
# you could use your original idea (here, taking 10% of the minimum volume across the three volume columns)
vol_columns = ['call_v', 'put_v', 'ulying_volume']
A3['max_lagged_trade_count'] = A3[vol_columns].min(axis=1) * 0.1

# Optionally drop the shifted price columns if you no longer need them
A3.drop(columns=['x_next', 'y_next'], inplace=True)

A3['total_profit'] = A3['lagged_profit']*A3['max_lagged_trade_count']

A3['capital_per_trade'] = A3['x'].abs() + A3['y'].abs()
# Calculate returns for each trade scenario (profit relative to capital required)
A3['returns'] = A3['lagged_profit'] / A3['capital_per_trade']

print(A3['total_profit'].describe())
print("A3 total profit:", A3['total_profit'].sum())
print(A3['returns'].describe())

#3B ei tiedä osinkoja, kuluilla ja lagilla
B3 = pd.read_csv('processed_data/gen_processed_data.csv')
direct_fees = 15 #SEKeissä

B3 = B3.drop(columns=['Date', 'put_moneyness', 'call_moneyness',
    'put_log_moneyness', 'call_log_moneyness', 'call_tv', 'put_tv',
    'Ivalue_put', 'Ivalue_call', 'IV_put', 'IV_call', 'put_delta', 'call_delta',
    'gamma', 'vega', 'call_theta', 'put_theta', 'vega', 'call_rho',
    'put_rho', 'country', 'underlying_return', 'underlying_log_return',
    'underlying_volatility'])
B3['x'] = B3['x'] + B3['PV_alldivs']

B3['error'] = B3['y'] - B3['x']
B3['x_next'] = B3['x'].shift(-1)
B3['y_next'] = B3['y'].shift(-1)

# Apply the function row-wise:
B3['lagged_profit'] = B3.apply(lambda row: compute_lagged_profit(row, direct_fees), axis=1)

# Optionally, if you still want to compute a "max lagged trade count" (e.g., a trade size limiter),
# you could use your original idea (here, taking 10% of the minimum volume across the three volume columns)
vol_columns = ['call_v', 'put_v', 'ulying_volume']
B3['max_lagged_trade_count'] = B3[vol_columns].min(axis=1) * 0.1

# Optionally drop the shifted price columns if you no longer need them
B3.drop(columns=['x_next', 'y_next'], inplace=True)

B3['total_profit'] = B3['lagged_profit']*B3['max_lagged_trade_count']

B3['capital_per_trade'] = B3['x'].abs() + B3['y'].abs()
# Calculate returns for each trade scenario (profit relative to capital required)
B3['returns'] = B3['lagged_profit'] / B3['capital_per_trade']

print(B3['total_profit'].describe())
print("B3 total profit:", B3['total_profit'].sum())
print(B3['returns'].describe())

### ex POST analyysit loppuu
###
