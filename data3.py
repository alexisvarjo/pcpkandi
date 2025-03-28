import pandas as pd
import numpy as np

# ------------------------------
# Exchange Rates Adjustments
# ------------------------------
#
linreg_dk = pd.read_csv("processed_data/dk_processed_data.csv", parse_dates=['Date'], index_col='Date')
linreg_no = pd.read_csv("processed_data/no_processed_data.csv", parse_dates=['Date'], index_col='Date')
linreg_se = pd.read_csv("processed_data/se_processed_data.csv", parse_dates=['Date'], index_col='Date')

linreg_dk = linreg_dk.rename(columns={'eep_call': 'EEP_call', 'eep_put': 'EEP_put'})
linreg_no = linreg_no.rename(columns={'eep_call': 'EEP_call', 'eep_put': 'EEP_put'})
linreg_se = linreg_se.rename(columns={'eep_call': 'EEP_call', 'eep_put': 'EEP_put'})

# Read and prepare exchange rates with dates parsed and set as index
exchange_rates = pd.read_csv("unprocessed_data/exchange_rates.csv", parse_dates=['Date'], index_col='Date')

# Reindex the exchange rate series to match the dates in the country-specific DataFrames
dk_exchange = exchange_rates['SEKDKK=X'].reindex(linreg_dk.index, method='ffill')
no_exchange = exchange_rates['SEKNOK=X'].reindex(linreg_no.index, method='ffill')

columns_to_multiply = ['y', 'x','PV_alldivs','strike','call_price', 'put_price', 'ulying_price',
    'EEP_call', 'EEP_put']
linreg_dk[columns_to_multiply] = linreg_dk[columns_to_multiply].multiply(dk_exchange, axis=0)
linreg_no[columns_to_multiply] = linreg_no[columns_to_multiply].multiply(no_exchange, axis=0)

# Combine all country data (Sweden remains unchanged)
linreg_gen = pd.concat([linreg_dk, linreg_se, linreg_no])
linreg_gen.sort_index(inplace=True)
# Reset index so that Date becomes a column
linreg_gen = linreg_gen.reset_index()


linreg_gen['Date'] = pd.to_datetime(linreg_gen['Date'])


# ------------------------------
# Save the Final DataFrame
# ------------------------------

linreg_gen.to_csv('processed_data/gen_processed_data.csv', index=False)
