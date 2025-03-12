import yfinance as yf
import pandas as pd

# Define the currency pairs
currency_pairs = ['SEKNOK=X', 'SEKDKK=X']

# Define the start and end dates
start_date = '2010-01-01'
end_date = '2025-02-09'

# Download the historical data
data = yf.download(currency_pairs, start=start_date, end=end_date)

# Extract the 'Adj Close' prices
exchange_rates = data['Close']

# Display the data

# Save the data to a CSV file
exchange_rates.to_csv('exchange_rates.csv')
