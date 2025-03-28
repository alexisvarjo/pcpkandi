import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed

# ------------------------------
# Read Data and Prepare Options
# ------------------------------

# Read risk-free rates if needed
rates_o = pd.read_csv("unprocessed_data/risk_free_rates2.csv", parse_dates=['Date'], index_col='Date')
# Read options data; note that the index is set and then converted to datetime
options = pd.read_csv("unprocessed_data/kovadata3.csv", header=[0, 1, 2])
options = options.set_index('Date')
# The index contains tuples; we take the first element and convert to datetime
options.index = options.index.map(lambda x: x[0])
options.index = pd.to_datetime(options.index, format='%m/%d/%y')

# Initialize empty DataFrames for each country.
linreg_dk = pd.DataFrame()
linreg_se = pd.DataFrame()
linreg_no = pd.DataFrame()

# ------------------------------
# Helper Functions
# ------------------------------

def safe_clip(x, lower, upper=None):
    if hasattr(x, "clip"):
        if upper is not None:
            return x.clip(lower, upper)
        else:
            return x.clip(lower, None)
    else:
        if upper is not None:
            return np.clip(x, lower, upper)
        else:
            return np.clip(x, lower, None)

def bs_d1(S, K, T, rf, sigma):
    epsilon = 1e-10
    S = safe_clip(S, epsilon)
    K = safe_clip(K, epsilon)
    sigma = safe_clip(sigma, epsilon)
    T = safe_clip(T, epsilon)
    log_term = np.log(S / K)
    denom = sigma * np.sqrt(T)
    d1 = (log_term + (rf + sigma**2 / 2) * T) / denom
    if hasattr(d1, "replace"):
        d1.replace([np.inf, -np.inf], np.nan, inplace=True)
    else:
        d1 = np.where(np.isinf(d1), np.nan, d1)
    return d1

def bs_d2(S, K, T, rf, sigma):
    d1 = bs_d1(S, K, T, rf, sigma)
    return d1 - sigma * np.sqrt(T)

def vega(S, K, R, T, sigma):
    d1 = bs_d1(S, K, T, R, sigma)
    # Vega per 1 percentage point change in volatility
    return S/100 * np.exp(-R*T) * np.sqrt(T) * norm.pdf(d1)

def get_iv(S, K, R, T, mktprice, call: bool):
    epsilon = 1e-10
    max_iv = 5
    min_iv = 1e-4
    iv_series = pd.Series(index=S.index, dtype=float)
    for idx in S.index:
        s, k, r, t, mp = S[idx], K[idx], R[idx], T[idx], mktprice[idx]
        if s <= 0 or k <= 0 or t <= 0 or mp <= 0:
            iv_series[idx] = np.nan
            continue
        sign = 1 if call else -1
        iv = 0.01
        prev_iv = 0
        tol = 1e-6
        for _ in range(100):
            d1 = bs_d1(s, k, t, r, iv)
            d2 = d1 - iv * np.sqrt(t)
            model_price = sign * (s * np.exp(-r * t) * norm.cdf(sign * d1) - k * np.exp(-r * t) * norm.cdf(sign * d2))
            vega_val = vega(s, k, r, t, iv) * 100
            if vega_val < epsilon:
                break
            iv -= (model_price - mp) / vega_val
            iv = np.clip(iv, min_iv, max_iv)
            if abs(iv - prev_iv) < tol:
                break
            prev_iv = iv
        iv_series[idx] = iv
    return iv_series

def get_rate_for_maturity(row_rates, maturity, mapping):
    """
    For a given maturity (in days), attempt to retrieve the rate from row_rates using the mapping.
    If the desired column is missing or NaN, search upward (and then downward) for the nearest available rate.
    """
    col = mapping[maturity]
    rate = row_rates.get(col, np.nan)
    if not pd.isna(rate):
        return rate
    # Search upward in the sorted maturities:
    sorted_keys = sorted(mapping.keys())
    idx = sorted_keys.index(maturity)
    for j in range(idx + 1, len(sorted_keys)):
        candidate = sorted_keys[j]
        candidate_rate = row_rates.get(mapping[candidate], np.nan)
        if not pd.isna(candidate_rate):
            return candidate_rate
    # If still not found, search downward:
    for j in range(idx - 1, -1, -1):
        candidate = sorted_keys[j]
        candidate_rate = row_rates.get(mapping[candidate], np.nan)
        if not pd.isna(candidate_rate):
            return candidate_rate
    return np.nan

def get_interpolated_rate(row_rates, T_days, mapping):
    """
    Interpolates the risk-free rate based on time to dividend T_days (in days)
    using linear interpolation between the two bracketing maturities in the mapping.
    """
    maturities = sorted(mapping.keys())

    def get_rate(maturity):
        return get_rate_for_maturity(row_rates, maturity, mapping)

    if T_days <= maturities[0]:
        return get_rate(maturities[0])
    elif T_days >= maturities[-1]:
        return get_rate(maturities[-1])
    else:
        for i in range(1, len(maturities)):
            if T_days <= maturities[i]:
                m_lower = maturities[i - 1]
                m_upper = maturities[i]
                break
        r_lower = get_rate(m_lower)
        r_upper = get_rate(m_upper)
        # If one rate is missing, use the other as fallback.
        if pd.isna(r_lower) and not pd.isna(r_upper):
            r_lower = r_upper
        if pd.isna(r_upper) and not pd.isna(r_lower):
            r_upper = r_lower
        if pd.isna(r_lower) and pd.isna(r_upper):
            return np.nan
        frac = (T_days - m_lower) / (m_upper - m_lower)
        return r_lower + (r_upper - r_lower) * frac

def calculate_pv_alldivs(df, rates_df, country):
    """
    For each pricing day in df, compute the present value of all future dividends that occur
    before option expiration (given by the 'maturity' column in years).

    Each future dividend (where df["ulying_div"] is nonzero) is discounted using a risk-free rate
    that is linearly interpolated from rates_df based on the time (in days) until that dividend.

    Parameters:
      - df: DataFrame with a Date (or index convertible to a Date column),
            'ulying_div' (dividend amount on dividend payment days), and
            'maturity' (option time-to-expiration in years).
      - rates_df: DataFrame with risk-free rate observations by Date. It should have columns
                  for various maturities (as in your CSV).
      - country: String identifier ("NORWAY", "SWEDEN", or "DENMARK") to select the appropriate mapping.

    Returns:
      - The original DataFrame with an added 'PV_alldivs' column.
    """
    # Ensure the DataFrame has a "Date" column.
    if "Date" not in df.columns:
        df = df.reset_index()

    df = df.sort_values("Date").copy()

    # Define the mapping dictionary for the given country.
    if country.upper() == "NORWAY":
        mapping = {
            1: "NOKONZ=R",
            7: "OINOKSWD=",
            30: "OINOK1MD=",
            60: "OINOK2MD=",
            90: "OINOK3MD=",
            180: "OINOK6MD=",
            270: "NOK9MZ=R",
            365: "NOK1YZ=R",
            455: "NOK1Y3MZ=R"
        }
    elif country.upper() == "SWEDEN":
        mapping = {
            1: "STISEKTNDFI=",
            7: "STISEK1WDFI=",
            30: "STISEK1MDFI=",
            60: "STISEK2MDFI=",
            90: "STISEK3MDFI=",
            180: "STISEK6MDFI=",
            270: "SEK9MZ=R",
            365: "SEGOV1YZ=R",
            455: "SEGOV1Y3MZ=R"
        }
    elif country.upper() == "DENMARK":
        mapping = {
            1: "DKKONZ=R",
            7: "CIDKKSWD=",
            30: "CIDKK1MD=",
            60: "DKK2MZ=R",
            90: "DKK9MZ=R",
            180: "CIDKK6MD=",
            270: "DKK9MZ=R",
            365: "CIDKK1YD=",
            455: "DKKABQCD1Y3MZ=R"
        }
    else:
        raise ValueError("Country not recognized. Use 'NORWAY', 'SWEDEN', or 'DENMARK'.")

    # Extract dividend payment days (rows with nonzero dividend amounts).
    divs = df[df["ulying_div"] != 0][["Date", "ulying_div"]].copy()

    # Sort the risk-free rates DataFrame by Date.
    rates_df = (rates_df.sort_index())/100
    pv_alldivs_list = []

    # Loop over each pricing day.
    for i, pricing_row in df.iterrows():
        pricing_date = pricing_row["Date"]
        pv_sum = 0.0

        # Determine option expiration (in days) from the 'maturity' column.
        T_expiry = pricing_row["maturity"] * 365.0

        # Select dividend events that occur after the pricing date.
        future_divs = divs[divs["Date"] > pricing_date]
        for j, div_row in future_divs.iterrows():
            div_date = div_row["Date"]
            dividend = div_row["ulying_div"]
            T_days = (div_date - pricing_date).days

            # Only include dividends that are paid before option expiration.
            if T_days >= T_expiry:
                continue
            # Retrieve the rates row using an "asof" lookup.
            asof_date = rates_df.index.asof(pricing_date)
            if pd.isna(asof_date):
                continue
            row_rates = rates_df.loc[asof_date]

            # Interpolate the risk-free rate for the dividend's time to payment.
            r_div = get_interpolated_rate(row_rates, T_days, mapping)
            if pd.isna(r_div):
                continue

            # Discount the dividend using continuous compounding.
            discount_factor = np.exp(-r_div * (T_days / 365.0))
            pv_sum += dividend * discount_factor

        pv_alldivs_list.append(pv_sum)

    df["PV_alldivs"] = pv_alldivs_list
    return df


def get_risk_free_rate(maturities, country, dates):
    rates = pd.read_csv('unprocessed_data/risk_free_rates2.csv', parse_dates=['Date'], index_col='Date')
    if country == "NORWAY":
        mapping = {
            1: "NOKONZ=R",
            7: "OINOKSWD=",
            30: "OINOK1MD=",
            60: "OINOK2MD=",
            90: "OINOK3MD=",
            180: "OINOK6MD=",
            270: "NOK9MZ=R",
            365: "NOK1YZ=R",
            455: "NOK1Y3MZ=R"
        }
    elif country == "SWEDEN":
        mapping = {
            1: "STISEKTNDFI=",
            7: "STISEK1WDFI=",
            30: "STISEK1MDFI=",
            60: "STISEK2MDFI=",
            90: "STISEK3MDFI=",
            180: "STISEK6MDFI=",
            270: "SEK9MZ=R",
            365: "SEGOV1YZ=R",
            455: "SEGOV1Y3MZ=R"
        }
    elif country == "DENMARK":
        mapping = {
            1: "DKKONZ=R",
            7: "CIDKKSWD=",
            30: "CIDKK1MD=",
            60: "DKK2MZ=R",
            90: 'DKK9MZ=R',
            180: "CIDKK6MD=",
            270: "DKK9MZ=R",
            365: "CIDKK1YD=",
            455: "DKKABQCD1Y3MZ=R"
        }
    else:
        print("country:", country)
        raise ValueError("Country not recognized")

    dates_norm = dates.normalize()
    rates = rates.reindex(dates_norm, method='ffill')
    days = maturities * 365
    interp_rates = []
    for d, day in zip(dates_norm, days):
        if day < 7:
            keys = (mapping[1], mapping[7])
            x_points = [0, 1]
        elif day < 30:
            keys = (mapping[7], mapping[30])
            x_points = [7, 30]
        elif day < 60:
            keys = (mapping[30], mapping[60])
            x_points = [30, 60]
        elif day < 90:
            keys = (mapping[60], mapping[90])
            x_points = [60, 90]
        elif day < 180:
            keys = (mapping[90], mapping[180])
            x_points = [90, 180]
        elif day < 270:
            keys = (mapping[180], mapping[270])
            x_points = [180, 270]
        elif day < 365:
            keys = (mapping[270], mapping[365])
            x_points = [270, 365]
        elif day < 455:
            keys = (mapping[365], mapping[455])
            x_points = [365, 455]
        else:
            print("maturity of ", day, "too long")
            raise ValueError("Maturity too long")
        try:
            r1 = rates.at[d, keys[0]]
            r2 = rates.at[d, keys[1]]
        except KeyError:
            r1, r2 = np.nan, np.nan
        interp_rate = np.interp(day, x_points, [r1, r2])
        interp_rates.append(interp_rate)
    interp_rates_series = pd.Series(interp_rates, index=dates_norm)
    return interp_rates_series / 100.0

# ------------------------------
# Main Processing Loop
# ------------------------------

def process_option_group(i, options, rates_o):
    # Process one option group (columns i to i+8)
    nonzero_indices = [i+1, i+2, i+3, i+4, i+5, i+7]
    mask_zeros = (options.iloc[:, nonzero_indices] != 0).all(axis=1)
    mask_nas = (options.iloc[:, nonzero_indices].notna()).all(axis=1)
    combined_mask = mask_zeros & mask_nas
    filtered_options = options[combined_mask]

    ulying_div = filtered_options.iloc[:, i]
    ulying_volume = filtered_options.iloc[:, i+1] * 1000
    maturity = filtered_options.iloc[:, i+2] / 365
    strike = filtered_options.iloc[:, i+3]
    call_price = filtered_options.iloc[:, i+4]
    ulying_price = filtered_options.iloc[:, i+5]
    call_v = filtered_options.iloc[:, i+6]
    put_price = filtered_options.iloc[:, i+7]
    put_v = filtered_options.iloc[:, i+8]

    # Get country identifier from column information and add it as a new column.
    country = options.columns[i][2]
    rates = get_risk_free_rate(maturity, country, filtered_options.index)
    call_moneyness = ulying_price / strike
    put_moneyness = strike / ulying_price
    IV_put = get_iv(ulying_price, strike, rates, maturity, put_price, False)
    IV_call = get_iv(ulying_price, strike, rates, maturity, call_price, True)
    eksp = -rates * maturity
    new_y = call_price - put_price
    new_x = ulying_price - (strike * np.exp(eksp))

    # Build the regression DataFrame using the filtered options index (date)
    reg_data = pd.DataFrame({
        'y': new_y,
        'x': new_x,
        'call_v': call_v,
        'put_v': put_v,
        'ulying_div': ulying_div,
        'ulying_volume': ulying_volume,
        'strike': strike,
        'maturity': maturity,
        'call_price': call_price,
        'put_price': put_price,
        'ulying_price': ulying_price,
        'risk_free_rate': rates,
        'put_moneyness': put_moneyness,
        'call_moneyness': call_moneyness,
        'IV_put': IV_put,
        'IV_call': IV_call,
    }, index=filtered_options.index)
    reg_data.index.name = 'Date'
    # Add the country identifier
    reg_data['country'] = country
    print("getting all divs", ulying_price.name)

    reg_data = calculate_pv_alldivs(reg_data, rates_o, country)

    reg_data['x'] = reg_data['x']-reg_data['PV_alldivs']

    return reg_data

group_indices = list(range(0, len(options.columns), 9))
results = Parallel(n_jobs=-1)(
    delayed(process_option_group)(i, options, rates_o) for i in group_indices
)

# Separate by country (filter out empty DataFrames, if any)
linreg_dk = pd.concat([df for df in results if not df.empty and df['country'].iloc[0] == "DENMARK"])
linreg_se = pd.concat([df for df in results if not df.empty and df['country'].iloc[0] == "SWEDEN"])
linreg_no = pd.concat([df for df in results if not df.empty and df['country'].iloc[0] == "NORWAY"])


linreg_dk.to_csv('processed_data/dk_processed_data.csv', index=False)
linreg_se.to_csv('processed_data/se_processed_data.csv', index=False)
linreg_no.to_csv('processed_data/no_processed_data.csv', index=False)
