import numpy as np
import pandas as pd
from xmlrpc.client import MAXINT
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

def american_option_price_fd(S0, K, T, r, sigma, Smax, M, N, option_type='put', omega=1.2, tol=1e-2, max_iter=10000):
    """
    Price an American option using a Crank-Nicolson finite difference scheme with PSOR.

    Parameters:
      S0         : current asset price (adjusted for dividends)
      K          : strike price
      T          : time to maturity (in years)
      r          : risk-free rate (annual)
      sigma      : volatility (annual)
      Smax       : maximum asset price in the grid
      M          : number of spatial grid steps
      N          : number of time steps
      option_type: 'put' or 'call'
      omega      : relaxation parameter for PSOR
      tol        : convergence tolerance for PSOR
      max_iter   : maximum iterations per time step

    Returns:
      Option price computed via the finite difference method.
    """
    # Spatial and time step sizes
    dS = Smax / M
    dt = T / N
    # Asset price grid
    S_values = np.linspace(0, Smax, M+1)


    # Terminal condition (option payoff at maturity)
    if option_type == 'put':
        payoff = np.maximum(K - S_values, 0)
        boundary_low = K      # At S=0, put value = K (intrinsic)
        boundary_high = 0     # At S=Smax, put value = 0
        min_value = np.maximum(K-S0, 0)
    else:
        payoff = np.maximum(S_values - K, 0)
        boundary_low = 0      # At S=0, call value = 0
        boundary_high = Smax - K  # At S=Smax, call value = Smax-K
        min_value = np.maximum(S0 - K, 0)

    # Initialize the solution grid: rows = time levels, columns = spatial nodes.
    V = np.zeros((N+1, M+1))
    V[-1, :] = payoff  # At maturity

    # Set boundary conditions for all time levels
    V[:, 0] = boundary_low
    V[:, -1] = boundary_high

    # Loop backward in time
    for n in reversed(range(N)):
        # Initialize the solution at time level n (using the solution from the next time level)
        V[n, :] = V[n+1, :]
        error = 1.0
        iter_count = 0

        # Solve using PSOR until convergence at this time step
        while error > tol and iter_count < max_iter:
            error = 0.0
            for i in range(1, M):
                S_i = i * dS
                # Coefficients for the Crank-Nicolson discretization at node i.
                # (These coefficients arise from averaging the explicit and implicit discretizations.)
                alpha = 0.25 * dt * (sigma**2 * (i**2) - r * i)
                beta  = -0.5 * dt * (sigma**2 * (i**2) + r)
                gamma = 0.25 * dt * (sigma**2 * (i**2) + r * i)

                # In the Crank-Nicolson scheme, the linear system for interior node i is:
                #   A * V_{i-1}^n + B * V_i^n + C * V_{i+1}^n = D,
                # where
                A = -alpha
                B = 1 - beta
                C = -gamma
                D = alpha * V[n+1, i-1] + (1 + beta) * V[n+1, i] + gamma * V[n+1, i+1]

                # PSOR update for V[n, i]
                V_old = V[n, i]
                V_new = (1 - omega) * V_old + (omega / B) * (D - A * V[n, i-1] - C * V[n, i+1])

                # Enforce the early exercise (free-boundary) condition
                intrinsic = (K - S_i) if option_type == 'put' else (S_i - K)
                V_new = max(V_new, intrinsic)

                error = max(error, abs(V_new - V_old))
                V[n, i] = V_new
            iter_count += 1
        # End PSOR iteration for time step n
    # Interpolate the option price at S0
    option_price = np.interp(S0, S_values, V[0, :])
    option_price = max(option_price, min_value)
    return option_price

def compute_early_exercise_premium_fd(row, M=100, N=100):
    """
    Compute the early exercise premium (EEP) using the finite difference method.

    The adjusted underlying is computed as:
         S_adj = ulying_price - PV_alldivs

    American option prices on S_adj are computed via FD.

    Then the EEP is given by:
         EEP = (A_call - A_put) - [S_adj - K*exp(-r*T)]

    Parameters:
      row : a row from your DataFrame containing the required columns.
      M   : number of spatial grid steps (default 100)
      N   : number of time steps (default 100)

    Returns:
      The early exercise premium as computed by the FD method.
    """
    S = row['ulying_price']
    K = row['strike']
    T = row['maturity']
    r = row['risk_free_rate']
    sigma = (row['IV_put'] + row['IV_call']) / 2  # you might use an average of IV_call and IV_put if desired
    PV_div = row['PV_alldivs']

    # Check for missing or invalid values
    if pd.isna(S) or pd.isna(K) or pd.isna(T) or pd.isna(r) or pd.isna(sigma) or pd.isna(PV_div):
        return np.nan

    # Adjust the underlying price for dividends
    S_adj = S - PV_div
    if S_adj <= 0:
        return np.nan

    # Define Smax for the finite difference grid.
    # Here we set Smax as a multiple of the adjusted underlying.
    Smax = max(2 * S_adj, K * 2)

    # Price American call and put options on the adjusted underlying using FD.
    A_call = american_option_price_fd(S_adj, K, T, r, sigma, Smax, M, N, option_type='call')
    A_put  = american_option_price_fd(S_adj, K, T, r, sigma, Smax, M, N, option_type='put')

    # The theoretical European put-call parity on the dividend-adjusted underlying is:
    euro_parity = S_adj - K * np.exp(-r * T)

    # The early exercise premium is the difference between the American parity difference and the European parity.
    EEP_fd = (A_call - A_put) - euro_parity
    return np.maximum(EEP_fd, 0)

def compute_eep_for_df(df, n_jobs=-1):
    total = len(df)
    # Wrap the Parallel call with tqdm_joblib to get a progress bar.
    with tqdm_joblib(tqdm(total=total, desc="Calculating EEP", unit="obs")):
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_early_exercise_premium_fd)(row) for _, row in df.iterrows()
        )
    df['EEP_fd'] = results
    return df

linreg_se = pd.read_csv('processed_data/se_processed_data.csv')
linreg_dk = pd.read_csv('processed_data/dk_processed_data.csv')
linreg_no = pd.read_csv('processed_data/no_processed_data.csv')

print("calculating eep for sweden")
linreg_se = compute_eep_for_df(linreg_se)
linreg_se['x'] = linreg_se['x'] + linreg_se['EEP_fd']
linreg_se.to_csv('processed_data/se_processed_data.csv', index=False)

print("calculating eep for denmark")
linreg_dk = compute_eep_for_df(linreg_dk)
linreg_dk['x'] = linreg_dk['x'] + linreg_dk['EEP_fd']
linreg_dk.to_csv('processed_data/dk_processed_data.csv', index=False)

print("calculating eep for norway")
linreg_no = compute_eep_for_df(linreg_no)
linreg_no['x'] = linreg_no['x'] + linreg_no['EEP_fd']
linreg_no.to_csv('processed_data/no_processed_data.csv', index=False)
