import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

#------------------------------------------------
# 1. LOAD & CLEAN THE DATA
#------------------------------------------------

# Load data from CSV (update the path/filename as needed)
df = pd.read_csv("processed_data/gen_processed_data.csv")

# Drop rows with missing values (alternatively, you might impute)
df.dropna(inplace=True)

# For clarity, let’s assume these are our key raw variables:
#   - call_v: Call option price
#   - put_v: Put option price
#   - spot: Underlying asset price (spot)
#   - strike: Strike price of the options
#   - rf: Risk‐free rate (annualized)
#   - maturity: Time to maturity (in years)
#
# If your data has additional columns (volumes, greeks, etc.), note that many are
# derived from these inputs. Our analysis will concentrate on the “essentials.”

#------------------------------------------------
# 2. CALCULATE PUT–CALL PARITY VARIABLES
#------------------------------------------------

# The theoretical put–call parity is:
#   call - put = spot - strike * exp(-rf * maturity)
# We define:
#   x_calc = call price - put price
#   y_calc = spot - strike * exp(-rf * maturity)
#   parity_error = (call - put) - (spot - strike * exp(-rf * maturity))

df['parity_error'] = df['y'] - df['x']
df['IV'] = (df['IV_put'] + df['IV_call']) / 2
df['opt_volume'] = (df['put_v'] + df['call_v'])/2
df['moneyness'] = (df['call_moneyness'] + df['put_moneyness']) / 2

# It is expected that under no-arbitrage conditions, parity_error ~ 0.
print("Summary of calculated variables:")
print(df[['x', 'y', 'parity_error']].describe())

#------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
#------------------------------------------------

# Histogram of the parity error with a kernel density estimate
plt.figure(figsize=(10, 6))
sns.histplot(df['parity_error'], kde=True, color='skyblue', bins=30)
plt.title("Distribution of Put–Call Parity Error")
plt.xlabel("Parity Error (y-x)")
plt.ylabel("Frequency")
#plt.show()

# QQ-plot to assess normality of the parity error distribution
sm.qqplot(df['parity_error'], line='45', fit=True)
plt.title("QQ-Plot of Parity Error")
#plt.show()

# Test if the mean parity error is statistically different from zero
t_stat, p_val = stats.ttest_1samp(df['parity_error'], 0)
print(f"T-test for mean parity error = 0: t-stat = {t_stat:.3f}, p-value = {p_val:.3e}")

#------------------------------------------------
# 4. REGRESSION ANALYSIS ON PARITY ERROR
#------------------------------------------------

# Even though put–call parity should hold exactly,
# in practice small deviations occur. We might want to see if these deviations
# are systematically related to other variables. For example, we can check whether
# time to maturity or (if available) implied volatility (IV) affects the parity error.
#
# (Note: Volume and many other derived measures are not directly used in put–call parity.)
#
# Here we assume you may have an implied volatility column named "IV". If not, the regression
# below will use only 'maturity'. Adjust or add other predictors as you see fit.

if 'IV_put' in df.columns:
    formula = "parity_error ~ maturity + IV"
else:
    formula = "parity_error ~ maturity"

model = smf.ols(formula, data=df).fit()
print("\nRegression results on parity error:")
print(model.summary())

# Diagnostic plot: Parity error vs. maturity (and IV if available)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='maturity', y='parity_error', data=df, color='darkorange')
plt.title("Parity Error vs. Time to Maturity")
plt.xlabel("Maturity (years)")
plt.ylabel("Parity Error")
#plt.show()

if 'IV' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='IV', y='parity_error', data=df, color='green')
    plt.title("Parity Error vs. Implied Volatility")
    plt.xlabel("Implied Volatility")
    plt.ylabel("Parity Error")
    #plt.show()

# Additional regression diagnostics (e.g., residual plots)
fig = sm.graphics.plot_regress_exog(model, "maturity")
fig.tight_layout(pad=1.5)
#plt.show()

#------------------------------------------------
# 5. CONCLUSIONS & FURTHER CONSIDERATIONS
#------------------------------------------------

# The analysis above focuses on the core put–call parity relation.
# In an ideal market with no arbitrage, parity_error should be indistinguishable from zero.
# Any significant deviation might be due to transaction costs, market frictions, or data issues.
#
# In your thesis, you could:
#   - Examine if the deviations are systematically related to market conditions (e.g., during high volatility periods).
#   - Explore the effect of maturity (and possibly other predictors) on the observed deviation.
#   - Discuss the practical limitations and the role of bid–ask spreads or other microstructure effects.
#
# Remember, many variables in your dataset are derived directly from the basic market inputs.
# Hence, focusing your analysis on the key raw variables (and their immediate combinations, like x_calc and y_calc)
# provides a clearer picture of the put–call parity relationship.

if 'opt_volume' in df.columns:
    formula = "parity_error ~ opt_volume"

model = smf.ols(formula, data=df).fit()
print("\nRegression results on parity error:")
print(model.summary())

# Diagnostic plot: Parity error vs. option trade volume
plt.figure(figsize=(10, 6))
sns.scatterplot(x='opt_volume', y='parity_error', data=df, color='darkorange')
plt.title("Parity Error vs. opt_volume")
plt.xlabel("Option Volume")
plt.ylabel("Parity Error")
#plt.show()


#parity error vs underlying volatility
if 'underlying_volatility' in df.columns:
    formula = "parity_error ~ underlying_volatility"

model = smf.ols(formula, data=df).fit()
print("\nRegression results on parity error:")
print(model.summary())

# Diagnostic plot: Parity error vs. option trade volume
plt.figure(figsize=(10, 6))
sns.scatterplot(x='underlying_volatility', y='parity_error', data=df, color='darkorange')
plt.title("Parity Error vs. Underlying Volume")
plt.xlabel("Underlying Volume")
plt.ylabel("Parity Error")
#plt.show()

#parity error vs underlying returns
if 'underlying_return' in df.columns:
    formula = "parity_error ~ underlying_return"

model = smf.ols(formula, data=df).fit()
print("\nRegression results on parity error:")
print(model.summary())

# Diagnostic plot: Parity error vs. option trade volume
plt.figure(figsize=(10, 6))
sns.scatterplot(x='underlying_return', y='parity_error', data=df, color='darkorange')
plt.title("Parity Error vs. Underlying Returns")
plt.xlabel("Underlying Returns")
plt.ylabel("Parity Error")
#plt.show()

#parity error vs underlying log returns
if 'underlying_log_return' in df.columns:
    formula = "parity_error ~ underlying_log_return"

model = smf.ols(formula, data=df).fit()
print("\nRegression results on parity error:")
print(model.summary())

# Diagnostic plot: Parity error vs. log returns
plt.figure(figsize=(10, 6))
sns.scatterplot(x='underlying_log_return', y='parity_error', data=df, color='darkorange')
plt.title("Parity Error vs. Underlying Log Returns")
plt.xlabel("Underlying Log Returns")
plt.ylabel("Parity Error")
#plt.show()

#parity error vs moneyness
if 'moneyness' in df.columns:
    formula = "parity_error ~ moneyness"

model = smf.ols(formula, data=df).fit()
print("\nRegression results on parity error:")

print(model.summary())
plt.figure(figsize=(10, 6))
sns.scatterplot(x='moneyness', y='parity_error', data=df, color='darkorange')
plt.title("Parity Error vs. moneyness")
plt.xlabel("Moneyness")
plt.ylabel("Parity Error")
#plt.show()

#parity error vs risk free rate

if 'risk_free_rate' in df.columns:
    formula = "parity_error ~ risk_free_rate"

model = smf.ols(formula, data=df).fit()
print("\nRegression results on parity error:")
print(model.summary())

plt.figure(figsize=(10, 6))
sns.scatterplot(x='risk_free_rate', y='parity_error', data=df, color='darkorange')
plt.title("Parity Error vs. Risk Free Rate")
plt.xlabel("Risk Free Rate")
plt.ylabel("Parity Error")
#plt.show()

# Create a correlation matrix for parity error and these variables
cols_for_corr = ['parity_error'] + ['moneyness', 'risk_free_rate', 'IV', 'maturity', 'opt_volume', 'underlying_log_return', 'underlying_return', 'strike', 'ulying_illiquidity', 'call_illiquidity', 'put_illiquidity']
corr_matrix = df[cols_for_corr].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix: Parity Error & Additional Variables")
plt.show()
print("\nCorrelation matrix:\n", corr_matrix)

#parity error vs strike
formula = "parity_error ~ strike"

model = smf.ols(formula, data=df).fit()
print("\nRegression results on parity error:")
print(model.summary())

# Diagnostic plot: Parity error vs. Strike
plt.figure(figsize=(10, 6))
sns.scatterplot(x='strike', y='parity_error', data=df, color='darkorange')
plt.title("Parity Error vs. Strike")
plt.xlabel("Strike")
plt.ylabel("Parity Error")
#plt.show()

#parity error vs all raw variables
formula = "parity_error ~ risk_free_rate + IV + underlying_volatility + maturity + opt_volume + underlying_log_return + strike"
model = smf.ols(formula, data=df).fit()
print("\nRegression results on parity error:")
print(model.summary())

#parity error vs all raw variables
formula = "parity_error ~ risk_free_rate + IV + maturity + opt_volume + underlying_log_return + strike"
model = smf.ols(formula, data=df).fit()
print("\nRegression results on parity error:")
print(model.summary())

#parity error vs all raw variables
formula = "parity_error ~ risk_free_rate + IV + maturity + opt_volume + strike"
model = smf.ols(formula, data=df).fit()
print("\nRegression results on parity error:")
print(model.summary())

#parity error vs all raw variables
formula = "parity_error ~ risk_free_rate + IV + maturity + strike"
model = smf.ols(formula, data=df).fit()
print("\nRegression results on parity error:")
print(model.summary())

formula = "parity_error ~ ulying_illiquidity + call_illiquidity + put_illiquidity"
model = smf.ols(formula, data=df).fit()
print("\nRegression results on parity error:")
print(model.summary())

formula = "parity_error ~ strike + ulying_illiquidity + call_illiquidity + put_illiquidity"
model = smf.ols(formula, data=df).fit()
print("\nRegression results on parity error:")
print(model.summary())
