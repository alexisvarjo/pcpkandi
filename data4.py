import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

def drop_low_volume(df, min_vol):
    df = df[df['call_v'] >= min_vol]
    df = df[df['put_v'] >= min_vol]
    df = df[df['ulying_volume'] >= min_vol]
    return df

def restrict_interestrate(df, min_interestrate):
    df = df[df['risk_free_rate'] >= min_interestrate]
    return df

def dropilliquid(df, max_illiquidity):
    df = df[df['ulying_illiquidity'] <= max_illiquidity]
    df = df[df['call_illiquidity'] <= max_illiquidity]
    df = df[df['put_illiquidity'] <= max_illiquidity]
    #df = df.dropna(subset=['ulying_illiquidity', 'call_illiquidity', 'put_illiquidity'])
    return df

linreg_dk = pd.read_csv('processed_data/dk_processed_data.csv')
linreg_se = pd.read_csv('processed_data/se_processed_data.csv')
linreg_no = pd.read_csv('processed_data/no_processed_data.csv')
linreg_gen = pd.read_csv('processed_data/gen_processed_data.csv')


min_vol = 10

linreg_dk = drop_low_volume(linreg_dk, min_vol)
linreg_se = drop_low_volume(linreg_se, min_vol)
linreg_no = drop_low_volume(linreg_no, min_vol)
linreg_gen = drop_low_volume(linreg_gen, min_vol)

flag = False
max_illiquidity = 0.15 #0,01 on pienin järkevä (tippuu liikaa sampleja pois sen alle),
#0.15 on suurin järkevä ja tiputtaa about 10 pros kaikista havainnoista
#ILLIQUID LASKUTOIMITUS ON VÄÄRIN TOISTAISEKSI!!!

if flag:
    linreg_dk = dropilliquid(linreg_dk, max_illiquidity)
    linreg_se = dropilliquid(linreg_se, max_illiquidity)
    linreg_no = dropilliquid(linreg_no, max_illiquidity)
    linreg_gen = dropilliquid(linreg_gen, max_illiquidity)

count = 5

if not linreg_dk.empty:
    linreg_dk = linreg_dk.dropna(subset=['x', 'y'])

    print("dk linreg:")

    X_dk = pd.to_numeric(linreg_dk['x'], errors='coerce')
    y_dk = pd.to_numeric(linreg_dk['y'], errors='coerce')

    X_dk = sm.add_constant(X_dk)

    model_dk = sm.OLS(y_dk, X_dk).fit()

    print(model_dk.summary())

    plt.figure(figsize=(6,4))
    plt.tight_layout()
    plt.scatter(linreg_dk['x'], linreg_dk['y'], label='Data')
    x_dk_sorted = np.sort(linreg_dk['x'])
    y_dk_pred = model_dk.params[0] + model_dk.params[1] * x_dk_sorted
    plt.plot(x_dk_sorted, y_dk_pred, color='red', label='Fit')
    plt.plot(x_dk_sorted, x_dk_sorted, '--', color='gray', label='y = x')
    plt.title("Denmark")
    plt.xlabel("Stock position value, DKK")
    plt.ylabel("Synthetic stock position value, DKK")
    plt.legend()
    #plt.show()
    plt.savefig(f"käyrät/plot{count}.png", dpi=600)
    plt.close()
    count += 1

if not linreg_se.empty:
    linreg_se = linreg_se.dropna(subset=['x', 'y'])
    linreg_se['x'] = linreg_se['x']

    print("se linreg")
    X_se = pd.to_numeric(linreg_se['x'], errors='coerce')
    y_se = pd.to_numeric(linreg_se['y'], errors='coerce')

    X_se = sm.add_constant(X_se)

    model_se = sm.OLS(y_se, X_se).fit()

    print(model_se.summary())

    plt.figure(figsize=(6,4))
    plt.tight_layout()
    plt.scatter(linreg_se['x'], linreg_se['y'], label='Data')
    x_se_sorted = np.sort(linreg_se['x'])
    y_se_pred = model_se.params[0] + model_se.params[1] * x_se_sorted
    plt.plot(x_se_sorted, y_se_pred, color='red', label='Fit')
    plt.plot(x_se_sorted, x_se_sorted, '--', color='gray', label='y = x')
    plt.title("Sweden")
    plt.xlabel("Stock position value, SEK")
    plt.ylabel("Synthetic stock position value, SEK")
    plt.legend()
    #plt.show()
    plt.savefig(f"käyrät/plot{count}.png", dpi=600)
    plt.close()
    count += 1


if not linreg_no.empty:
    linreg_no = linreg_no.dropna(subset=['x', 'y'])

    print("no linreg")
    X_no = pd.to_numeric(linreg_no['x'], errors='coerce')
    y_no = pd.to_numeric(linreg_no['y'], errors='coerce')

    X_no = sm.add_constant(X_no)

    model_no = sm.OLS(y_no, X_no).fit()

    print(model_no.summary())

    plt.figure(figsize=(6,4))
    plt.tight_layout()
    plt.scatter(linreg_no['x'], linreg_no['y'], label='Data')
    x_no_sorted = np.sort(linreg_no['x'])
    y_no_pred = model_no.params[0] + model_no.params[1] * x_no_sorted
    plt.plot(x_no_sorted, y_no_pred, color='red', label='Fit')
    plt.plot(x_no_sorted, x_no_sorted, '--', color='gray', label='y = x')
    plt.title("Norway")
    plt.xlabel("Stock position value, NOK")
    plt.ylabel("Synthetic stock position value, NOK")
    plt.legend()
    #plt.show()
    plt.savefig(f"käyrät/plot{count}.png", dpi=600)
    plt.close()
    count += 1


X_gen = pd.to_numeric(linreg_gen['x'].dropna(), errors='coerce')
y_gen = pd.to_numeric(linreg_gen['y'].dropna(), errors='coerce')

X_gen = sm.add_constant(X_gen)

model_gen = sm.OLS(y_gen, X_gen).fit()

print(model_gen.summary())

plt.figure(figsize=(6,4))
plt.tight_layout()
plt.scatter(linreg_gen['x'], linreg_gen['y'], label='Data')
x_gen_sorted = np.sort(linreg_gen['x'])
y_gen_pred = model_gen.params[0] + model_gen.params[1] * x_gen_sorted
plt.plot(x_gen_sorted, y_gen_pred, color='green', label='Fit')
plt.plot(x_gen_sorted, x_gen_sorted, '--', color='gray', label='y = x')
plt.title("General Linear Regression")
plt.xlabel("Stock position value, SEK")
plt.ylabel("Synthetic stock position value, SEK")
plt.legend()
#plt.show()
plt.savefig(f"käyrät/plot{count}.png", dpi=600)
plt.close()
count += 1


df = linreg_no
df['parity_error'] = df['y'] - df['x']
df['IV'] = (df['IV_put'] + df['IV_call']) / 2
df['opt_volume'] = (df['put_v'] + df['call_v'])/2
df['moneyness'] = (df['call_moneyness'] + df['put_moneyness']) / 2
#parity error vs all raw variables
formula = "parity_error ~ risk_free_rate + IV + underlying_volatility + maturity + opt_volume + underlying_log_return + strike"
model = smf.ols(formula, data=df).fit()
print("\nnorway Regression results on parity error:")
print(model.summary())

#parity error vs all raw variables
formula = "parity_error ~ risk_free_rate + IV + maturity + opt_volume + underlying_log_return + strike"
model = smf.ols(formula, data=df).fit()
print("\nnorway Regression results on parity error:")
print(model.summary())

#parity error vs all raw variables
formula = "parity_error ~ risk_free_rate + IV + maturity + opt_volume + strike"
model = smf.ols(formula, data=df).fit()
print("\nnorway Regression results on parity error:")
print(model.summary())

#parity error vs all raw variables
formula = "parity_error ~ risk_free_rate + IV + maturity + strike"
model = smf.ols(formula, data=df).fit()
print("\n norway Regression results on parity error:")
print(model.summary())

formula = "parity_error ~ ulying_illiquidity + call_illiquidity + put_illiquidity"
model = smf.ols(formula, data=df).fit()
print("\nnorway Regression results on parity error:")
print(model.summary())

formula = "parity_error ~ strike + ulying_illiquidity + call_illiquidity + put_illiquidity"
model = smf.ols(formula, data=df).fit()
print("\nnorway Regression results on parity error:")
print(model.summary())

df = linreg_se
df['parity_error'] = df['y'] - df['x']
df['IV'] = (df['IV_put'] + df['IV_call']) / 2
df['opt_volume'] = (df['put_v'] + df['call_v'])/2
df['moneyness'] = (df['call_moneyness'] + df['put_moneyness']) / 2

#parity error vs all raw variables
formula = "parity_error ~ risk_free_rate + IV + underlying_volatility + maturity + opt_volume + underlying_log_return + strike"
model = smf.ols(formula, data=df).fit()
print("\nsweden Regression results on parity error:")
print(model.summary())

#parity error vs all raw variables
formula = "parity_error ~ risk_free_rate + IV + maturity + opt_volume + underlying_log_return + strike"
model = smf.ols(formula, data=df).fit()
print("\nsweden Regression results on parity error:")
print(model.summary())

#parity error vs all raw variables
formula = "parity_error ~ risk_free_rate + IV + maturity + opt_volume + strike"
model = smf.ols(formula, data=df).fit()
print("\nsweden Regression results on parity error:")
print(model.summary())

#parity error vs all raw variables
formula = "parity_error ~ risk_free_rate + IV + maturity + strike"
model = smf.ols(formula, data=df).fit()
print("\nsweden Regression results on parity error:")
print(model.summary())

formula = "parity_error ~ ulying_illiquidity + call_illiquidity + put_illiquidity"
model = smf.ols(formula, data=df).fit()
print("\nsweden Regression results on parity error:")
print(model.summary())

formula = "parity_error ~ strike + ulying_illiquidity + call_illiquidity + put_illiquidity"
model = smf.ols(formula, data=df).fit()
print("\nsweden Regression results on parity error:")
print(model.summary())

df = linreg_dk
df['parity_error'] = df['y'] - df['x']
df['IV'] = (df['IV_put'] + df['IV_call']) / 2
df['opt_volume'] = (df['put_v'] + df['call_v'])/2
df['moneyness'] = (df['call_moneyness'] + df['put_moneyness']) / 2

#parity error vs all raw variables
formula = "parity_error ~ risk_free_rate + IV + underlying_volatility + maturity + opt_volume + underlying_log_return + strike"
model = smf.ols(formula, data=df).fit()
print("\ndenmark Regression results on parity error:")
print(model.summary())

#parity error vs all raw variables
formula = "parity_error ~ risk_free_rate + IV + maturity + opt_volume + underlying_log_return + strike"
model = smf.ols(formula, data=df).fit()
print("\ndenmark Regression results on parity error:")
print(model.summary())

#parity error vs all raw variables
formula = "parity_error ~ risk_free_rate + IV + maturity + opt_volume + strike"
model = smf.ols(formula, data=df).fit()
print("\ndenmark Regression results on parity error:")
print(model.summary())

#parity error vs all raw variables
formula = "parity_error ~ risk_free_rate + IV + maturity + strike"
model = smf.ols(formula, data=df).fit()
print("\ndenmark Regression results on parity error:")
print(model.summary())

formula = "parity_error ~ ulying_illiquidity + call_illiquidity + put_illiquidity"
model = smf.ols(formula, data=df).fit()
print("\ndenmark Regression results on parity error:")
print(model.summary())

formula = "parity_error ~ strike + ulying_illiquidity + call_illiquidity + put_illiquidity"
model = smf.ols(formula, data=df).fit()
print("\ndenmark Regression results on parity error:")
print(model.summary())
