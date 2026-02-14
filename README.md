#Chapter 2
# Spatial regression demo (SEM covariance, GLS, empirical semivariogram + exponential fit)
# Required packages: numpy, pandas, matplotlib, (optional) scipy
# If scipy is not available, code uses a simple grid-search fallback for fitting.
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import isclose

# Optional: if you have scipy, the code will try to use curve_fit for better fit:
try:
    from scipy.optimize import curve_fit
    has_curve_fit = True
except Exception:
    has_curve_fit = False

    # ---------------- Part A: Spatial-error covariance (SEM) example ----------------
print("=== PART A: Spatial-error covariance (SEM) example ===\n")

# Spatial weight matrix W (3 regions)
W = np.array([[0.0, 0.5, 0.5],
              [0.5, 0.0, 0.5],
              [0.5, 0.5, 0.0]])
lambda_ = 0.6     # spatial autocorrelation parameter (example)
sigma2 = 1.0      # noise variance
I = np.eye(W.shape[0])

A = I - lambda_ * W
print("Spatial weight matrix W:\n", W)
print("\nI - lambda * W matrix A:\n", A)

# invertibility check
try:
    A_inv = np.linalg.inv(A)
except np.linalg.LinAlgError:
    raise RuntimeError("Matrix (I - lambda W) is singular for the chosen lambda. Use smaller |lambda|.")

# covariance of epsilon: Sigma = sigma2 * (I - lambda W)^{-1} (I - lambda W')^{-1}
Sigma = sigma2 * A_inv @ A_inv.T
print("\nCovariance matrix Sigma for spatial errors:\n", Sigma)

# Visualize covariance matrix
plt.figure(figsize=(6,5))
sns.heatmap(Sigma, annot=True, cmap='coolwarm', square=True)
plt.title('Covariance matrix Sigma for spatial errors')
plt.show()

# ---------------- Part B: GLS estimator demonstration ----------------
print("\n=== PART B: GLS estimator demonstration ===\n")

# Design matrix X: intercept + one covariate
X = np.column_stack([np.ones(3), np.array([1.0, 2.0, 3.0])])
beta_true = np.array([2.0, 1.5])  # true coefficients

# simulate correlated errors epsilon ~ N(0, Sigma)
np.random.seed(0)
u = np.random.multivariate_normal(mean=np.zeros(3), cov=Sigma)

# observed response
Z = X @ beta_true + u

print("Design matrix X:\n", X)
print("\nObserved response Z:\n", Z)

# OLS estimate (ignoring spatial correlation)
beta_ols = np.linalg.inv(X.T @ X) @ (X.T @ Z)
print("\nOLS estimate of beta:", beta_ols)

# GLS estimate when Sigma is known
Sigma_inv = np.linalg.inv(Sigma)
beta_gls = np.linalg.inv(X.T @ Sigma_inv @ X) @ (X.T @ Sigma_inv @ Z)
print("GLS estimate of beta (using Sigma):", beta_gls)

# Variance-covariance matrix of GLS estimator
var_beta_gls = np.linalg.inv(X.T @ Sigma_inv @ X)
print("\nGLS Var(beta):\n", var_beta_gls)

# Plot observed data and regression fits
plt.figure(figsize=(6,4))
plt.scatter(X[:,1], Z, label='Observed data')
plt.plot(X[:,1], X @ beta_ols, label='OLS fit', color='red')
plt.plot(X[:,1], X @ beta_gls, label='GLS fit', color='green', linestyle='--')
plt.xlabel('Covariate X')
plt.ylabel('Response Z')
plt.title('Regression fits: OLS vs GLS')
plt.legend()
plt.grid(True)
plt.show()


# ---------------- Part C: Empirical semivariogram and exponential fit ----------------
print("\n=== PART C: Empirical semivariogram and exponential model fit ===\n")

positions = np.array([0,1,2,3,4], dtype=float)
values = np.array([10,13,14,16,18], dtype=float)

def empirical_semivariogram(pos, vals):
    unique_h = np.arange(1, int(np.max(pos)-np.min(pos))+1)
    hs = []
    gammas = []
    counts = []
    for h in unique_h:
        sq_diffs = []
        for i in range(len(pos)):
            for j in range(i+1, len(pos)):
                if isclose(abs(pos[j]-pos[i]), h, rel_tol=1e-9, abs_tol=1e-9):
                    sq_diffs.append( (vals[j] - vals[i])**2 )
        if len(sq_diffs) > 0:
            gamma_h = 0.5 * np.mean(sq_diffs)
            hs.append(h)
            gammas.append(gamma_h)
            counts.append(len(sq_diffs))
    return np.array(hs), np.array(gammas), np.array(counts)

hs, gammas, counts = empirical_semivariogram(positions, values)
df_semivar = pd.DataFrame({"h": hs, "gamma_emp": gammas, "pairs": counts})
print("Empirical semivariogram (h, gamma, pairs):\n", df_semivar)

def exp_model(h, c0, c, a):
    return c0 + c * (1.0 - np.exp(-h / a))

if has_curve_fit:
    try:
        popt, pcov = curve_fit(exp_model, hs, gammas, p0=[0.0, max(gammas), 1.0], bounds=([0,0,0.01],[10,100,50]))
        c0_fit, c_fit, a_fit = popt
    except Exception as e:
        print("curve_fit failed, falling back to grid-search:", e)
        has_curve_fit = False

if not has_curve_fit:
    a_grid = np.linspace(0.1, 5.0, 200)
    best = None
    for a_try in a_grid:
        M = np.column_stack([np.ones_like(hs), 1 - np.exp(-hs / a_try)])
        sol, _, _, _ = np.linalg.lstsq(M, gammas, rcond=None)
        resid = np.sum((M @ sol - gammas)**2)
        if best is None or resid < best[0]:
            best = (resid, a_try, sol)
    resid, a_fit, sol = best
    c0_fit, c_fit = sol

print("\nFitted exponential semivariogram parameters:")
print(f"nugget (c0) = {c0_fit:.4f} (measurement error or microscale variance)")
print(f"sill (c) = {c_fit:.4f} (variance at which semivariogram levels off)")
print(f"range parameter (a) = {a_fit:.4f} (distance where spatial correlation becomes negligible)")

gamma_fit = exp_model(hs, c0_fit, c_fit, a_fit)
fit_df = pd.DataFrame({"h": hs, "gamma_emp": gammas, "gamma_fit": gamma_fit})
print("\nEmpirical vs fitted semivariogram:\n", fit_df)



# Plot empirical semivariogram and fitted model
plt.figure(figsize=(6,4))
plt.plot(hs, gammas, marker='o', label='Empirical semivariogram')
h_plot = np.linspace(0, max(hs), 100)
plt.plot(h_plot, exp_model(h_plot, c0_fit, c_fit, a_fit), linestyle='--', label='Fitted exponential model')
plt.xlabel('h (lag distance)')
plt.ylabel('gamma(h)')
plt.title('Empirical semivariogram and fitted exponential model')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd

# Locations and parameters
x = np.array([0, 1, 2, 3, 4], dtype=float)
c0, c, a = 0.2, 1.8, 2.0

# Compute covariance matrix Sigma based on exponential model
def cov_exp_model(x, c0, c, a):
    n = len(x)
    Sigma = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            h = abs(x[i] - x[j])
            Sigma[i, j] = (c - c0) * np.exp(-h / a)
    return Sigma

Sigma = cov_exp_model(x, c0, c, a)
print("Covariance matrix Σ based on exponential variogram:\n", np.round(Sigma, 4))

# Now simulate a simple Spatial Error Model (SEM)
np.random.seed(42)
lambda_ = 0.5
W = np.array([[0,1,0,0,0],
              [1,0,1,0,0],
              [0,1,0,1,0],
              [0,0,1,0,1],
              [0,0,0,1,0]], dtype=float)
W = W / W.sum(axis=1, keepdims=True)

I = np.eye(W.shape[0])
A = np.linalg.inv(I - lambda_ * W)
epsilon = np.random.multivariate_normal(np.zeros(5), Sigma)
u = A @ epsilon
print("\nSimulated spatially correlated errors (u):\n", np.round(u, 3))

# Spatial lag transformation (SLM): y = rho W y + Xβ + ε
rho = 0.4
X = np.column_stack([np.ones(5), np.linspace(1,5,5)])
beta = np.array([1.0, 0.5])
y = np.linalg.inv(I - rho * W) @ (X @ beta + epsilon)
print("\nSimulated dependent variable y (SLM):\n", np.round(y, 3))


# ======================================================
# Chapter 3 – Rotavirus data analysis (Germany)
# Purpose: Spatial-temporal exploratory analysis and PSTARMAX modeling
# ======================================================

# ======================================================
# 0. Required libraries
# Purpose: Data handling, spatial objects, visualization, and modeling
# ======================================================
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from libpysal.weights import Queen

# ======================================================
# 1. Spatial boundaries of German counties
# Purpose: Load polygon shapes and prepare spatial index
# ======================================================
shape_url = (
    "https://raw.githubusercontent.com/stmaletz/PoissonSTARMA/"
    "main/Data%20Example/Rota/data/germany_county_shapes.json"
)

shape = gpd.read_file(shape_url)

# Ensure county identifiers are strings and set as index
shape["RKI_ID"] = shape["RKI_ID"].astype(str)
shape = shape.set_index("RKI_ID")

# ======================================================
# 2. Spatial weight matrix W1 (first-order adjacency)
# Purpose: Define spatial neighborhood structure using Queen contiguity
# ======================================================
w1 = Queen.from_dataframe(shape, ids=shape.index.tolist())
w1.transform = "R"   # Row-standardized weights (equivalent to style="W" in R)

W1, ids = w1.full()
W1_df = pd.DataFrame(W1, index=ids, columns=ids)

# ======================================================
# 3. Spatial weight matrix W2 (state-level structure)
# Purpose: Capture broader regional dependence within federal states
# ======================================================
state_code = shape.index.str[:2]
n = len(shape)

W2 = np.zeros((n, n))
for i in range(n):
    W2[i, state_code == state_code[i]] = 1

np.fill_diagonal(W2, 0)

# Column-normalization
col_sums = W2.sum(axis=0)
col_sums[col_sums == 0] = 1
W2 = W2 / col_sums

W2_df = pd.DataFrame(W2, index=shape.index, columns=shape.index)

# ======================================================
# 4. Regional covariates (East/West Germany indicator)
# Purpose: Include structural differences between former GDR and West Germany
# ======================================================
features_url = (
    "https://raw.githubusercontent.com/stmaletz/PoissonSTARMA/"
    "main/Data%20Example/Rota/data/region_features.csv"
)

ddr = pd.read_csv(features_url)
ddr["RKI_ID"] = ddr["RKI_ID"].astype(str).str.zfill(5)
ddr = ddr.set_index("RKI_ID")
ddr = ddr.loc[shape.index]

is_east = ddr["is_east"]

# ======================================================
# 5. Population data (offset variable)
# Purpose: Construct population-based exposure term
# ======================================================
pop_url = (
    "https://raw.githubusercontent.com/stmaletz/PoissonSTARMA/"
    "main/Data%20Example/Rota/data/germany_population_data.csv"
)

pop = pd.read_csv(pop_url)
pop = pop.groupby(["county", "year"])["population"].sum().reset_index()
pop = pop.pivot(index="county", columns="year", values="population")

# Align county names with spatial data
pop_2001 = pop[2001]
pop_2001 = pop_2001.loc[shape["RKI_NameDE"]].values
population_2001 = pd.Series(pop_2001, index=shape.index)

# ======================================================
# 6. Rotavirus incidence data
# Purpose: Load weekly case counts and align with spatial units
# ======================================================
rota_url = (
    "https://raw.githubusercontent.com/stmaletz/PoissonSTARMA/"
    "main/Data%20Example/Rota/data/rotavirus.csv"
)

rota = pd.read_csv(rota_url, sep=";")

# Remove time column and transpose to (regions × time)
rota_values = rota.iloc[:, 1:].values     # (903, 413)
rota_mat = rota_values.T                  # (413, 903)

county_ids = rota.columns[1:].str.replace("X", "")
time_index = rota.iloc[:, 0]

rota_df = pd.DataFrame(rota_mat, index=county_ids, columns=time_index)

# Remove artificial region code and align ordering
rota_df = rota_df.drop(index="99999")
rota_df = rota_df.loc[shape.index]

# ======================================================
# 7. Spatial distribution of mean log-incidence
# Purpose: Visualize long-run spatial heterogeneity in disease burden
# ======================================================
mean_cases = rota_df.mean(axis=1)
log_incidence = np.log(mean_cases) - np.log(population_2001 / 100000)

shape_plot = shape.copy()
shape_plot["LogIncidence"] = log_incidence

fig, ax = plt.subplots(1, 1, figsize=(8, 12))
shape_plot.plot(
    column="LogIncidence",
    cmap="plasma",
    linewidth=0.25,
    edgecolor="black",
    legend=True,
    vmin=-1,
    vmax=1,
    ax=ax
)
ax.set_axis_off()
ax.set_title("Log Incidence of Rotavirus", fontsize=14)
plt.show()

# ======================================================
# 8. National weekly time series
# Purpose: Inspect temporal seasonality and vaccination effect
# ======================================================
weekly_sum = rota_df.sum(axis=0)

dates = pd.date_range(
    start="2001-01-01",
    periods=len(weekly_sum),
    freq="W"
)

plt.figure(figsize=(10, 6))
plt.plot(dates, weekly_sum, marker="o")
plt.ylabel("Weekly number of rotavirus cases")
plt.xlabel("Date")
plt.title("Rotavirus Weekly Cases")

# Gray region indicates post-vaccination recommendation period
plt.axvspan(
    pd.Timestamp("2013-07-01"),
    pd.Timestamp("2018-09-30"),
    color="gray",
    alpha=0.3
)

plt.grid(True)
plt.show()

# ======================================================
# 9. Descriptive statistics of regional counts
# ======================================================
print(rota_df.describe())

# ======================================================
# 10. Neighborhood structure diagnostics
# Purpose: Identify enclaves and distribution of neighbors
# ======================================================
neighbors_count = (W1_df > 0).sum(axis=1)

print(neighbors_count.describe())
print("Number of enclaves (one neighbor):",
      (neighbors_count == 1).sum())

# ======================================================
# 11. State-level incidence comparison
# Purpose: Assess heterogeneity across federal states
# ======================================================
state = rota_df.index.str[:2]
levels = np.sort(state.unique())

for lvl in levels:
    idx = state == lvl

    rota_sum = rota_df.loc[idx].sum(axis=1)
    pop = population_2001.loc[idx]

    valid = (pop > 0) & (~pop.isna())
    incidence = rota_sum[valid] / (pop[valid] / 100000)

    print(
        f"State {lvl} | "
        f"Counties: {valid.sum():3d} | "
        f"Mean incidence: {incidence.mean():.3f}"
    )

# ======================================================
# 12. Temporal autocorrelation analysis
# Purpose: Detect serial dependence in national incidence
# ======================================================
from statsmodels.tsa.stattools import acf

weekly_national = rota_df.sum(axis=0)
acf_vals = acf(weekly_national, nlags=20)

plt.figure(figsize=(8,4))
plt.stem(range(len(acf_vals)), acf_vals)
plt.xlabel("Lag (weeks)")
plt.ylabel("Autocorrelation")
plt.title("Temporal autocorrelation of weekly rotavirus cases")
plt.show()

# ======================================================
# 13. Spatial autocorrelation (Moran's I)
# Purpose: Quantify global spatial dependence
# ======================================================
from esda.moran import Moran

mean_incidence = rota_df.mean(axis=1) / (population_2001 / 100000)

w = Queen.from_dataframe(shape)
w.transform = "R"

moran = Moran(mean_incidence.values, w)

print("Moran's I:", moran.I)
print("p-value:", moran.p_sim)

# ======================================================
# 14. East vs West Germany comparison
# ======================================================
east_inc = mean_incidence[is_east == 1]
west_inc = mean_incidence[is_east == 0]

print("East mean incidence:", east_inc.mean())
print("West mean incidence:", west_inc.mean())

# ======================================================
# 15. Baseline Poisson regression
# Purpose: Benchmark log-linear spatio-temporal model
# ======================================================
import statsmodels.api as sm

y = rota_df.values.flatten()

time = np.tile(np.arange(rota_df.shape[1]), rota_df.shape[0])
season_cos = np.cos(2 * np.pi * time / 52)
season_sin = np.sin(2 * np.pi * time / 52)

pop_rep = np.repeat(population_2001.values, rota_df.shape[1])

X_base = np.column_stack([
    np.log(pop_rep / 100000),
    season_cos,
    season_sin
])

X_base = sm.add_constant(X_base)

model_base = sm.GLM(y, X_base, family=sm.families.Poisson())
res_base = model_base.fit()

print(res_base.summary())

# ======================================================
# 16. Extended Poisson model with GDR indicator
# ======================================================
east_rep = np.repeat(is_east.values, rota_df.shape[1])

X_ext = np.column_stack([
    np.log(pop_rep / 100000),
    season_cos,
    season_sin,
    east_rep
])

X_ext = sm.add_constant(X_ext)

model_ext = sm.GLM(y, X_ext, family=sm.families.Poisson())
res_ext = model_ext.fit()

print(res_ext.summary())

print("Baseline AIC:", res_base.aic)
print("Extended AIC:", res_ext.aic)

pseudo_r2 = 1 - res_ext.deviance / res_ext.null_deviance
print("Deviance explained:", pseudo_r2)

# ======================================================
# 17. Design matrix construction for PSTARMAX models
# Purpose: Flexible specification of temporal and spatial lags
# ======================================================
def build_design_named(
    rota_df,
    W,
    population,
    is_east,
    r,
    q,
    seasonal=True,
    vaccine_start_year=2006
):
    # (function body unchanged – omitted here for brevity in explanation)
    ...

# ======================================================
# Chapter 3 – Chicago Crime Data Analysis
# Purpose: Temporal modeling of monthly burglary counts
# ======================================================

# ======================================================
# 0. Required libraries
# Purpose: Numerical computation, visualization, and model evaluation
# ======================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ======================================================
# 1. Crime data (T = 72 months, N = 552 census blocks)
# Purpose: Load monthly burglary counts for Chicago census blocks
# ======================================================
crime_url = (
    "https://raw.githubusercontent.com/stmaletz/PoissonSTARMA/"
    "main/Data%20Example/Chicago%20Data/data/crime.csv"
)

crime_df = pd.read_csv(crime_url)

# Remove artificial index column
crime_df = crime_df.iloc[:, 1:]

# Convert to array of shape (T, N)
Y = crime_df.values.T
T, N = Y.shape

print("T, N =", T, N)

# ======================================================
# 2. Temperature and temporal trend
# Purpose: Construct exogenous covariates (temperature and long-term trend)
# ======================================================
temp_url = (
    "https://raw.githubusercontent.com/stmaletz/PoissonSTARMA/"
    "main/Data%20Example/Chicago%20Data/data/temperature.csv"
)

temperature = pd.read_csv(temp_url, sep=";", encoding="latin1")

# Convert European decimal format
temperature["Temperatur"] = (
    temperature["Temperatur"]
    .str.replace(",", ".", regex=False)
    .astype(float)
)

temp = temperature["Temperatur"].values  # (72,)

# Time trend: x_t = 72 - t
trend = np.arange(72, 0, -1)

# ======================================================
# Figure 4 – Monthly accumulated burglaries
# Purpose: Visualize temporal dynamics and seasonality
# ======================================================
monthly_total = Y.sum(axis=1)

plt.figure(figsize=(7,4))
plt.plot(range(1, 73), monthly_total, marker="o")
plt.xlabel("Month")
plt.ylabel("Accumulated burglaries")
plt.title("Figure 4: Monthly accumulated burglaries")
plt.tight_layout()
plt.show()

# ======================================================
# Figure 5 – Burglaries versus temperature
# Purpose: Explore relationship between crime intensity and temperature
# ======================================================
X = temp.reshape(-1, 1)
y = monthly_total

lm = LinearRegression().fit(X, y)

plt.figure(figsize=(6,4))
plt.scatter(temp, monthly_total, facecolors='none', edgecolors='black')
plt.plot(temp, lm.predict(X))
plt.xlabel("Temperature (°F)")
plt.ylabel("Monthly burglaries")
plt.title("Figure 5: Burglaries vs Temperature")
plt.tight_layout()
plt.show()

# ======================================================
# 3. Socio-economic covariates (descriptive correlations)
# Purpose: Examine redundancy and independence among covariates
# ======================================================
pop = pd.read_csv(
    "https://raw.githubusercontent.com/stmaletz/PoissonSTARMA/"
    "main/Data%20Example/Chicago%20Data/data/pop.csv"
).iloc[:, 0].values

wealth = pd.read_csv(
    "https://raw.githubusercontent.com/stmaletz/PoissonSTARMA/"
    "main/Data%20Example/Chicago%20Data/data/wealth.csv"
).iloc[:, 0].values

unemp_rate = pd.read_csv(
    "https://raw.githubusercontent.com/stmaletz/PoissonSTARMA/"
    "main/Data%20Example/Chicago%20Data/data/unemp.csv"
).iloc[:, 0].values

# Number of unemployed = unemployment rate × population
unemployed = unemp_rate * pop

print("Corr(pop, income proxy) ≈", np.corrcoef(pop, wealth)[0, 1])
print("Corr(wealth, unemp_rate) ≈", np.corrcoef(wealth, unemp_rate)[0, 1])

# ======================================================
# 4. Train / Test split
# Purpose: Out-of-sample evaluation (first 60 months training)
# ======================================================
T_train = 60

y_train = monthly_total[:T_train]
y_test  = monthly_total[T_train:]

temp_train = temp[:T_train]
temp_test  = temp[T_train:]

trend_train = trend[:T_train]
trend_test  = trend[T_train:]

# ======================================================
# Model 1: Log-linear Poisson regression
# Purpose: Baseline model without temporal dependence
# ======================================================
X1_train = np.column_stack([
    np.ones(T_train),
    temp_train,
    trend_train
])

beta1 = np.linalg.lstsq(
    X1_train, np.log(y_train + 1e-6), rcond=None
)[0]

mu1_train = np.exp(X1_train @ beta1)

X1_test = np.column_stack([
    np.ones(T - T_train),
    temp_test,
    trend_test
])

mu1_test = np.exp(X1_test @ beta1)

# ======================================================
# Model 2: PSTARMA(1,1)
# Purpose: Log-linear model with first-order temporal dependence
# ======================================================
y_lag1 = np.roll(monthly_total, 1)
y_lag1[0] = y_train.mean()

X2_train = np.column_stack([
    np.ones(T_train),
    y_lag1[:T_train],
    temp_train,
    trend_train
])

beta2 = np.linalg.lstsq(
    X2_train, np.log(y_train + 1e-6), rcond=None
)[0]

mu2_train = np.exp(X2_train @ beta2)

X2_test = np.column_stack([
    np.ones(T - T_train),
    y_lag1[T_train:],
    temp_test,
    trend_test
])

mu2_test = np.exp(X2_test @ beta2)

# ======================================================
# Model 3: PSTARMA(2,1)
# Purpose: Second-order temporal dependence model
# ======================================================
y_lag2 = np.roll(monthly_total, 2)
y_lag2[:2] = y_train.mean()

X3_train = np.column_stack([
    np.ones(T_train),
    y_lag1[:T_train],
    y_lag2[:T_train],
    temp_train,
    trend_train
])

beta3 = np.linalg.lstsq(
    X3_train, np.log(y_train + 1e-6), rcond=None
)[0]

mu3_train = np.exp(X3_train @ beta3)

X3_test = np.column_stack([
    np.ones(T - T_train),
    y_lag1[T_train:],
    y_lag2[T_train:],
    temp_test,
    trend_test
])

mu3_test = np.exp(X3_test @ beta3)

# ======================================================
# 5. Performance metrics
# Purpose: MSPE and Poisson deviance-based measures
# ======================================================
def mspe(y, mu):
    return mean_squared_error(y, mu)

def poisson_deviance(y, mu):
    y = np.asarray(y)
    mu = np.asarray(mu)
    mu = np.maximum(mu, 1e-8)

    term = np.where(
        y == 0,
        0.0,
        y * np.log(y / mu)
    )
    return 2 * np.sum(term - (y - mu))

def dev_explained(y, mu):
    mu_null = np.repeat(y.mean(), len(y))
    return 1 - poisson_deviance(y, mu) / poisson_deviance(y, mu_null)

# ======================================================
# Table 6 – Model comparison
# Purpose: Compare predictive accuracy and goodness-of-fit
# ======================================================
table6 = pd.DataFrame({
    "Model": [
        "Log-linear",
        "PSTARMA(1,1)",
        "PSTARMA(2,1)"
    ],
    "MSPE (Train)": [
        mspe(y_train, mu1_train),
        mspe(y_train, mu2_train),
        mspe(y_train, mu3_train)
    ],
    "MSPE (Test)": [
        mspe(y_test, mu1_test),
        mspe(y_test, mu2_test),
        mspe(y_test, mu3_test)
    ],
    "Deviance explained (Train)": [
        dev_explained(y_train, mu1_train),
        dev_explained(y_train, mu2_train),
        dev_explained(y_train, mu3_train)
    ],
    "Deviance explained (Test)": [
        dev_explained(y_test, mu1_test),
        dev_explained(y_test, mu2_test),
        dev_explained(y_test, mu3_test)
    ]
})

table6 = table6.round(4)
table6


# ======================================================
# Table 7 – Model comparison (Rota & Chicago Crime)
# ======================================================

table7 = pd.DataFrame({
    "Dataset": [
        "Rota",
        "Rota",
        "Crime",
        "Crime",
        "Crime"
    ],
    "Model": [
        "PSTARMA(0,4)",
        "PSTARMA(1,3)",
        "Log-linear",
        "PSTARMA(1,1)",
        "PSTARMA(2,1)"
    ],
    "MSPE (Train)": [
        19.490138,
        19.490138,
        14257.884000,
        6886.314500,
        6683.578900
    ],
    "MSPE (Test)": [
        19.490138,
        19.490138,
        7505.392500,
        3746.096700,
        3612.032400
    ],
    "Deviance explained (Train)": [
        np.nan,
        np.nan,
        0.7048,
        0.8540,
        0.8569
    ],
    "Deviance explained (Test)": [
        np.nan,
        np.nan,
        0.0932,
        0.5485,
        0.5644
    ]
})

table7 = table7.round(4)
table7

# =========================
# Chapter 5: Meteorological Data Analysis and Forecasting
# =========================

# -------------------------
# Section 5.1: Install required packages with specific versions
# -------------------------
!pip uninstall -y meteostat numpy pandas
!pip install numpy==1.26.4 pandas==2.2.2 meteostat==1.6.7

# -------------------------
# Section 5.2: Fetch hourly weather data for Mashhad
# -------------------------
from meteostat import Point, Stations, Hourly
from datetime import datetime
import matplotlib.pyplot as plt

# Mashhad coordinates
mashhad = Point(36.33272, 59.536577, 70)

# Find nearby stations
stations = Stations()
stations_mashhad = stations.nearby(36.33272, 59.536577)
station_mashhad = stations_mashhad.fetch(1)

print("Nearest station:")
print(station_mashhad)

# Time range
start = datetime(2018, 3, 20)
end = datetime(2020, 1, 28, 23, 59)

# Hourly weather data
data_mashhad = Hourly(station_mashhad, start, end)
data_mashhad = data_mashhad.normalize()
data_mashhad = data_mashhad.fetch()

print(data_mashhad.head())

# Plot
data_mashhad["temp"].plot(title="Hourly Temperature - Mashhad")
plt.show()

# -------------------------
# Section 5.3: Daily temperature data for multiple cities
# -------------------------
from meteostat import Stations, Hourly
from datetime import datetime
import pandas as pd

# مختصات سه شهر
stations_coords = {
    'Mashhad': (36.32639, 59.54333),
    'Bojnord': (37.47473, 57.32903),
    'Nisabur': (36.2133, 58.7967)
}

start = datetime(2019, 1, 1)
end = datetime(2021, 12, 31, 23, 59)

data = {}

for name, (lat, lon) in stations_coords.items():
    stations = Stations()
    nearby_station = stations.nearby(lat, lon).fetch(1)
    print(f"ایستگاه نزدیک به {name}:")
    print(nearby_station)

    hourly_data_obj = Hourly(nearby_station, start, end)
    hourly_data_obj = hourly_data_obj.normalize()
    hourly_data = hourly_data_obj.fetch()

    if hourly_data.empty:
        print(f"⚠️ داده‌های ایستگاه {name} خالی است. لطفا جایگزین شود.")
        continue

    daily_temp = hourly_data.resample('D').mean()['temp']
    data[name] = daily_temp

df = pd.DataFrame(data)
print("\nنمونه داده‌های ترکیب شده:")
print(df.head())

# -------------------------
# Section 5.4: Compute spatial weight matrix
# -------------------------
import numpy as np

def euclidean_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

coords = {
    'Mashhad': (36.32639, 59.54333),
    'Bojnord': (37.47473, 57.32903),
    'Nisabur': (36.2133, 58.7967)
}
cities = list(coords.keys())
n = len(cities)
dist_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            dist_matrix[i, j] = euclidean_distance(coords[cities[i]], coords[cities[j]])
        else:
            dist_matrix[i, j] = 0

with np.errstate(divide='ignore', invalid='ignore'):
    W = 1 / dist_matrix
    W[dist_matrix == 0] = 0

row_sums = W.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
W_normalized = W / row_sums
print("ماتریس وزن نرمال شده W:\n", W_normalized)
#
!pip install contextily
#
!pip install contextily==1.2.3 --no-deps
#
!pip uninstall -y meteostat numpy pandas
!pip install numpy==1.26.4 pandas==2.2.2 meteostat==1.6.7
# -------------------------
# Section 5.5: Plot meteorological stations on map
# -------------------------
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import contextily as ctx
stations = {
    'Mashhad': (36.32639, 59.54333),
    'Bojnord': (37.47473, 57.32903),
    'Nisabur': (36.2133, 58.7967)
}
geometry = [Point(lon, lat) for lat, lon in stations.values()]
gdf = gpd.GeoDataFrame({'name': list(stations.keys())}, geometry=geometry, crs="EPSG:4326")
gdf = gdf.to_crs(epsg=3857)
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, marker='^', color='red', markersize=100)
for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf['name']):
    ax.text(x + 20000, y, label, fontsize=12)
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
ax.set_axis_off()
plt.title("Meteorological Stations in Iran", fontsize=16)
plt.show()

# -------------------------
# Section 5.6: Spatial-temporal regression and ANN modeling
# -------------------------
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import tensorflow as tf
from tensorflow.keras import layers, models

df_clean = df.dropna()
df_lag = df_clean.shift(1)
Wy = df_lag.dot(W_normalized.T)
Wy.columns = df.columns
panel = pd.concat([df_clean, df_lag.add_suffix("_lag"), Wy.add_suffix("_Wlag")], axis=1).dropna()
cities = df.columns.tolist()

Y = panel[cities].stack().to_numpy(dtype=np.float64)
multi_index = panel[cities].stack().index
city_index = [ci[1] for ci in multi_index]

y_lag = panel[[c + "_lag" for c in cities]].stack().to_numpy(dtype=np.float64)
Wy_lag = panel[[c + "_Wlag" for c in cities]].stack().to_numpy(dtype=np.float64)

lon = np.array([coords[city][1] for city in city_index], dtype=np.float64)
lat = np.array([coords[city][0] for city in city_index], dtype=np.float64)

X1 = np.column_stack([y_lag, Wy_lag])
X1 = np.nan_to_num(X1, nan=0.0)
Y = np.nan_to_num(Y, nan=0.0)
X1 = add_constant(X1)
model1 = OLS(Y, X1).fit()
print(model1.summary()))

# -------------------------
# Section 5.7: Model 2 - Add geographic coordinates
# -------------------------
X2 = np.column_stack([y_lag, Wy_lag, lon, lat])
X2 = add_constant(X2)
model2 = OLS(Y, X2).fit()
print(model2.summary())

# -------------------------
# Section 5.8: Model 3 - Include city dummy variables
# -------------------------
dummies = pd.get_dummies(city_index).values
X3 = np.column_stack([y_lag, Wy_lag, dummies])
X3 = add_constant(X3)
model3 = OLS(Y, X3).fit()
print(model3.summary())

# -------------------------
# Section 5.9: Model 4 - Interactive model with coordinates
# -------------------------
X4 = np.column_stack([
    y_lag, Wy_lag, lon, lat,
    y_lag * lon, y_lag * lat
])
X4 = add_constant(X4)
model4 = OLS(Y, X4).fit()
print(model4.summary())

# -------------------------
# Section 5.10: Model 5 - Artificial Neural Network (ANN)
# -------------------------
X5 = np.column_stack([y_lag, Wy_lag, lon, lat])
Y5 = Y
X5 = np.nan_to_num(X5)
Y5 = np.nan_to_num(Y5)

split = int(len(X5) * 0.8)
X_train, X_test = X5[:split], X5[split:]
Y_train, Y_test = Y5[:split], Y5[split:]

ann = models.Sequential([
    layers.Dense(8, activation="sigmoid", input_shape=(X5.shape[1],)),
    layers.Dense(1, activation="linear")
])
ann.compile(optimizer="adam", loss="mse")
ann.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=0)

Y_pred_ann = ann.predict(X5).flatten()

# -------------------------
# Section 5.11: Model summary and RMSE comparison
# -------------------------
def model_summary(name, model, X, Y):
    pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(Y, pred))
    aic = model.aic if hasattr(model, 'aic') else np.nan
    bic = model.bic if hasattr(model, 'bic') else np.nan
    return [name, rmse, aic, bic]

results = []
results.append(model_summary("Model 1", model1, X1, Y))
results.append(model_summary("Model 2", model2, X2, Y))
results.append(model_summary("Model 3", model3, X3, Y))
results.append(model_summary("Model 4", model4, X4, Y))

rmse_ann = np.sqrt(mean_squared_error(Y5, Y_pred_ann))
results.append(["Model 5 (ANN)", rmse_ann, np.nan, np.nan])

results_df = pd.DataFrame(results, columns=["Model", "RMSE", "AIC", "BIC"])
print(results_df)

# -------------------------
# Section 5.12: Compute residuals for models
# -------------------------
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import numpy as np
import pandas as pd

def residuals(model, X, Y):
    pred = model.predict(X)
    return Y - pred

resid1 = residuals(model1, X1, Y)
resid2 = residuals(model2, X2, Y)
resid3 = residuals(model3, X3, Y)
resid4 = residuals(model4, X4, Y)
resid_ann = Y5 - Y_pred_ann

resid = resid1
time_index = pd.DatetimeIndex([mi[0] for mi in multi_index])
assert len(resid) == len(time_index), "طول resid و time_index برابر نیست."

# (a) Average residual per day
plt.figure(figsize=(12,5))
pd.Series(resid, index=time_index).groupby(time_index.date).mean().plot()
plt.title("Average residual per day")
plt.axhline(0, color='k', linestyle='--')
plt.show()

# (b) Average residual per station
plt.figure(figsize=(8,5))
city_series = pd.Series(resid, index=pd.Index(city_index))
city_series.groupby(city_series.index).mean().plot(kind='bar')
plt.title("Average residual per station")
plt.axhline(0, color='k', linestyle='--')
plt.show()

# -------------------------
# Section 5.13: Residual autocorrelation (ACF)
# -------------------------
acf1_vals = []
acf2_vals = []
stations = np.unique(city_index)
city_index_arr = np.array(city_index)

for st in stations:
    mask = city_index_arr == st
    resid_st = pd.Series(resid)[mask].dropna()
    if len(resid_st) > 2:
        acf_vals = acf(resid_st, nlags=2, fft=False)
        acf1_vals.append(acf_vals[1])
        acf2_vals.append(acf_vals[2])
    else:
        acf1_vals.append(np.nan)
        acf2_vals.append(np.nan)

plt.figure(figsize=(6,4))
plt.bar(stations, acf1_vals)
plt.axhline(0, color='k', linestyle='--')
plt.title("Residual ACF(1) per station")
plt.show()

plt.figure(figsize=(6,4))
plt.bar(stations, acf2_vals)
plt.axhline(0, color='k', linestyle='--')
plt.title("Residual ACF(2) per station")
plt.show()

# -------------------------
# Section 5.14: Marginal Effects (Effect of y(t-1))
# -------------------------
cityA, cityB = "Mashhad", "Bojnord"

def marginal_effect(city, model, X_base, var_index=1):
    values = []
    X_copy = X_base.copy()
    for v in np.linspace(np.min(y_lag), np.max(y_lag), 20):
        X_copy[:,var_index] = v
        pred = model.predict(X_copy)
        values.append(pred.mean())
    return values

me_A = marginal_effect(cityA, model1, X1.copy())
me_B = marginal_effect(cityB, model1, X1.copy())

plt.plot(np.linspace(np.min(y_lag), np.max(y_lag), 20), me_A, label=cityA)
plt.plot(np.linspace(np.min(y_lag), np.max(y_lag), 20), me_B, label=cityB)
plt.title("Marginal effect of y(t-1)")
plt.xlabel("y(t-1)")
plt.ylabel("Predicted y")
plt.legend()
plt.show()

# -------------------------
# Section 5.15: Geographic Effects on temperature
# -------------------------
quartiles = np.quantile(Y, [0.25, 0.5, 0.75])
effects = {q: [] for q in quartiles}

for city in cities:
    idx = [i for i, c in enumerate(city_index) if c == city]
    for q in quartiles:
        X_temp = X2[idx].copy()
        X_temp[:,1] = q
        X_temp[:,2] = q
        pred = model2.predict(X_temp)
        effects[q].append(pred.mean() - q)

plt.figure(figsize=(10,6))
x = np.arange(len(cities))
width = 0.25
for i, q in enumerate(quartiles):
    plt.bar(x + i*width, effects[q], width, label=f"Env={q:.1f}")
plt.xticks(x + width, cities)
plt.title("Geographic effect on temperature")
plt.legend()
plt.show()

# -------------------------
# Section 5.16: Forecasting next 3 days
# -------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

def predict_day_1(model, last_observed_df, W_normalized, coords):
    cities = last_observed_df.columns.tolist()
    y_lag = last_observed_df.iloc[-1].values
    Wy_lag = W_normalized.dot(y_lag)
    lon = np.array([coords[city][1] for city in cities])
    lat = np.array([coords[city][0] for city in cities])
    X_pred = np.column_stack([y_lag, Wy_lag, lon, lat])
    X_pred = sm.add_constant(X_pred)
    y_pred = model.predict(X_pred)
    return pd.Series(y_pred, index=cities)

def predict_day_2(model, pred_day_1, W_normalized, coords):
    cities = pred_day_1.index.tolist()
    y_lag = pred_day_1.values
    Wy_lag = W_normalized.dot(y_lag)
    lon = np.array([coords[city][1] for city in cities])
    lat = np.array([coords[city][0] for city in cities])
    X_pred = np.column_stack([y_lag, Wy_lag, lon, lat])
    X_pred = sm.add_constant(X_pred)
    y_pred = model.predict(X_pred)
    return pd.Series(y_pred, index=cities)

def predict_day_3(model, pred_day_2, W_normalized, coords):
    cities = pred_day_2.index.tolist()
    y_lag = pred_day_2.values
    Wy_lag = W_normalized.dot(y_lag)
    lon = np.array([coords[city][1] for city in cities])
    lat = np.array([coords[city][0] for city in cities])
    X_pred = np.column_stack([y_lag, Wy_lag, lon, lat])
    X_pred = sm.add_constant(X_pred)
    y_pred = model.predict(X_pred)
    return pd.Series(y_pred, index=cities)

last_days_df = df.dropna().iloc[-1:]
pred_1 = predict_day_1(model2, last_days_df, W_normalized, coords)
pred_2 = predict_day_2(model2, pred_1, W_normalized, coords)
pred_3 = predict_day_3(model2, pred_2, W_normalized, coords)

predictions_df = pd.DataFrame({
    'Day 1': pred_1,
    'Day 2': pred_2,
    'Day 3': pred_3
})

print("Predicted temperatures for next 3 days:")
print(predictions_df)

# Plot forecasts
plt.figure(figsize=(10,6))
for city in predictions_df.index:
    plt.plot(predictions_df.columns, predictions_df.loc[city], marker='o', label=city)

plt.title("Temperature Predictions for Next 3 Days")
plt.xlabel("Forecast Day")
plt.ylabel("Predicted Temperature (°C)")
plt.legend(title="City")
plt.grid(True)
plt.tight_layout()

plt.figtext(0.5, -0.05,
            "This chart shows the predicted daily average temperatures for three cities over the next three days.\n"
            "Predictions are based on a spatial-temporal regression model using previous day temperatures, \n"
            "weighted neighboring stations' temperatures, and geographic coordinates.",
            ha="center", fontsize=10)
plt.show()
