import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt


df = pd.read_excel("HW3_data_4_48.xlsx", sheet_name=1)

df = df["Fatalities"]
df.index = pd.date_range("1966", periods=len(df), freq="YS")
print(df)


fit1 = SimpleExpSmoothing(df, initialization_method="heuristic").fit()

fit2 = ExponentialSmoothing(
    df,
    # seasonal_periods=4,
    trend="multiplicative",
    # seasonal="multiplicative",
    use_boxcox=True,
    initialization_method="estimated",
).fit()

results = pd.DataFrame(
    index=[
        r"$\alpha$",
        r"$\beta$",
        r"$\phi$",
        r"$\gamma$",
        r"$l_0$",
        "$b_0$",
        "SSE",
    ]
)

params = [
    "smoothing_level",
    "smoothing_trend",
    "damping_trend",
    "smoothing_seasonal",
    "initial_level",
    "initial_trend",
]
results["First-Order"] = [fit1.params[p] for p in params] + [fit1.sse]
results["Second-Order"] = [fit2.params[p] for p in params] + [fit2.sse]


print(results)


ax = df.plot(
    figsize=(10, 6),
    color="black",
    title="Exponential Forecasts for Fatalities",
)
ax.set_ylabel("Fatalities")
ax.set_xlabel("Year")


fit1.fittedvalues.plot(ax=ax, style="--", color="lightgreen")
fit2.fittedvalues.plot(ax=ax, style="--", color="lightblue")

fit1.predict(start="2007-01-01", end="2012-01-01").rename(
    "first-order exponential smoothing"
).plot(ax=ax, style="-", color="green", legend=True)


fit2.predict(start="2007-01-01", end="2012-01-01").rename(
    "second-order exponential smoothing"
).plot(ax=ax, style="-", color="blue", legend=True)

plt.show()
