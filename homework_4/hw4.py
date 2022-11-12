import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.statespace.tools import diff
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def problem_5_3():
    df = pd.read_excel("hw_4_data.xlsx", sheet_name=0)
    df = df.set_index("Period")
    print(df.shape)
    df.plot()

    # 5.3. a & b
    # fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    # sm.graphics.tsa.plot_acf(df.values.squeeze(), lags=24, ax=ax[0])
    # sm.graphics.tsa.plot_pacf(df.values.squeeze(), lags=24, ax=ax[1])
    # plt.show()
    # plt.savefig("figs/series_plot_5_3.png")

    # 5.3. c
    model = ARIMA(df, order=(1, 0, 1))
    model_fit = model.fit()
    print(model_fit.summary())


def problem_5_4():
    pass


def problem_5_10():
    df = pd.read_excel("hw_4_data.xlsx", sheet_name=1)
    date_index = pd.date_range("4/1/1953", periods=len(df), freq="MS")
    df.index = date_index
    df = df = df.rename(columns={"Rate": "value"})
    # df = df.set_index("Month")
    print(type(df))

    # # Original Series
    # fig, axes = plt.subplots(3, 2)
    # axes[0, 0].plot(df.value)
    # axes[0, 0].set_title("Original Series")
    # plot_acf(df.value.squeeze(), ax=axes[0, 1])

    # # 1st Differencing
    # axes[1, 0].plot(df.value.diff())
    # axes[1, 0].set_title("1st Order Differencing")
    # plot_acf(df.value.diff().dropna(), ax=axes[1, 1])

    # # 2nd Differencing
    # axes[2, 0].plot(df.value.diff().diff())
    # axes[2, 0].set_title("2nd Order Differencing")
    # plot_acf(df.value.diff().diff().dropna(), ax=axes[2, 1])

    # # PACF plot of 1st differenced series
    # plt.rcParams.update({"figure.figsize": (9, 3), "figure.dpi": 120})

    # fig, axes = plt.subplots(1, 2)
    # axes[0].plot(df.value.diff())
    # axes[0].set_title("1st Differencing")
    # axes[1].set(ylim=(0, 5))
    # plot_pacf(df.value.diff().dropna(), ax=axes[1])

    # fig, axes = plt.subplots(1, 2)
    # axes[0].plot(df.value.diff())
    # axes[0].set_title("1st Differencing")
    # axes[1].set(ylim=(0, 1.2))
    # plot_acf(df.value.diff().dropna(), ax=axes[1])

    model = ARIMA(df.value, order=(1, 0, 2))
    model_fit = model.fit()
    print(model_fit.summary())

    # # Plot residual errors
    # residuals = pd.DataFrame(model_fit.resid)
    # fig, ax = plt.subplots(1, 2)
    # residuals.plot(title="Residuals", ax=ax[0])
    # residuals.plot(kind="kde", title="Density", ax=ax[1])

    fit1 = model.fit()

    # fit1.fittedvalues.plot(ax=ax, style="--", color="lightgreen")
    print(fit1.fittedvalues)
    # fig, ax = plt.subplots()

    ax = df.value.plot(
        figsize=(10, 6),
        color="black",
        title="Forecasts using ARIMA method",
    )
    ax.set_ylabel("Rate")
    ax.set_xlabel("Year")
    vals = ax.get_yticks()

    fit1.fittedvalues.plot(ax=ax, style="-", color="orange")

    fit1.predict(start="2005-06-01", end="2007-02-01").rename(
        "model = ARIMA(df.value, order=(1, 0, 2))"
    ).plot(ax=ax, style="-", color="red", legend=True)

    # Using Alpha = 0.2
    fit2 = SimpleExpSmoothing(df.value, initialization_method="estimated").fit(
        smoothing_level=0.2, optimized=False
    )

    fit2.fittedvalues.plot(ax=ax, style="-", color="green")

    fit2.predict(start="2005-06-01", end="2007-02-01").rename(
        "simple exponential smoothing with alpha = 0.2 "
    ).plot(ax=ax, style="-", color="lightgreen", legend=True)

    plt.show()
    print(fit2.summary())


def problem_5_16():
    pass


def problem_5_17():
    pass


def problem_5_24():
    pass


def problem_5_50():
    pass


if __name__ == "__main__":
    problem_5_10()
