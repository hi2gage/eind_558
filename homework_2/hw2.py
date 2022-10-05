import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics import tsaplots
import matplotlib.pyplot as plt
from pandas import Series
from statsmodels.graphics.tsaplots import plot_pacf


def problem_1a(show=True):
    df = pd.read_excel(
        "HMWK3_Data.xlsx",
        sheet_name=0,
        parse_dates=True,
        squeeze=True,
        index_col=0,
    )
    df.plot.hist(bins=15, rwidth=0.9)
    if show:
        plt.show()


def problem_1b(show=True):
    df = pd.read_excel(
        "HMWK3_Data.xlsx",
        sheet_name=0,
        parse_dates=True,
        squeeze=True,
        index_col=0,
    )

    x = df

    auto_correlation = sm.tsa.acf(x, nlags=1)
    print(auto_correlation)

    fig = tsaplots.plot_acf(x, lags=1)

    if show:
        plt.show()


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


def problem_1c(show=True):
    df = pd.read_excel(
        "HMWK3_Data.xlsx",
        sheet_name=0,
        parse_dates=True,
        index_col="Year",
    )

    x = df.diff()
    print(df.size)

    auto_correlation = sm.tsa.acf(x, missing="drop")
    print(auto_correlation)
    print(len(auto_correlation))

    fig = tsaplots.plot_acf(x, missing="drop")

    # df.diff().plot()

    if show:
        plt.show()


def problem_3(show=True):
    df = pd.read_excel(
        "HMWK3_Data.xlsx",
        parse_dates=True,
        sheet_name=1,
        names=["dates", "miles"],
    )
    x = df["miles"]

    auto_correlation = sm.tsa.acf(x)
    print(auto_correlation)

    fig = tsaplots.plot_acf(x, lags=10)

    if show:
        plt.show()


def problem_4(show=True):
    df = pd.read_excel(
        "HMWK3_Data.xlsx",
        parse_dates=True,
        sheet_name=1,
        names=["dates", "miles"],
        squeeze=True,
    )

    df["miles_log"] = np.log(df["miles"])
    x = np.log2(df["miles_log"])

    auto_correlation = sm.tsa.acf(x)
    print(auto_correlation)

    # fig = tsaplots.plot_acf(x, lags=10)

    df["miles"].plot(x="dates")

    if show:
        plt.show()


if __name__ == "__main__":
    # problem_1a(True)
    # problem_1b(True)
    problem_1c()
    # problem_3()
    # problem_4()
