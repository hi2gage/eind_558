import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics import tsaplots
import matplotlib.pyplot as plt
from pandas import Series
from statsmodels.graphics.tsaplots import plot_pacf
import seaborn as sns


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


def problem_2(show=True):
    df = pd.read_excel(
        "HMWK3_Data.xlsx",
        sheet_name=2,
        parse_dates=True,
        index_col="Date",
    )
    df = df.tail(200)

    x = df["Close"]

    auto_correlation = sm.tsa.acf(x, missing="drop")
    print(auto_correlation)
    print(len(auto_correlation))

    fig = tsaplots.plot_acf(x, missing="drop")

    tesla = df["Close"]
    tesla_5 = tesla.rolling(window=10).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(tesla, "k-", label="Original")
    plt.plot(tesla_5, "r-", label="Running average 5 unit")
    plt.xlabel("Date")
    plt.grid(linestyle=":")
    # plt.fill_between(co2.index, 0, co2, color="r", alpha=0.1)
    plt.legend(loc="upper left")

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
    for idx, value in enumerate(auto_correlation):
        print(f"lag {idx}: {value}")

    fig = tsaplots.plot_acf(x)

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
    x = df["miles_log"]

    auto_correlation = sm.tsa.acf(x)
    print(auto_correlation)

    fig = tsaplots.plot_acf(x)

    # df["miles_log"].plot()

    if show:
        plt.show()


def problem_5(show=True):
    df = pd.read_excel(
        "HMWK3_Data.xlsx",
        parse_dates=True,
        sheet_name=3,
        # squeeze=True,
    )
    co2 = df["co2"]
    co2_5 = co2.rolling(window=6).mean()

    print(df)
    print(co2_5)

    plt.figure(figsize=(10, 5))
    plt.plot(co2, "k-", label="Original")
    plt.plot(co2_5, "r-", label="Running average 5 unit")
    plt.xlabel("Date")
    plt.grid(linestyle=":")
    # plt.fill_between(co2.index, 0, co2, color="r", alpha=0.1)
    plt.legend(loc="upper left")
    plt.show()


def problem_6(show=True):
    df = pd.read_excel("HMWK3_Data.xlsx", sheet_name=4, index_col="Index")
    x = df["error"]
    print(x)
    # auto_correlation = sm.tsa.acf(x)
    # for idx, value in enumerate(auto_correlation):
    #     print(f"lag {idx}: {value}")

    # fig = tsaplots.plot_acf(x)

    MSE = 0
    for i in x.values.tolist():
        MSE += float(i) * float(i)
    print(f"MSE: {MSE}")

    absDev = 0
    mean = x.mean()
    for i in x.values.tolist():
        absDev += abs(i - mean)
    print(f"mean absolute deviation: {absDev}")

    # # define distributions
    # sample_size = 10000
    # standard_norm = x

    # # plots for standard distribution
    # fig, ax = plt.subplots(1, 2, figsize=(12, 7))
    # sns.histplot(standard_norm, kde=True, color="blue", ax=ax[0])
    # sm.ProbPlot(standard_norm).qqplot(line="s", ax=ax[1])

    # if show:
    #     plt.show()


if __name__ == "__main__":
    # problem_1a(True)
    # problem_1b(True)
    # problem_1c()
    # problem_2()
    # problem_3()
    # problem_4()
    # problem_5()
    problem_6(False)
