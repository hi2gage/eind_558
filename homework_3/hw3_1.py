import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.statespace.tools import diff


def problem_4_1():
    df = pd.read_excel("HW3_data_1.xlsx", sheet_name=0)
    df = df.set_index("Period")

    # Using Alpha = 0.2
    fit1 = SimpleExpSmoothing(df, initialization_method="heuristic").fit(
        smoothing_level=0.2, optimized=False
    )
    fcast1 = fit1.predict(start=40, end=50).rename(r"$\alpha=0.2$")

    # Using Alpha = Optimized
    fit2 = SimpleExpSmoothing(df, initialization_method="estimated").fit()
    fcast2 = fit2.predict(start=40, end=50).rename(
        r"$\alpha=%s$" % fit2.model.params["smoothing_level"]
    )

    # Plotting original
    plt.figure(figsize=(12, 8))
    plt.plot(df, marker="o", color="black")

    # Plotting forecast using alpha = 0.2
    plt.plot(fit1.fittedvalues, marker="o", color="lightblue")
    (line1,) = plt.plot(fcast1, marker="o", color="blue")

    # Plotting forecast using alpha = optimized
    plt.plot(fit2.fittedvalues, marker="o", color="lightgreen")
    (line2,) = plt.plot(fcast2, marker="o", color="green")

    # Plotting legend
    plt.legend([line1, line2], [fcast1.name, fcast2.name])

    # Saving figure
    plt.show()

    # Calculating forecast error
    error1 = []
    error2 = []
    for i in range(40, 51):
        error1.append(df["Yt"][i] - fcast1[i])
        error2.append(df["Yt"][i] - fcast2[i])

    # Calculating MSE
    mse1 = 0
    mse2 = 0
    for i in error1:
        mse1 = mse1 + i**2
    for i in error2:
        mse2 = mse2 + i**2
    mse1 = mse1 / len(error1)
    mse2 = mse2 / len(error2)

    print("MSE for alpha = 0.2 : ", mse1)
    print("MSE for alpha = Optimized : ", mse2)


def problem_4_20():
    df = pd.read_excel("HW3_data_2.xlsx", sheet_name=7)
    df = df.set_index("Year")
    column = "Anomaly, C"

    # Using Alpha = Optimized
    fit1 = SimpleExpSmoothing(
        df[column], initialization_method="estimated"
    ).fit()
    df["smoothed"] = fit1.fittedvalues
    df["diff"] = df[column] - df["smoothed"]
    fcast1 = fit1.predict(start=40, end=50).rename(
        r"$\alpha=%s$" % fit1.model.params["smoothing_level"]
    )

    # Using Alpha = 0.2
    fit2 = SimpleExpSmoothing(
        df[column], initialization_method="estimated"
    ).fit(smoothing_level=0.2, optimized=False)
    fcast2 = fit2.predict(start=40, end=50).rename(
        r"$\alpha=%s$" % fit2.model.params["smoothing_level"]
    )

    # Plotting original
    plt.figure(figsize=(12, 8))
    plt.plot(df[column], color="black")

    # Plotting forecast using alpha = optimized
    (line1,) = plt.plot(fit1.fittedvalues, color="green")
    (line2,) = plt.plot(fit2.fittedvalues, color="orange")
    plt.legend([line1, line2], [fcast1.name, fcast2.name])
    plt.savefig("new_figs/problem_4_20_1.png")

    # Plotting diff
    fig, ax = plt.subplots(figsize=(12, 8))

    fit2 = SimpleExpSmoothing(
        df["diff"],
        initialization_method="estimated",
    ).fit()
    fcast2 = fit2.predict(start=40, end=50).rename(
        r"$\alpha=%s$" % fit2.model.params["smoothing_level"]
    )
    plt.plot(df["diff"], color="black")

    # Plotting forecast using alpha = optimized
    (line2,) = plt.plot(fit2.fittedvalues, color="green")
    plt.legend([line2], [fcast2.name])

    df["diff"].plot()
    plt.savefig("new_figs/problem_4_20_2.png")


def problem_4_48():
    df = pd.read_excel("HW3_data_4_48.xlsx", sheet_name=0)

    df = df["Positive_Tests"]

    date_index = pd.date_range("2003", periods=len(df), freq="W")
    df.index = date_index
    # df = df.notnull()
    print(df.tail(185))
    print(type(df))
    print(df.min())

    # fit1 = ExponentialSmoothing(
    #     df,
    #     initialization_method="estimated",
    #     trend="multiplicative",
    #     seasonal="multiplicative",
    #     seasonal_periods=52,
    # ).fit()
    # fcast1 = fit1.predict(start="2010-12-26", end="2014-06-299").rename(
    #     r"$\alpha=%s$" % fit1.model.params["smoothing_level"]
    # )

    # (line1,) = plt.plot(fit1.fittedvalues, color="green")
    # plt.legend(
    #     [line1], [r"$\alpha=%s$" % fit1.model.params["smoothing_level"]]
    # )

    fit2 = Holt(
        df,
        exponential=True,
    ).fit()
    fit2.forecast(2).plot(marker="o", color="blue", legend=True)

    (line2,) = plt.plot(fit2.fittedvalues, color="green")
    plt.legend(
        [line2], [r"$\alpha=%s$" % fit2.model.params["smoothing_level"]]
    )

    # print(df)
    df.plot()
    # plt.show()


if __name__ == "__main__":
    # problem_4_1()
    # problem_4_2()
    # problem_4_3()
    # problem_4_10()

    # problem_4_20()
    problem_4_48()
    # problem_4_52()
