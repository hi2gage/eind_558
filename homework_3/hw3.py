import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt


def appendDictToDF(df, dictToAppend):
    df = pd.concat([df, pd.DataFrame.from_records([dictToAppend])])
    return df


def exponential_smoothing(df, smoothing_level, column):
    df["EWMA"] = df[column].ewm(alpha=smoothing_level).mean()
    return df


def find_optimal_alpha_exponential_smoothing(df, column, file_name=""):
    current_alpha = 0.01
    min_mse = 1000000
    optimal_alpha = 0.0
    df_sse = pd.DataFrame(columns=["alpha", "sse"])
    while current_alpha <= 1:
        df["EWMA"] = df[column].ewm(alpha=current_alpha).mean()
        mse = ((df[column] - df["EWMA"]).iloc[:40] ** 2).mean()
        df_sse = appendDictToDF(df_sse, {"alpha": current_alpha, "sse": mse})
        if mse < min_mse:
            min_mse = mse
            optimal_alpha = current_alpha
        current_alpha += 0.01
    df_sse.plot(x="alpha", y="sse")
    print(f"Optimal alpha: {optimal_alpha}")
    plt.savefig(file_name)
    return optimal_alpha


def find_sample_ACF(df, lag):
    df["Yt-1"] = df["Yt"].shift(lag)
    df["Yt-1"] = df["Yt-1"] - df["Yt-1"].mean()
    df["Yt"] = df["Yt"] - df["Yt"].mean()
    numerator = (df["Yt-1"] * df["Yt"]).sum()
    denominator = (df["Yt-1"] ** 2).sum()
    return numerator / denominator


def smooth_first_n(df, smoothing_level, n, column):
    df = exponential_smoothing(df, smoothing_level, column)
    df = df.iloc[:n]
    return df["EWMA"]


def n_step_ahead_forecast_last_k(
    df, smoothing_level, n_steps, k_forecasts, column
):
    df = df.shift(n_steps)
    df = exponential_smoothing(df, smoothing_level, column)
    df = df.iloc[-k_forecasts:]
    return df["EWMA"]


def problem_4_1():
    df = pd.read_excel("HW3_data_1.xlsx", sheet_name=0)
    df = df.set_index("Period")
    fig, ax = plt.subplots()
    df["Yt"].plot(ax=ax, label="Yt")
    smooth_first_n(df, 0.2, n=40, column="Yt").plot(ax=ax)
    n_step_ahead_forecast_last_k(
        df,
        smoothing_level=0.2,
        n_steps=1,
        k_forecasts=10,
        column="Yt",
    ).plot(ax=ax, label="Forecast", color="red")
    plt.legend()
    plt.savefig("figs/problem_4_1.png")


def forcast_using_stats_model():
    df = pd.read_excel("HW3_data_1.xlsx", sheet_name=0)
    df = df.set_index("Period")
    fit1 = SimpleExpSmoothing(df, initialization_method="heuristic").fit(
        smoothing_level=0.2, optimized=False
    )
    fcast1 = fit1.predict(start=40, end=50).rename(r"$\alpha=0.2$")

    # forecast(30).rename(r"$\alpha=0.2$")

    fit2 = SimpleExpSmoothing(df, initialization_method="estimated").fit()
    fcast2 = fit2.predict(start=40, end=50).rename(
        r"$\alpha=%s$" % fit2.model.params["smoothing_level"]
    )

    plt.figure(figsize=(12, 8))
    plt.plot(df, marker="o", color="black")

    plt.plot(fit1.fittedvalues, marker="o", color="lightblue")
    (line1,) = plt.plot(fcast1, marker="o", color="blue")

    plt.plot(fit2.fittedvalues, marker="o", color="lightgreen")
    (line2,) = plt.plot(fcast2, marker="o", color="green")

    plt.legend([line1, line2], [fcast1.name, fcast2.name])
    plt.show()


def problem_4_2():
    df = pd.read_excel("HW3_data_1.xlsx", sheet_name=0)
    df = df.set_index("Period")
    optimal_alpha = find_optimal_alpha_exponential_smoothing(
        df, column="Yt", file_name="figs/problem_4_2-2.png"
    )

    fig, ax = plt.subplots()
    df["Yt"].plot(ax=ax, label="Yt")
    smooth_first_n(df, optimal_alpha, n=40, column="Yt").plot(ax=ax)
    n_step_ahead_forecast_last_k(
        df,
        smoothing_level=optimal_alpha,
        n_steps=1,
        k_forecasts=10,
        column="Yt",
    ).plot(ax=ax, label="Forecast", color="red")
    plt.legend()
    plt.savefig("figs/problem_4_2.png")


def problem_4_3():
    df = pd.read_excel("HW3_data_1.xlsx", sheet_name=0)
    df = df.set_index("Period")
    sample_ACF = find_sample_ACF(df, 1)
    print(f"Sample ACF: {sample_ACF}")


def problem_4_10():
    df = pd.read_excel("HW3_data_2.xlsx", sheet_name=2)
    df = df.set_index("Month")

    fig, ax = plt.subplots()
    df["Rate"].plot(ax=ax, label="Rate")
    smooth_first_n(df, 0.2, n=627, column="Rate").plot(ax=ax)
    n_step_ahead_forecast_last_k(
        df,
        smoothing_level=0.2,
        n_steps=1,
        k_forecasts=20,
        column="Rate",
    ).plot(ax=ax, label="Forecast", color="red")
    plt.legend()
    plt.savefig("figs/problem_4_10.png")


def problem_4_20():
    df = pd.read_excel("HW3_data_2.xlsx", sheet_name=7)
    df = df.set_index("Year")
    column = "Anomaly, C"
    optimal_alpha = find_optimal_alpha_exponential_smoothing(
        df, column=column, file_name="figs/problem_4_20-2.png"
    )

    fig, ax = plt.subplots()
    df[column].plot(ax=ax, label=column)
    smooth_first_n(df, 0.2, n=125, column=column).plot(ax=ax)
    # n_step_ahead_forecast_last_k(
    #     df,
    #     smoothing_level=optimal_alpha,
    #     n_steps=1,
    #     k_forecasts=10,
    #     column=column,
    # ).plot(ax=ax, label="Forecast", color="red")
    plt.legend()
    # plt.savefig("figs/problem_4_20.png")


def problem_4_48():
    pass


def problem_4_52():
    pass


if __name__ == "__main__":
    forcast_using_stats_model()
    problem_4_1()
    # problem_4_2()
    # problem_4_3()
    # problem_4_10()

    # problem_4_20()
    # problem_4_48()
    # problem_4_52()
