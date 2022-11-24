import pandas as pd
import numpy as np
import pmdarima as pm
import matplotlib.pyplot as plt
from pmdarima.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA


def data_cleaning():
    data_frames = []
    facebook_df = pd.read_excel(
        "Project_3_Data.xlsx", sheet_name=1, usecols=["Date", "Close"]
    )

    data_frames.append(("facebook", facebook_df))

    apple_df = pd.read_excel(
        "Project_3_Data.xlsx", sheet_name=2, usecols=["Date", "Close"]
    )
    data_frames.append(("apple", apple_df))

    amazon_df = pd.read_excel(
        "Project_3_Data.xlsx", sheet_name=3, usecols=["Date", "Close"]
    )
    data_frames.append(("amazon", amazon_df))

    netflix_df = pd.read_excel(
        "Project_3_Data.xlsx", sheet_name=4, usecols=["Date", "Close"]
    )
    data_frames.append(("netflix", netflix_df))

    google_df = pd.read_excel(
        "Project_3_Data.xlsx", sheet_name=5, usecols=["Date", "Close"]
    )
    data_frames.append(("google", google_df))

    boeing_df = pd.read_excel(
        "Project_3_Data.xlsx", sheet_name=6, usecols=["Date", "Close"]
    )
    data_frames.append(("boeing", boeing_df))

    companies_interp = []
    for company in data_frames:
        df = company[1]
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df = df.reindex(
            pd.date_range(df.index.min(), df.index.max())
        ).sort_index(ascending=True)
        df.index.rename("Date", inplace=True)
        df = df.assign(
            Close_Linear_Interpolation=df.Close.interpolate(method="linear")
        )
        df["Close"] = df["Close_Linear_Interpolation"]
        df.drop(columns=["Close_Linear_Interpolation"], inplace=True)
        df = df.rename(columns={"Close": f"{company[0]}_close"})
        df.sort_index(inplace=True)
        print(df.head(20))
        companies_interp.append(df)

    df = pd.concat(companies_interp, axis=1)
    df = df.round(6)

    df["index_close"] = (
        df["facebook_close"]
        + df["apple_close"]
        + df["amazon_close"]
        + df["netflix_close"]
        + df["google_close"]
        + df["boeing_close"]
    )
    mask = df.index < "2022-11-8"
    df = df.iloc[mask]

    print(df)
    df.to_csv("project_3_data_combined.csv")


def auto(companies, all_df):
    fig, axs = plt.subplots(3, 2)
    plt.margins(0, 0)

    d_values = [1, 2, 1, 2, 1, 1]

    for i, company in enumerate(companies):
        print(company)
        row = int(i / 3)
        column = i % 3
        d = d_values[i]

        df = all_df[company]

        train, test = train_test_split(df, test_size=7)

        # Fit your model
        model = pm.auto_arima(train, seasonal=True, m=7, d=d)

        # make your forecasts
        forecasts = model.predict(test.shape[0]).rename(str(model)[1:13])

        df.plot(
            ax=axs[column, row],
            legend=True,
            color="black",
            title=f"{company} Stock Price Prediction",
        )

        axs[column, row].set_ylabel("Stock Price ($)")
        axs[column, row].set_xlabel("Date")

        fittedvalues = model.fittedvalues()
        fittedvalues = fittedvalues.drop(fittedvalues.index[0])
        fittedvalues = fittedvalues.drop(fittedvalues.index[0])
        fittedvalues.plot(ax=axs[column, row], style="-", color="orange")

        forecasts.plot(
            ax=axs[column, row], style="-", color="red", legend=True
        )
        # model.plot_diagnostics(figsize=(15, 12))
        # print(model.summary())

        mse = 0
        for i in range(len(forecasts)):
            mse += (forecasts[i] - test[i]) ** 2
        print(f"Mean Squared Error: {mse / len(forecasts)}")
        print()

    # plt.show()


def manual(companies, all_df):
    fig, axs = plt.subplots(3, 2)
    plt.margins(0, 0)

    d_values = [1, 2, 1, 2, 1, 1]

    for i, company in enumerate(companies):
        print(int(i / 3))
        print(i % 3)

        row = int(i / 3)
        column = i % 3
        d = d_values[i]
        df = all_df[company]

        df.plot(
            ax=axs[column, row],
            legend=True,
            color="black",
            title=f"{company} Stock Price Prediction",
        )
        axs[column, row].set_ylabel("Stock Price ($)")
        axs[column, row].set_xlabel("Date")

        # print(df.head(5))

        mask = all_df.index < "2022-11-1"
        model = ARIMA(df.iloc[mask], order=(1, 1, 0))
        fit1 = model.fit()

        fittedvalues = fit1.fittedvalues
        fittedvalues = fittedvalues.drop(fittedvalues.index[0])

        fittedvalues.plot(ax=axs[column, row], style="-", color="orange")

        fit1_values = fit1.predict(start="2022-11-01", end="2022-11-7").rename(
            "model = ARIMA(df.value, order=(4, 1, 2))"
        )

        fit1_values.plot(
            ax=axs[column, row], style="-", color="red", legend=True
        )

    plt.show()


def index(companies, all_df):
    fig, axs = plt.subplots(1, 1)
    plt.margins(0, 0)

    print("index_close")

    df = all_df["index_close"]

    train, test = train_test_split(df, test_size=7)

    # Fit your model
    model = pm.auto_arima(train, seasonal=False, m=1, d=1)

    # make your forecasts
    forecasts = model.predict(test.shape[0]).rename(str(model)[1:13])

    df.plot(
        ax=axs,
        legend=True,
        color="black",
        title=f"index_close Stock Price Prediction",
    )

    axs.set_ylabel("Stock Price ($)")
    axs.set_xlabel("Date")

    fittedvalues = model.fittedvalues()
    fittedvalues = fittedvalues.drop(fittedvalues.index[0])
    fittedvalues = fittedvalues.drop(fittedvalues.index[0])
    fittedvalues.plot(ax=axs, style="-", color="orange")

    forecasts.plot(ax=axs, style="-", color="red", legend=True)
    model.plot_diagnostics(figsize=(15, 12))
    print(model.summary())

    mse = 0
    for i in range(len(forecasts)):
        mse += (forecasts[i] - test[i]) ** 2
    print(f"Mean Squared Error: {mse / len(forecasts)}")
    print()

    # plt.show()


def main():
    # data_cleaning()
    all_df = pd.read_csv("project_3_data_combined.csv")
    all_df["Date"] = pd.to_datetime(all_df["Date"], format="%Y-%m-%d")
    all_df["Date"] = pd.to_datetime(all_df["Date"])
    all_df.set_index("Date", inplace=True)
    all_df = all_df.reindex(
        pd.date_range(all_df.index.min(), all_df.index.max())
    ).sort_index(ascending=True)
    all_df.index.rename("Date", inplace=True)

    print(all_df.tail(14))

    # General Variables
    mask = all_df.index < "2022-10-1"
    companies = [
        "facebook_close",
        "apple_close",
        "amazon_close",
        "netflix_close",
        "google_close",
        "boeing_close",
        # "index_close",
    ]

    auto(companies, all_df)
    index(companies, all_df)

    # fig, axs = plt.subplots(3, 4)
    # plt.margins(0, 0)

    # for i, company in enumerate(companies):
    #     print(int(i / 3))
    #     print(i % 3)

    #     row = int(i / 3)
    #     if row == 1:
    #         row = 2
    #     column = i % 3
    #     d = d_values[i]

    #     df = all_df[company]
    #     # print(df.head(5))
    #     axs[column, row + 1].plot(df.diff())
    #     plot_acf(df, ax=axs[column, row])
    # plt.gca().set_axis_off()
    # plt.subplots_adjust(
    #     top=1, bottom=0, right=1, left=0, hspace=0.1, wspace=0.1
    # )
    # plt.margins(0.1, 0.1)
    plt.show()


if __name__ == "__main__":
    main()
    # data_cleaning()
