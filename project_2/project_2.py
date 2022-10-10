import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from statsmodels.graphics import tsaplots
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pylab import rcParams
import datetime
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.stats import boxcox
from datetime import datetime, timedelta


import warnings

warnings.filterwarnings("ignore")


def main(show=True):
    df = pd.read_excel(
        "Project_2_TSLA_Data.xlsx", sheet_name=0, usecols=["Date", "Close"]
    )
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df = df.reindex(pd.date_range(df.index.min(), df.index.max())).sort_index(
        ascending=True
    )
    df.index.rename("Date", inplace=True)

    print(df.head(20))

    df = df.assign(
        Close_Linear_Interpolation=df.Close.interpolate(method="linear")
    )
    df["Close"] = df["Close_Linear_Interpolation"]
    df.drop(columns=["Close_Linear_Interpolation"], inplace=True)
    df.sort_index(inplace=True)
    print(df.head(20))

    # df.plot(figsize=(20, 5))
    # plt.grid()
    # plt.legend(loc="best")
    # plt.title("Close Price")

    # rcParams["figure.figsize"] = 20, 10

    # decomposition = sm.tsa.seasonal_decompose(
    #     df.Close, model="additive", period=31
    # )  # additive seasonal index
    # fig = decomposition.plot()
    # plt.show()

    close_data = df["Close"]
    close_data_14 = close_data.rolling(window=10).mean()

    # df["Close"].plot(figsize=(20, 5))
    # plt.plot(close_data_14, "r-", label="Running average 14 Days")
    # # plt.xlabel("Date")
    # plt.grid(linestyle=":")
    # # plt.fill_between(co2.index, 0, co2, color="r", alpha=0.1)
    # plt.legend(loc="upper left")
    # plt.show()

    # Part 1
    train_start = "12-26-2019"
    train_end = "12-31-2020"
    test_end = "4-10-2021"

    # Part 3
    # train_start = "1-1-2021"
    # train_end = "7-31-2021"
    # test_end = "9-25-2021"

    train_end_minus_one = datetime.strptime(train_end, "%M-%d-%Y") - timedelta(
        days=1
    )
    train_end_minus_one = train_end_minus_one.strftime("%M-%d-%Y")

    test_end_minus_one = datetime.strptime(test_end, "%M-%d-%Y") - timedelta(
        days=1
    )
    # test_end_minus_one = test_end_minus_one.strftime("%-d-%M-%Y")

    print(train_end)
    print(train_end_minus_one)

    train = df[train_start:train_end]
    # train = df["2019-12-26":"2020-12-31"]
    train_len = train.shape[0]
    test = df[train_end:test_end]

    data_boxcox = pd.Series(boxcox(df["Close"], lmbda=0), index=df.index)

    # plt.grid()
    # plt.plot(data_boxcox, label="After Box Cox tranformation")
    # plt.legend(loc="best")
    # plt.title("After Box Cox transform")
    # plt.show()

    data_boxcox_diff = pd.Series(data_boxcox - data_boxcox.shift(), df.index)
    # plt.figure(figsize=(20, 5))
    # plt.grid()
    # plt.plot(
    #     data_boxcox_diff, label="After Box Cox tranformation and differencing"
    # )
    # plt.legend(loc="best")
    # plt.title("After Box Cox transform and differencing")
    # plt.show()

    data_boxcox_diff.dropna(inplace=True)

    train_data_boxcox = data_boxcox[train_start:train_end]
    test_data_boxcox = data_boxcox[train_end:]
    train_data_boxcox_diff = data_boxcox_diff[:train_end_minus_one]
    test_data_boxcox_diff = data_boxcox_diff[train_end_minus_one:test_end]

    model = sm.tsa.arima.ARIMA(train_data_boxcox_diff, order=(0, 0, 14))
    model_fit = model.fit()
    print(model_fit.params)

    y_hat_ma = data_boxcox_diff.copy()
    y_hat_ma["ma_forecast_boxcox_diff"] = model_fit.predict(
        data_boxcox_diff.index.min(), data_boxcox_diff.index.max()
    )
    y_hat_ma["ma_forecast_boxcox"] = y_hat_ma[
        "ma_forecast_boxcox_diff"
    ].cumsum()
    y_hat_ma["ma_forecast_boxcox"] = y_hat_ma["ma_forecast_boxcox"].add(
        data_boxcox[0]
    )
    y_hat_ma["ma_forecast"] = np.exp(y_hat_ma["ma_forecast_boxcox"])

    plt.figure(figsize=(20, 5))

    plt.grid()
    plt.plot(df["Close"][train_start:train_end], label="Train")
    plt.plot(df["Close"][train_end:test_end], label="Test")
    plt.plot(
        y_hat_ma["ma_forecast"][test.index.min() : test_end],
        label="Moving average forecast",
    )
    plt.legend(loc="best")
    plt.title("Moving Average Method")
    plt.show()

    rmse = np.sqrt(
        mean_squared_error(
            test["Close"], y_hat_ma["ma_forecast"][test.index.min() : test_end]
        )
    ).round(2)
    mape = np.round(
        np.mean(
            np.abs(
                test["Close"]
                - y_hat_ma["ma_forecast"][test.index.min() : test_end]
            )
            / test["Close"]
        )
        * 100,
        2,
    )

    results = pd.DataFrame(
        {
            "Method": ["Moving Average (MA) method"],
            "RMSE": [rmse],
            "MAPE": [mape],
        }
    )
    results = results[["Method", "RMSE", "MAPE"]]
    print(results)


if __name__ == "__main__":
    main()
