import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.statespace.tools import diff
import statsmodels.api as sm


def problem_5_3():
    df = pd.read_excel("hw_4_data.xlsx", sheet_name=0)
    df = df.set_index("Period")
    print(df.shape)
    df.plot()

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    sm.graphics.tsa.plot_acf(df.values.squeeze(), lags=24, ax=ax[0])
    sm.graphics.tsa.plot_pacf(df.values.squeeze(), lags=24, ax=ax[1])
    plt.show()
    # plt.savefig("figs/series_plot_5_3.png")


def problem_5_4():
    pass


def problem_5_10():
    pass


def problem_5_16():
    pass


def problem_5_17():
    pass


def problem_5_24():
    pass


def problem_5_50():
    pass


if __name__ == "__main__":
    problem_5_3()
