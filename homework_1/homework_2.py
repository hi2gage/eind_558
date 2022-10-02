from random import randint
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm


def histograms(df):
    df.plot.hist(
        column="Calcium(mg)", bins=100, range=(0, 3000)
    ).get_figure().savefig("Calcium.png")

    df.plot.hist(
        column="Iron(mg)", bins=100, range=(0, 100)
    ).get_figure().savefig("Iron.png")

    df.plot.hist(
        column="Protein(g)", bins=100, range=(0, 300)
    ).get_figure().savefig("Protein.png")
    df.plot.hist(
        column="Vitamin A(μg)", bins=100, range=(0, 35000)
    ).get_figure().savefig("Vitamin A.png")
    df.plot.hist(
        column="Vitamin C(mg)", bins=100, range=(0, 500)
    ).get_figure().savefig("Vitamin C.png")


def correlation(df_train):
    corrmat = df_train.corr(method="spearman")
    f, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corrmat, ax=ax, cmap="YlGnBu", linewidths=0.1
    ).get_figure().savefig("correlation.png")


def question_nine():
    df = pd.read_csv(
        "homework_1/nutrient.csv",
        header=None,
        names=[
            "Calcium(mg)",
            "Iron(mg)",
            "Protein(g)",
            "Vitamin A(μg)",
            "Vitamin C(mg)",
        ],
    )
    print(df)
    print(df.describe())

    histograms(df)
    correlation(df)
    print(df.corr())


def question_ten_1():
    df = pd.read_csv("homework_1/chemical_process.csv")
    print(df.describe())
    x = df[["x6", "x7"]]
    y = df["y"]
    regr = linear_model.LinearRegression()
    regr.fit(x, y)

    print("Intercept: \n", regr.intercept_)
    print("Coefficients: \n", regr.coef_)

    # with statsmodele
    x = sm.add_constant(x)  # adding a constant

    model = sm.OLS(y, x).fit()
    predictions = model.predict(x)
    print(model.conf_int(alpha=0.05))

    print_model = model.summary()
    print(print_model)


def question_ten_2():
    df = pd.read_csv("homework_1/chemical_process.csv")
    print(df.describe())
    x = df[["x1", "x2", "x3", "x4", "x5", "x6", "x7"]]
    y = df["y"]
    regr = linear_model.LinearRegression()
    regr.fit(x, y)

    print("Intercept: \n", regr.intercept_)
    print("Coefficients: \n", regr.coef_)

    # with statsmodele
    x = sm.add_constant(x)  # adding a constant

    model = sm.OLS(y, x).fit()
    predictions = model.predict(x)
    print(model.conf_int(alpha=0.05))

    print_model = model.summary()
    print(print_model)


if __name__ == "__main__":
    question_ten_2()
