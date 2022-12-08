import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import missingno as mno

from statsmodels.tsa.arima.model import ARIMA

# import warnings

# warnings.filterwarnings("ignore")
# import logging

# logging.disable(logging.CRITICAL)


from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TransformerModel
from darts.models import NBEATSModel
from darts.metrics import mape, rmse
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression


pd.set_option("display.precision", 2)
np.set_printoptions(precision=2, suppress=True)
pd.options.display.float_format = "{:,.2f}".format


LOAD = False  # True = load previously saved model from disk?  False = (re)train the model
SAVE = "\_TForm_model10e.pth.tar"  # file name to save the model under

EPOCHS = 15
INLEN = 32  # input size
FEAT = 32  # d_model = number of expected features in the inputs, up to 512
HEADS = 4  # default 8
ENCODE = 4  # encoder layers
DECODE = 4  # decoder layers
DIM_FF = 128  # dimensions of the feedforward network, default 2048
BATCH = 32  # batch size
ACTF = "relu"  # activation function, relu (default) or gelu
SCHLEARN = None  # a PyTorch learning rate scheduler; None = constant rate
LEARN = 1e-3  # learning rate
VALWAIT = 1  # epochs to wait before evaluating the loss on the test/validation set
DROPOUT = 0.1  # dropout rate
N_FC = 1  # output size

RAND = 42  # random seed
N_SAMPLES = 100  # number of times a prediction is sampled from a probabilistic model
N_JOBS = 3  # parallel processors to use;  -1 = all processors

# default quantiles for QuantileRegression
QUANTILES = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99]

SPLIT = 0.9  # train/test %

FIGSIZE = (9, 6)


qL1, qL2 = 0.01, 0.10  # percentiles of predictions: lower bounds
qU1, qU2 = (
    1 - qL1,
    1 - qL2,
)  # upper bounds derived from lower bounds
label_q1 = f"{int(qU1 * 100)} / {int(qL1 * 100)} percentile band"
label_q2 = f"{int(qU2 * 100)} / {int(qL2 * 100)} percentile band"

mpath = os.path.abspath(os.getcwd()) + SAVE  # path and file name to save the model


df1 = pd.read_csv("Data/all_data.csv", header=0, parse_dates=["datetime"])
# datetime
df1["datetime"] = pd.to_datetime(df1["datetime"], infer_datetime_format=True)
df1.set_index("datetime", inplace=True)

df1.index.rename("time", inplace=True)


print(df1)

print(df1.info())
print(df1.describe())
df_resampled = df1.copy()
df_resampled = df1.resample("1D").agg(
    {
        "season": "mean",
        "holiday": "mean",
        "workingday": "mean",
        "weather": "mean",
        "temp": "mean",
        "atemp": "mean",
        "humidity": "mean",
        "windspeed": "mean",
        "casual": "sum",
        "registered": "sum",
        "count": "sum",
    }
)
df_resampled = df_resampled.truncate(after="2012-10-27")


print(df_resampled)
print(df_resampled.info())
print(df_resampled.describe())


# print("count of duplicates:", df1.duplicated(subset=["time"], keep="first").sum())


df1_outlier = df_resampled.copy()
print("=============================================")
print("count of outliers: largest 10")
print(df1_outlier["count"].nlargest(20))

print("count of outliers: smallest 10")
print(df1_outlier["count"].nsmallest(20))


df1_outlier["count"].where(df1_outlier["count"] >= 500, inplace=True)
df1_outlier["count"].where(df1_outlier["count"] != 1009, inplace=True)
# df1_outlier["pressure"].where(df1_outlier["pressure"] >= 948, inplace=True)

# df1_outlier["wind_speed"].where(df1_outlier["wind_speed"] <= 120, inplace=True)
# df1_outlier["clouds_all"].where(df1_outlier["clouds_all"] <= 40, inplace=True)

df1_outlier = df1_outlier.interpolate(method="bfill")

print("=============================================")
print("count of outliers: largest 10")
print(df1_outlier["count"].nlargest(20))

print("count of outliers: smallest 10")
print(df1_outlier["count"].nsmallest(20))


print(df1_outlier)
print(df1_outlier.info())
print(df1_outlier.describe())


# plt.figure(100, figsize=(20, 7))
# sns.lineplot(x="time", y="count", data=df1_outlier, palette="coolwarm")
# plt.show()
# os.abort()

df2 = df1_outlier.copy()
# any null values?
print("any missing values?", df2.isnull().values.any())

# any ducplicate time periods?
print("count of duplicates:", df2.duplicated(keep="first").sum())

print("++++++++++++++++++++++++++++++++++++++++++++++==")

df4 = df2.copy()

# create time series object for target variable
ts_P = TimeSeries.from_series(df4["count"], fill_missing_dates=True)

# check attributes of the time series
print("components:", ts_P.components)
print("duration:", ts_P.duration)
print("frequency:", ts_P.freq)
print("frequency:", ts_P.freq_str)
print(
    "has date time index? (or else, it must have an integer index):",
    ts_P.has_datetime_index,
)
print("deterministic:", ts_P.is_deterministic)
print("univariate:", ts_P.is_univariate)

# create time series object for the feature columns
df_covF = df4.loc[:, df4.columns != "count"]
ts_covF = TimeSeries.from_dataframe(df_covF, fill_missing_dates=True)

print("=====================================")

# check attributes of the time series
print("components (columns) of feature time series:", ts_covF.components)
print("duration:", ts_covF.duration)
print("frequency:", ts_covF.freq)
print("frequency:", ts_covF.freq_str)
print(
    "has date time index? (or else, it must have an integer index):",
    ts_covF.has_datetime_index,
)
print("deterministic:", ts_covF.is_deterministic)


# example: operating with time series objects:
# we can also create a 3-dimensional numpy array from a time series object
# 3 dimensions: time (rows) / components (columns) / samples
ar_covF = ts_covF.all_values()
print(type(ar_covF))
ar_covF.shape


# example: operating with time series objects:
# we can also create a pandas series or dataframe from a time series object
df_covF = ts_covF.pd_dataframe()
type(df_covF)

# train/test split and scaling of target variable
ts_train, ts_test = ts_P.split_after(SPLIT)
print("training start:", ts_train.start_time())
print("training end:", ts_train.end_time())
print("training duration:", ts_train.duration)
print("test start:", ts_test.start_time())
print("test end:", ts_test.end_time())
print("test duration:", ts_test.duration)


df3 = df2["count"]
# print(df3)

model = ARIMA(df2["count"], order=(2, 1, 0))
model_fit = model.fit()
print(model_fit.summary())

# ts_tpred = model.predict(n=len(ts_test))
ts_tpred = model.predict(start="2012-09-01", end="2012-10-01")
print(ts_tpred)


# retrieve forecast series for chosen quantiles,
# inverse-transform each series,
# insert them as columns in a new dataframe dfY
q50_RMSE = np.inf
q50_MAPE = np.inf
ts_q50 = None
pd.options.display.float_format = "{:,.2f}".format
dfY = pd.DataFrame()
dfY["Actual"] = TimeSeries.pd_series(ts_test)


# helper function: get forecast values for selected quantile q and insert them in dataframe dfY
def predQ(ts_t, q):
    ts_tq = ts_t.quantile_timeseries(q)
    ts_q = scalerP.inverse_transform(ts_tq)
    s = TimeSeries.pd_series(ts_q)
    header = "Q" + format(int(q * 100), "02d")
    dfY[header] = s
    if q == 0.5:
        ts_q50 = ts_q
        q50_RMSE = rmse(ts_q50, ts_test)
        q50_MAPE = mape(ts_q50, ts_test)
        print("RMSE:", f"{q50_RMSE:.2f}")
        print("MAPE:", f"{q50_MAPE:.2f}")


# call helper function predQ, once for every quantile
_ = [predQ(ts_tpred, q) for q in QUANTILES]

# move Q50 column to the left of the Actual column
col = dfY.pop("Q50")
dfY.insert(1, col.name, col)
print(dfY.iloc[np.r_[0:2, -2:0]])


print(dfY)

# plot the forecast
plt.figure(100, figsize=(20, 7))
sns.set(font_scale=1.3)

p = sns.lineplot(x="time", y="Q50", data=dfY, palette="coolwarm")
sns.lineplot(x="time", y="Actual", data=dfY, palette="coolwarm")
# sns.lineplot(x="time", y="Training", data=dfY, palette="coolwarm")

plt.legend(labels=["forecast median count Q50", "actual count"])
p.set_ylabel("Count")
p.set_xlabel("")
p.set_title("Biker Count (test set)")
plt.show()
