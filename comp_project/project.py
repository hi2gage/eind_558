import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import missingno as mno

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

EPOCHS = 200
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

SPLIT = 0.90  # train/test %

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

# plt.figure(100, figsize=(20, 7))
# sns.lineplot(x="time", y="count", data=df1, palette="coolwarm")
# plt.show()
# os.abort()
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

# check correlations of features with price
df2 = df2.drop("registered", axis="columns")
df2 = df2.drop("casual", axis="columns")

df_corr = df2.corr(method="pearson")
print(df_corr.shape)
print("correlation with count:")
df_corrP = pd.DataFrame(df_corr["count"].sort_values(ascending=False))
print(df_corrP)

# highest absolute correlations with count
print("correlation with count > 0.15:")
pd.options.display.float_format = "{:,.2f}".format
df_corrH = df_corrP[np.abs(df_corrP["count"]) > 0.10]
print(df_corrH)


df3 = df2[df_corrH.index]
idx = df3.corr().sort_values("count", ascending=False).index
df3_sorted = df3.loc[:, idx]
# sort dataframe columns by their correlation with Appliances

# plt.figure(figsize=(15, 15))
# sns.set(font_scale=0.75)
# ax = sns.heatmap(
#     df3_sorted.corr().round(3),
#     annot=True,
#     square=True,
#     linewidths=0.75,
#     cmap="coolwarm",
#     fmt=".2f",
#     annot_kws={"size": 11},
# )
# ax.xaxis.tick_bottom()
# plt.title("correlation matrix")
# plt.show()


# additional datetime columns: feature engineering
df3["month"] = df3.index.month

df3["wday"] = df3.index.dayofweek
dict_days = {
    0: "1_Mon",
    1: "2_Tue",
    2: "3_Wed",
    3: "4_Thu",
    4: "5_Fri",
    5: "6_Sat",
    6: "7_Sun",
}
df3["weekday"] = df3["wday"].apply(lambda x: dict_days[x])

df3["hour"] = df3.index.hour

df3 = df3.astype({"hour": float, "wday": float, "month": float})

df3.iloc[[0, -1]]


piv = pd.pivot_table(
    df3,
    values="count",
    index="month",
    columns="weekday",
    aggfunc="mean",
    margins=True,
    margins_name="Avg",
    fill_value=0,
)
pd.options.display.float_format = "{:,.0f}".format

plt.figure(figsize=(10, 15))
sns.set(font_scale=1)
sns.heatmap(
    piv.round(0),
    annot=True,
    square=True,
    linewidths=0.75,
    cmap="coolwarm",
    fmt=".0f",
    annot_kws={"size": 11},
)
plt.title("Count by weekday by month")
# plt.show()


# pivot table: hours in weekdays
piv = pd.pivot_table(
    df3,
    values="count",
    index="hour",
    columns="weekday",
    aggfunc="mean",
    margins=True,
    margins_name="Avg",
    fill_value=0,
)
pd.options.display.float_format = "{:,.0f}".format

plt.figure(figsize=(7, 20))
sns.set(font_scale=1)
sns.heatmap(
    piv.round(0),
    annot=True,
    square=True,
    linewidths=0.75,
    cmap="coolwarm",
    fmt=".0f",
    annot_kws={"size": 11},
)
plt.title("Count by hour by weekday")
# plt.show()


# dataframe with price and features only
df4 = df3.copy()
df4.drop(["weekday", "month", "wday", "hour"], inplace=True, axis=1)


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


scalerP = Scaler()
scalerP.fit_transform(ts_train)
ts_ttrain = scalerP.transform(ts_train)
ts_ttest = scalerP.transform(ts_test)
ts_t = scalerP.transform(ts_P)

# make sure data are of type float
ts_t = ts_t.astype(np.float32)
ts_ttrain = ts_ttrain.astype(np.float32)
ts_ttest = ts_ttest.astype(np.float32)

print("first and last row of scaled price time series:")
pd.options.display.float_format = "{:,.2f}".format
print(ts_t.pd_dataframe().iloc[[0, -1]])


# train/test split and scaling of feature covariates
covF_train, covF_test = ts_covF.split_after(SPLIT)

scalerF = Scaler()
scalerF.fit_transform(covF_train)
covF_ttrain = scalerF.transform(covF_train)
covF_ttest = scalerF.transform(covF_test)
covF_t = scalerF.transform(ts_covF)

# make sure data are of type float
covF_ttrain = ts_ttrain.astype(np.float32)
covF_ttest = ts_ttest.astype(np.float32)

pd.options.display.float_format = "{:.2f}".format
print("first and last row of scaled feature covariates:")
print(covF_t.pd_dataframe().iloc[[0, -1]])


# feature engineering - create time covariates: hour, weekday, month, year, country-specific holidays
covT = datetime_attribute_timeseries(
    ts_P.time_index,
    attribute="day",
    until=pd.Timestamp("2012-11-01 00:00:00"),
    one_hot=False,
)
covT = covT.stack(
    datetime_attribute_timeseries(covT.time_index, attribute="day_of_week", one_hot=False)
)
covT = covT.stack(
    datetime_attribute_timeseries(covT.time_index, attribute="month", one_hot=False)
)
covT = covT.stack(
    datetime_attribute_timeseries(covT.time_index, attribute="year", one_hot=False)
)

covT = covT.add_holidays(country_code="ES")
covT = covT.astype(np.float32)
covT.drop_columns(["day"])


# train/test split
covT_train, covT_test = covT.split_after(SPLIT)
print("covT_train")
print(covT_train)


# rescale the covariates: fitting on the training set
scalerT = Scaler()
scalerT.fit(covT_train)
covT_ttrain = scalerT.transform(covT_train)
covT_ttest = scalerT.transform(covT_test)
covT_t = scalerT.transform(covT)

covT_t = covT_t.astype(np.float32)
print(covT_t)


pd.options.display.float_format = "{:.0f}".format
print("first and last row of unscaled time covariates:")
print(covT.pd_dataframe().iloc[[0, -1]])

# combine feature and time covariates along component dimension: axis=1
ts_cov = ts_covF.concatenate(covT.slice_intersect(ts_covF), axis=1)  # unscaled F+T
cov_t = covF_t.concatenate(covT_t.slice_intersect(covF_t), axis=1)  # scaled F+T
cov_t = cov_t.astype(np.float32)
print("cov_t")
print(cov_t)
print("====")
# cov_ttrain = covF_ttrain.concatenate(
#     covT_ttrain.slice_intersect(covF_ttrain), axis=1
# )  # scaled F+T training set
# cov_ttest = covF_ttest.concatenate(
#     covT_ttest.slice_intersect(covF_ttest), axis=1
# )  # scaled F+T test set


pd.options.display.float_format = "{:.2f}".format
print("first and last row of unscaled covariates:")
ts_cov.pd_dataframe().iloc[[0, -1]]

model = TransformerModel(
    input_chunk_length=INLEN,
    output_chunk_length=N_FC,
    batch_size=BATCH,
    n_epochs=EPOCHS,
    model_name="Transformer_count",
    nr_epochs_val_period=VALWAIT,
    d_model=FEAT,
    nhead=HEADS,
    num_encoder_layers=ENCODE,
    num_decoder_layers=DECODE,
    dim_feedforward=DIM_FF,
    dropout=DROPOUT,
    activation=ACTF,
    random_state=RAND,
    likelihood=QuantileRegression(quantiles=QUANTILES),
    optimizer_kwargs={"lr": LEARN},
    add_encoders={"cyclic": {"future": ["hour", "dayofweek", "month"]}},
    save_checkpoints=True,
    force_reset=True,
)

print("s_ttrain.shape:")
print(ts_ttrain)


# training: load a saved model or (re)train
if LOAD:
    print("have loaded a previously saved model from disk:" + mpath)
    model = TransformerModel.load_model(mpath)  # load previously model from disk
else:
    model.fit(ts_ttrain, past_covariates=cov_t, verbose=True)
    print("have saved the model after training:", mpath)
    model.save_model(mpath)

# testing: generate predictions
ts_tpred = model.predict(
    n=len(ts_ttest), num_samples=N_SAMPLES, n_jobs=N_JOBS, verbose=True
)
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
sns.lineplot(x="time", y="Actual", data=dfY, palette="r")
# sns.lineplot(x="time", y="Training", data=dfY, palette="coolwarm")

plt.legend(labels=["forecast median count Q50", "actual count"])
p.set_ylabel("price")
p.set_xlabel("")
p.set_title("Bike Rental Count per Day (test set)")
plt.show()
