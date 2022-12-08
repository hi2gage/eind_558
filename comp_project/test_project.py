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
from darts.metrics import mape, rmse
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression


# pd.set_option("display.precision", 2)
# np.set_printoptions(precision=2, suppress=True)
# pd.options.display.float_format = "{:,.2f}".format


LOAD = False  # True = load previously saved model from disk?  False = (re)train the model
SAVE = "\_test_TForm_model10e.pth.tar"  # file name to save the model under

EPOCHS = 5
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


# load
# df0 = pd.read_csv("data/archive-2/energy_dataset.csv", header=0, parse_dates=["time"])

df0_list = []
filename = "data/archive-2/energy_dataset.csv"
lines_number = sum(1 for line in open(filename))
lines_in_chunk = 500  # I don't know what size is better
counter = 0
completed = 0
reader = pd.read_csv(filename, chunksize=lines_in_chunk)
for chunk in reader:
    df0_list.append(chunk)
    # < ... reading the chunk somehow... >
    # showing progress:
    counter += lines_in_chunk
    new_completed = int(round(float(counter) / lines_number * 100))
    if new_completed > completed:
        completed = new_completed
        print("Completed", completed, "%")

df0 = pd.concat(df0_list, ignore_index=True)
print(df0.shape)

# dfw0 = pd.read_csv(
#     "data/archive-2/weather_features.csv", header=0, parse_dates=["dt_iso"]
# )

dfw0_list = []
filename = "data/archive-2/weather_features.csv"
lines_number = sum(1 for line in open(filename))
lines_in_chunk = 500  # I don't know what size is better
counter = 0
completed = 0
reader = pd.read_csv(filename, chunksize=lines_in_chunk)
for chunk in reader:
    dfw0_list.append(chunk)
    # < ... reading the chunk somehow... >
    # showing progress:
    counter += lines_in_chunk
    new_completed = int(round(float(counter) / lines_number * 100))
    if new_completed > completed:
        completed = new_completed
        print("Completed", completed, "%")

dfw0 = pd.concat(dfw0_list, ignore_index=True)
print(df0.shape)

df1 = df0.copy()
dfw1 = dfw0.copy()

# datetime
df1["time"] = pd.to_datetime(df1["time"], utc=True, infer_datetime_format=True)


# any duplicate time periods?
print("count of duplicates:", df1.duplicated(subset=["time"], keep="first").sum())


df1.set_index("time", inplace=True)


# any non-numeric types?
print("non-numeric columns:", list(df1.dtypes[df1.dtypes == "object"].index))


# any missing values?
def gaps(df):
    if df.isnull().values.any():
        print("MISSING values:\n")
        mno.matrix(df)
    else:
        print("no missing values\n")


# drop the NaN and zero columns, and also the 'forecast' columns
df1 = df1.drop(df1.filter(regex="forecast").columns, axis=1, errors="ignore")
df1.dropna(axis=1, how="all", inplace=True)
df1 = df1.loc[:, (df1 != 0).any(axis=0)]


# handle missing values in rows of remaining columns
df1 = df1.interpolate(method="bfill")
# any missing values left?
gaps(df1)

df1 = df1.loc[:, (df1 != 0).any(axis=0)]


# rename columns
colnames_old = df1.columns
colnames_new = [
    "gen_bio",
    "gen_lig",
    "gen_gas",
    "gen_coal",
    "gen_oil",
    "gen_hyd_pump",
    "gen_hyd_river",
    "gen_hyd_res",
    "gen_nuc",
    "gen_other",
    "gen_oth_renew",
    "gen_solar",
    "gen_waste",
    "gen_wind",
    "load_actual",
    "price_dayahead",
    "price",
]
dict_cols = dict(zip(colnames_old, colnames_new))
df1.rename(columns=dict_cols, inplace=True)
print(df1.info())
df1.describe()


####################
# WEATHER
####################

# datetime
dfw1["time"] = pd.to_datetime(dfw1["dt_iso"], utc=True, infer_datetime_format=True)
dfw1.set_index("time", inplace=True)


# any non-numeric types?
print("non-numeric columns:", list(dfw1.dtypes[dfw1.dtypes == "object"].index))


# any missing values?
def gaps(df):
    if df.isnull().values.any():
        print("MISSING values:\n")
        mno.matrix(df)
    else:
        print("no missing values\n")


print(dfw1.describe())

# drop unnecessary columns
dfw1.drop(
    ["rain_3h", "weather_id", "weather_main", "weather_description", "weather_icon"],
    inplace=True,
    axis=1,
    errors="ignore",
)

# temperature: kelvin to celsius
temp_cols = [col for col in dfw1.columns if "temp" in col]
dfw1[temp_cols] = dfw1[temp_cols].filter(like="temp").applymap(lambda t: t - 273.15)


# convert int and float64 columns to float32
intcols = list(dfw1.dtypes[dfw1.dtypes == np.int64].index)
dfw1[intcols] = dfw1[intcols].applymap(np.float32)

f64cols = list(dfw1.dtypes[dfw1.dtypes == np.float64].index)
dfw1[f64cols] = dfw1[f64cols].applymap(np.float32)

f32cols = list(dfw1.dtypes[dfw1.dtypes == np.float32].index)
print(dfw1.info())

# investigate the outliers in the pressure column
print(dfw1["pressure"].nlargest(10))

# investigate the outliers in the wind_speed column
print(dfw1["wind_speed"].nlargest(10))


# start and end of energy and weather time series
print("earliest weather time period:", dfw1.index.min())
print("latest weather time period:", dfw1.index.max())

print("earliest energy time period:", df1.index.min())
print("latest energy time period:", df1.index.max())


# drop duplicate time periods
print(
    "count of duplicates before treatment:",
    dfw1.duplicated(subset=["dt_iso", "city_name"], keep="first").sum(),
)

dfw1 = dfw1.drop_duplicates(subset=["dt_iso", "city_name"], keep="first")
dfw1.reset_index()
print(
    "count of duplicates after treatment:",
    dfw1.duplicated(subset=["dt_iso", "city_name"], keep="first").sum(),
)


print("ENDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")


# set datetime index
dfw1["time"] = pd.to_datetime(dfw1["dt_iso"], utc=True, infer_datetime_format=True)
dfw1.set_index("time", inplace=True)
dfw1.drop("dt_iso", inplace=True, axis=1)

dfw1["city_name"] = dfw1["city_name"].apply(lambda x: x.strip())
print("size of energy dataframe:", df1.shape[0])
dfw1_city = dfw1.groupby("city_name").count()
print(dfw1_city)


unique_names = dfw1["city_name"].unique()
dict_city_weather = {city: pd.DataFrame() for city in unique_names}
for key in dict_city_weather.keys():
    dict_city_weather[key] = dfw1[:][dfw1["city_name"] == key]

# separate the cities: a weather dataframe for each of them

# dict_city_weather = {}
# for city, df_city in dfw1_city:
#     dict_city_weather[city] = df_city


# dict_city_weather = {city: df_city for city, df_city in dfw1_city}
print(dict_city_weather.keys())


# example: Bilbao weather dataframe
dfw_Bilbao = dict_city_weather.get("Barcelona")
print("Bilbao weather:")
print(dfw_Bilbao.describe())


# merge the energy and weather dataframes
df2 = df1.copy()
for city, df in dict_city_weather.items():
    city_name = str(city) + "_"
    print(city_name)
    df = df.add_suffix("_{}".format(city))
    print(df.columns)
    df2 = pd.concat([df2, df], axis=1)
    df2.drop("city_name_" + city, inplace=True, axis=1)
print(df2.info())
print(df2.iloc[[0, -1]])


# any null values?
print("any missing values?", df2.isnull().values.any())

# any ducplicate time periods?
print("count of duplicates:", df2.duplicated(keep="first").sum())


# limit the dataframe's date range
df2 = df2[df2.index >= "2018-01-01 00:00:00+00:00"]
df2.iloc[[0, -1]]


# check correlations of features with price
df_corr = df2.corr(method="pearson")
print(df_corr.shape)
print("correlation with price:")
df_corrP = pd.DataFrame(df_corr["price"].sort_values(ascending=False))
print(df_corrP)

# highest absolute correlations with price
pd.options.display.float_format = "{:,.2f}".format
df_corrH = df_corrP[np.abs(df_corrP["price"]) > 0.25]
print(df_corrH)


# limit energy dataframe to columns that have
# at least a moderate correlation with price
df3 = df2[df_corrH.index]
print(df3.info())


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


# dataframe with price and features only
df4 = df3.copy()
df4.drop(["weekday", "month", "wday", "hour"], inplace=True, axis=1)


# create time series object for target variable
ts_P = TimeSeries.from_series(df4["price"])

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
df_covF = df4.loc[:, df4.columns != "price"]
ts_covF = TimeSeries.from_dataframe(df_covF)

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
print("univariate:", ts_covF.is_univariate)


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
    attribute="hour",
    until=pd.Timestamp("2019-01-04 22:00:00+00:00"),
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


# train/test split
covT_train, covT_test = covT.split_after(SPLIT)


# rescale the covariates: fitting on the training set
scalerT = Scaler()
scalerT.fit(covT_train)
covT_ttrain = scalerT.transform(covT_train)
covT_ttest = scalerT.transform(covT_test)
covT_t = scalerT.transform(covT)

covT_t = covT_t.astype(np.float32)


pd.options.display.float_format = "{:.0f}".format
print("first and last row of unscaled time covariates:")
print(covT.pd_dataframe().iloc[[0, -1]])


model = TransformerModel(
    input_chunk_length=INLEN,
    output_chunk_length=N_FC,
    batch_size=BATCH,
    n_epochs=EPOCHS,
    model_name="Transformer_price",
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


# training: load a saved model or (re)train
if LOAD:
    print("have loaded a previously saved model from disk:" + mpath)
    model = TransformerModel.load_model(mpath)  # load previously model from disk
else:
    model.fit(ts_ttrain, past_covariates=covT_t, verbose=True)
    print("have saved the model after training:", mpath)
    model.save_model(mpath)


# testing: generate predictions
ts_tpred = model.predict(
    n=len(ts_ttest), num_samples=N_SAMPLES, n_jobs=N_JOBS, verbose=True
)


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
