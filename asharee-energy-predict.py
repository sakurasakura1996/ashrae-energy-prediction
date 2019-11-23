import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import datetime
import gc

# 加载数据
train_data = pd.read_csv('work/train.csv')
print(train_data.shape)

# train_data.info()
# train_data.head()
# train_data.describe()

# 数据量太大，我们要想着合理划分使用，首先，看看train_data 中的列，电表代码，测试时间，电表值，我们可以把四种电表值分开来算
train_data_meter0 = train_data[train_data.meter == 0]
train_data_meter1 = train_data[train_data.meter == 1]
train_data_meter2 = train_data[train_data.meter == 2]
train_data_meter3 = train_data[train_data.meter == 3]
print(train_data_meter0.shape)  # 电表id为1 的数据明显多于其他几个
print(train_data_meter1.shape)
print(train_data_meter2.shape)
print(train_data_meter3.shape)

# remove outliers 去除离群值
train_df = train_data[train_data['building_id'] != 1099]
train_df = train_df.query('not (building_id <= 104 & meter ==0 & timestamp <="2016-05-20")')

building_df = pd.read_csv('work/building_metadata.csv')
weather_df = pd.read_csv('work/weather_train.csv')


# In[20]:


# Utility Functions
def fill_weather_dataset(weather_df):
    # Find Missing Dates
    time_format = "%Y-%m-%d %H:%M:%S"
    start_date = datetime.datetime.strptime(weather_df['timestamp'].min(), time_format)
    end_date = datetime.datetime.strptime(weather_df['timestamp'].max(), time_format)
    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)
    hours_list = [(end_date - datetime.timedelta(hours=x)).strftime(time_format) for x in range(total_hours)]

    missing_hours = []
    for site_id in range(16):
        site_hours = np.array(weather_df[weather_df['site_id'] == site_id]['timestamp'])
        new_rows = pd.DataFrame(np.setdiff1d(hours_list, site_hours), columns=['timestamp'])
        new_rows['site_id'] = site_id
        weather_df = pd.concat([weather_df, new_rows])

        weather_df = weather_df.reset_index(drop=True)

        # Add new Features
    weather_df["datetime"] = pd.to_datetime(weather_df["timestamp"])
    weather_df["day"] = weather_df["datetime"].dt.day
    weather_df["week"] = weather_df["datetime"].dt.week
    weather_df["month"] = weather_df["datetime"].dt.month

    # Reset Index for Fast Update
    weather_df = weather_df.set_index(['site_id', 'day', 'month'])

    air_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id', 'day', 'month'])['air_temperature'].mean(),
                                          columns=["air_temperature"])
    weather_df.update(air_temperature_filler, overwrite=False)

    # Step 1
    cloud_coverage_filler = weather_df.groupby(['site_id', 'day', 'month'])['cloud_coverage'].mean()
    # Step 2
    cloud_coverage_filler = pd.DataFrame(cloud_coverage_filler.fillna(method='ffill'), columns=["cloud_coverage"])

    weather_df.update(cloud_coverage_filler, overwrite=False)

    due_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id', 'day', 'month'])['dew_temperature'].mean(),
                                          columns=["dew_temperature"])
    weather_df.update(due_temperature_filler, overwrite=False)

    # Step 1
    sea_level_filler = weather_df.groupby(['site_id', 'day', 'month'])['sea_level_pressure'].mean()
    # Step 2
    sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method='ffill'), columns=['sea_level_pressure'])

    weather_df.update(sea_level_filler, overwrite=False)

    wind_direction_filler = pd.DataFrame(weather_df.groupby(['site_id', 'day', 'month'])['wind_direction'].mean(),
                                         columns=['wind_direction'])
    weather_df.update(wind_direction_filler, overwrite=False)

    wind_speed_filler = pd.DataFrame(weather_df.groupby(['site_id', 'day', 'month'])['wind_speed'].mean(),
                                     columns=['wind_speed'])
    weather_df.update(wind_speed_filler, overwrite=False)

    # Step 1
    precip_depth_filler = weather_df.groupby(['site_id', 'day', 'month'])['precip_depth_1_hr'].mean()
    # Step 2
    precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method='ffill'), columns=['precip_depth_1_hr'])

    weather_df.update(precip_depth_filler, overwrite=False)

    weather_df = weather_df.reset_index()
    weather_df = weather_df.drop(['datetime', 'day', 'week', 'month'], axis=1)

    return weather_df


# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin

from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype


def reduce_mem_usage(df, use_float16=False):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        
    """

    start_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def features_engineering(df):
    # Sort by timestamp
    df.sort_values("timestamp")
    df.reset_index(drop=True)

    # Add more features
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
    df["hour"] = df["timestamp"].dt.hour
    df["weekend"] = df["timestamp"].dt.weekday
    holidays = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",
                "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",
                "2017-01-02", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",
                "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",
                "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",
                "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",
                "2019-01-01"]
    df["is_holiday"] = (df.timestamp.isin(holidays)).astype(int)
    df['square_feet'] = np.log1p(df['square_feet'])

    # Remove Unused Columns
    drop = ["timestamp", "sea_level_pressure", "wind_direction", "wind_speed", "year_built", "floor_count"]
    df = df.drop(drop, axis=1)
    gc.collect()

    # Encode Categorical Data
    le = LabelEncoder()
    df["primary_use"] = le.fit_transform(df["primary_use"])

    return df


# In[23]:


# Fill Weather Information
# using this kernel to handle missing weather information
weather_df = fill_weather_dataset(weather_df)

# In[26]:


# Memory reduction
train_df = reduce_mem_usage(train_df, use_float16=True)
building_df = reduce_mem_usage(building_df, use_float16=True)
weather_df = reduce_mem_usage(weather_df, use_float16=True)

# In[27]:


# Merge Data   We need to add building and weather information into training dataset.
train_df = train_df.merge(buildinf_df, left_on='building_id', right_on='building_id', how='left')
train_df = train_df.merge(weather_df, how='left', left_on=['site_id', 'timestamp'], right_on=['site_id', 'timestamp'])
del weather_df
gc.collect()

# In[28]:


# Features Engineering
train_df = features_engineering(train_df)

# In[29]:


train_df.head(20)

# In[30]:


# Features & Target Varibales
target = np.log1p(train_df["meter_reading"])
features = train_df.drop('meter_reading', axis=1)
del train_df
gc.collect()

# In[31]:


# KFold LightGBM Model
categorical_features = ["building_id", "site_id", "meter", "primary_use", "is_holiday", "weekend"]
params = {
    "objective": "regression",
    "boosting": "gbdt",
    "num_leaves": 1280,
    "learning_rate": 0.05,
    "feature_fraction": 0.85,
    "reg_lambda": 2,
    "metric": "rmse",
}

kf = KFold(n_splits=3)
models = []
for train_index, test_index in kf.split(features):
    train_features = features.loc[train_index]
    train_target = target.loc[train_index]

    test_features = features.loc[test_index]
    test_target = target.loc[test_index]

    d_training = lgb.Dataset(train_features, label=train_target, categorical_feature=categorical_features,
                             free_raw_data=False)
    d_test = lgb.Dataset(test_features, label=test_target, categorical_feature=categorical_features,
                         free_raw_data=False)

    model = lgb.train(params, train_set=d_training, num_boost_round=1000, valid_sets=[d_training, d_test],
                      verbose_eval=25, early_stopping_rounds=50)
    models.append(model)
    del train_features, train_target, test_features, test_target, d_training, d_test
    gc.collect()

# In[32]:


del features, target
gc.collect()

# In[34]:


# Load test data
test_df = pd.read_csv('work/test.csv')
row_ids = test_df["row_id"]
test_df.drop("row_id", axis=1, inplace=True)
test_df = reduce_mem_usage(test_df)

# In[35]:


# Merge Building Data
test_df = test_df.merge(building_df, left_on='building_id', right_on='building_id', how='left')
del building_df
gc.collect()

# In[36]:


# Fill Weather Information
weather_df = pd.read_csv('work/weather_test.csv')
weather_df = fill_weather_dataset(weather_df)
weather_df = reduce_mem_usage(weather_df)

# In[37]:


# Merge Weather Data
test_df = test_df.merge(weather_df, how='left', on=['timestamp', 'site_id'])
del weather_df
gc.collect()

# In[38]:


# Features Engineering
test_df = features_engineering(test_df)

# In[39]:


# prediction
results = []
for model in models:
    if results == []:
        results = np.expm1(model.predict(test_df, num_iteration=model.best_iteration)) / len(models)
    else:
        results += np.expm1(model.predict(test_df, num_iteration=model.best_iteration)) / len(models)
    del model
    gc.collect()

# In[40]:


del test_df, models
gc.collect()

# In[41]:


# submission
results_df = pd.DataFrame({"row_id": row_ids, "meter_reading": np.clip(results, 0, a_max=None)})
del row_ids, results
gc.collect()
results_df.to_csv("submission.csv", index=False)

# In[42]:


results_df.head(20)
