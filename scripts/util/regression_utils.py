import pandas as pd
import numpy as np
import datetime
import holidays
from catboost import CatBoostRegressor
from catboost import Pool
from sklearn.model_selection import train_test_split
from explainerdashboard import ExplainerDashboard



def get_datetime_features(df):
    """
    Add datetime features to the dataset
    :param df:
    :return: pd.DataFrame
    """

    data_transformed = df.copy()

    data_transformed[['Ticket_Opened_Date', 'Ticket_Closed']] = data_transformed[['Ticket_Opened_Date', 'Ticket_Closed']].apply(pd.to_datetime)
    data_transformed['open_day_of_week'] = data_transformed['Ticket_Opened_Date'].dt.dayofweek.astype("int")
    data_transformed['close_day_of_week'] = data_transformed['Ticket_Closed'].dt.dayofweek.astype("int")
    data_transformed['open_calender_week'] = data_transformed['Ticket_Opened_Date'].dt.week
    data_transformed['close_calender_week'] = data_transformed['Ticket_Closed'].dt.week
    data_transformed['open_month'] = data_transformed['Ticket_Opened_Date'].dt.month.astype("int")
    data_transformed['open_year'] = data_transformed['Ticket_Opened_Date'].dt.year.astype("int")
    data_transformed['is_friday'] = np.where(data_transformed['open_day_of_week'] == 4, 1, 0)
    data_transformed['duration'] = data_transformed['Ticket_Closed'] - data_transformed['Ticket_Opened_Date']
    data_transformed['hr_duration'] = data_transformed['duration']/np.timedelta64(1, 'h')
    data_transformed['opened_hour'] = data_transformed['Ticket_Opened_Date'].dt.hour
    data_transformed['open_hour_sin'] = np.sin(2 * np.pi * data_transformed['opened_hour']/24.0)
    data_transformed['open_hour_cos'] = np.cos(2 * np.pi * data_transformed['opened_hour']/24.0)

    return data_transformed

def is_outlier(s):
    """
    Identify outliers using z score
    """
    lower_limit = s.mean() - (s.std() * 5)
    upper_limit = s.mean() + (s.std() * 3)

    return ~s.between(lower_limit, upper_limit)

def add_dt_vars(df):
    """
    Convert start + end variables to datetime
    """
    df_added = df.copy()
    df_added[['Ticket_Opened_Date', 'Ticket_Closed']] = df_added[['Ticket_Opened_Date', 'Ticket_Closed']].apply(pd.to_datetime)

    return df_added

def generate_calendar_based_features(country_name, datetime_index):
    """
    Generates calendar-based exogenous features based on a datetime index (month of the year, if the day is a holiday)
    :param country_name: str
    :param datetime_index: pd.DatetimeIndex
    :return: pd.DataFrame
    """
    X_dataframe = pd.DataFrame(index=datetime_index)
    years_list = datetime_index.year.unique()
    country_specific_holidays = get_country_holidays(country_name, years_list)
    X_dataframe['month'] = X_dataframe.index.month
    X_dataframe['dayoftheweek'] = X_dataframe.index.dayofweek
    X_dataframe['year'] = X_dataframe.index.year
    X_dataframe['is_holiday'] = [1 if x.date() in country_specific_holidays else 0 for x in X_dataframe.index]

    return X_dataframe

def get_country_holidays(country_name, years_list):
    """
    Based on a country name, it generates the holidays for that country.
    :param country_name:  str
    :param years_list: list of int
    :return: list of str
    """
    holidays_list = holidays.country_holidays(country=country_name, years=years_list)
    dates_list = list(holidays_list.keys())

    additional_holidays_list = []
    for each_year in years_list:
        additional_holidays_list.append(datetime.date(each_year, 12, 31))
        additional_holidays_list.append(datetime.date(each_year, 12, 24))

    return dates_list+additional_holidays_list


def get_relevant_features(df, r="open|is|Team"):
    """
    Get relevant features of the data set
    """
    feature_index = df.filter(regex=r).columns

    return feature_index

def modelling_pipeline(df, features):
    """
    Wrapper for the predictive workflow
    """

    X_train, y_train = df.drop(['hr_duration'], axis=1)[features], (df['hr_duration'])
    model = CatBoostRegressor(iterations=1000, learning_rate=0.3, depth=4, one_hot_max_size=1000)
    pool_train = Pool(X_train, y_train, cat_features = ['Team'])
    X_test = data_test.drop(['hr_duration'], axis=1)[features]
    pool_test = Pool(X_test, cat_features = ['Team'])
    fit = model.fit(pool_train)

    y_pred = model.predict(pool_test)
    y_pred = (np.abs(y_pred)+y_pred)/2

    return model, y_pred