from scripts.util.regression_utils import *



data_supervised = pd.read_csv("datasets/ML_DATASET_Hackathon_Supervised.csv").dropna().drop_duplicates().drop(
    'Unnamed: 0', axis=1)
data_supervised[['Ticket_Opened_Date', 'Ticket_Closed']] = data_supervised[
    ['Ticket_Opened_Date', 'Ticket_Closed']].apply(pd.to_datetime)

data_supervised = get_datetime_features(data_supervised)
data_supervised = data_supervised.drop_duplicates(subset=['Problem_Abstract', 'open_calender_week', 'Team'])

holiday_df = \
generate_calendar_based_features("Czechia", pd.DatetimeIndex(data_supervised['Ticket_Opened_Date'])).reset_index()[
    ['Ticket_Opened_Date', 'is_holiday']]

data_supervised = data_supervised.merge(holiday_df)

data_train, data_test = train_test_split(data_supervised)
data_train = data_train[~data_train.groupby('Team')['hr_duration'].apply(is_outlier)]

features = get_relevant_features(data_train)

model, y_pred = modelling_pipeline(data_train, features)

