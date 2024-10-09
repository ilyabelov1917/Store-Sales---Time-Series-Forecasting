import pandas as pd
import seaborn as sns
import matplotlib as plt

#Loading data
import seaborn as sns
import matplotlib.pyplot as plt

#Loading data
train_df = pd.read_csv('/Users/ilyabelov/Desktop/Kaggle/store-sales-time-series-forecasting/train.csv')
test_df = pd.read_csv('/Users/ilyabelov/Desktop/Kaggle/store-sales-time-series-forecasting/test.csv')
holidays_df = pd.read_csv('/Users/ilyabelov/Desktop/Kaggle/store-sales-time-series-forecasting/holidays_events.csv')
stores_df = pd.read_csv('/Users/ilyabelov/Desktop/Kaggle/store-sales-time-series-forecasting/stores.csv')
transactions_df = pd.read_csv('/Users/ilyabelov/Desktop/Kaggle/store-sales-time-series-forecasting/transactions.csv')
oil_df = pd.read_csv('/Users/ilyabelov/Desktop/Kaggle/store-sales-time-series-forecasting/oil.csv')
#Statistics info
print("Train info: ")
print(train_df.info())
print(train_df.describe())


print("Test info: ")
print(test_df.info())
print(test_df.describe())

print("Holidays info: ")
print(holidays_df.info())
print(holidays_df.describe())

print("Stores info: ")
print(stores_df.info())
print(stores_df.describe())

print("Transactions info: ")
print(transactions_df.info())
print(transactions_df.describe())

print("Oil info: ")
print(oil_df.info())
print(oil_df.describe())

#Missing values
print(f"Train null {train_df.isnull().sum()}")
print(f"Oil null {oil_df.isnull().sum()}")

#Using interpolation method to fill out missing data
oil_df['dcoilwtico_interp'] = oil_df['dcoilwtico'].interpolate(method='linear')

#Convert date (object) to datetime
train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])
oil_df['date'] = pd.to_datetime(oil_df['date'])
holidays_df['date'] = pd.to_datetime(holidays_df['date'])
#To find out the total amount of sales by date
daily_sales = train_df.groupby('date').sum().reset_index()
#Visualisation of the sales trend using Seaborn
g = sns.relplot(data=daily_sales, x="date", y="sales", kind="line")
g.fig.suptitle('Sales Trend')

#Merge external data like Oil, Stores, Holidays

#Oil merge
train_df = pd.merge(train_df, oil_df, how = 'left', on = 'date')
test_df = pd.merge(test_df, oil_df, how = 'left', on = 'date')
#Stores merge
train_df = pd.merge(train_df, stores_df, how = 'left', on = 'store_nbr')
test_df = pd.merge(test_df, stores_df, how = 'left', on = 'store_nbr')
#Merging with holidays_df
train_df = pd.merge(train_df, holidays_df[['date', 'type']], how='left', on='date')
test_df = pd.merge(test_df, holidays_df[['date', 'type']], how='left', on='date')

#Creating a 'is_holiday' column to identify holidays
train_df['is_holiday'] = train_df['type'].notnull().astype(int)
test_df['is_holiday'] = test_df['type'].notnull().astype(int)

#Dropping the 'type' column after creating 'is_holiday', using errors='ignore' to avoid KeyError if 'type' doesn't exist
train_df = train_df.drop(columns=['type'], errors='ignore')
test_df = test_df.drop(columns=['type'], errors='ignore')
#Merging with holidays_df
train_df = pd.merge(train_df, holidays_df[['date', 'type']], how='left', on='date')
test_df = pd.merge(test_df, holidays_df[['date', 'type']], how='left', on='date')

#Creating a 'is_holiday' column to identify holidays
train_df['is_holiday'] = train_df['type'].notnull().astype(int)
test_df['is_holiday'] = test_df['type'].notnull().astype(int)

#Dropping the 'type' column after creating 'is_holiday', using errors='ignore' to avoid KeyError if 'type' doesn't exist
train_df = train_df.drop(columns=['type'], errors='ignore')
test_df = test_df.drop(columns=['type'], errors='ignore')

#Feature Engineering

#Feature Engineering on Date
train_df['day'] = train_df['date'].dt.dayofweek
train_df['month'] = train_df['date'].dt.month
test_df['day'] = test_df['date'].dt.dayofweek
test_df['month'] = test_df['date'].dt.month
#Creating Lag Features for Sales
for lag in [1, 7, 30]:
    train_df[f'sales_lag_{lag}'] = train_df.groupby(['store_nbr', 'family'])['sales'].shift(lag)

#Creating Rolling Window Features for Sales
for window in [7, 14, 30]:
    train_df[f'sales_rolling_mean_{window}'] = train_df.groupby(['store_nbr', 'family'])['sales'].transform(lambda x: x.rolling(window, min_periods=1).mean())
# Weekday or Weekend
train_df['is_weekend'] = train_df['day'].isin([5, 6]).astype(int)
test_df['is_weekend'] = test_df['day'].isin([5, 6]).astype(int)
