# Step 1: Import Libraries and Load Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your stock market dataset
df = pd.read_csv('stockmarketdetails.csv')

# Step 2: Clean and Preprocess Data

# Handle Missing Values
print(df.isnull().sum())
df.dropna(inplace=True)

# Remove Duplicates
df.drop_duplicates(inplace=True)

# Convert Data Types
df['date'] = pd.to_datetime(df['date'])

# Step 3: Conduct Exploratory Data Analysis (EDA)

# Visualize Stock Prices Over Time
plt.figure(figsize=(14,7))
sns.lineplot(data=df, x='date', y='close')
plt.title('Stock Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.show()

# Calculate Moving Averages
df['20_day_MA'] = df['close'].rolling(window=20).mean()
df['50_day_MA'] = df['close'].rolling(window=50).mean()

plt.figure(figsize=(14,7))
sns.lineplot(data=df, x='date', y='close', label='Close Price')
sns.lineplot(data=df, x='date', y='20_day_MA', label='20 Day MA')
sns.lineplot(data=df, x='date', y='50_day_MA', label='50 Day MA')
plt.title('Stock Prices with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Calculate Daily Returns
df['daily_return'] = df['close'].pct_change()

plt.figure(figsize=(14,7))
sns.histplot(df['daily_return'].dropna(), bins=100, kde=True)
plt.title('Distribution of Daily Returns')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.show()

# Step 4: Draw Conclusions and Insights

# Trends in Stock Prices
print(f"The moving averages indicate the trends in the stock prices. Observe the plots to analyze trends over different time windows.")

# Stock Volatility
volatility = df['daily_return'].std()
print(f"The stock's volatility, measured by the standard deviation of daily returns, is {volatility}.")

# Correlation Between Different Stocks
# Assuming df2 is another DataFrame containing stock data for a different stock
# df2 = pd.read_csv('another_stock_dataset.csv')
# df2['date'] = pd.to_datetime(df2['date'])
# df_merged = pd.merge(df, df2, on='date', suffixes=('_stock1', '_stock2'))
# correlation = df_merged['close_stock1'].corr(df_merged['close_stock2'])
# print(f"The correlation between the two stocks is {correlation}.")

# Stock Price Forecasting
# Using ARIMA for time series forecasting
from statsmodels.tsa.arima.model import ARIMA

# Fit model
model = ARIMA(df['close'], order=(5,1,0))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=30)
plt.figure(figsize=(14,7))
plt.plot(df['date'], df['close'], label='Historical')
plt.plot(pd.date_range(df['date'].iloc[-1], periods=30, freq='D'), forecast, label='Forecast')
plt.title('Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
