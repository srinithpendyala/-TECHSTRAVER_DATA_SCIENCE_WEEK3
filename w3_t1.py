import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Handle Missing Values
print(df.isnull().sum())
df.dropna(inplace=True)

# Remove Duplicates
df.drop_duplicates(inplace=True)

# Convert Data Types
df['date_column'] = pd.to_datetime(df['date_column'])

# Extract Useful Information
df['month'] = df['date_column'].dt.month
df['city'] = df['address_column'].apply(lambda x: x.split(',')[1].strip())

# Descriptive Statistics
print(df.describe())

# Visualizations
plt.figure(figsize=(10,6))
sns.histplot(df['sales_column'])
plt.title('Sales Distribution')
plt.show()

monthly_sales = df.groupby('month')['sales_column'].sum()
plt.figure(figsize=(10,6))
sns.barplot(x=monthly_sales.index, y=monthly_sales.values)
plt.title('Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.show()

city_sales = df.groupby('city')['sales_column'].sum()
plt.figure(figsize=(10,6))
sns.barplot(x=city_sales.index, y=city_sales.values)
plt.title('Sales by City')
plt.xlabel('City')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()

# Best Month for Sales
best_month = monthly_sales.idxmax()
best_month_sales = monthly_sales.max()
print(f"The best month for sales is {best_month} with total sales of {best_month_sales}.")

# City with the Most Product Sales
best_city = city_sales.idxmax()
best_city_sales = city_sales.max()
print(f"The city with the most product sales is {best_city} with total sales of {best_city_sales}.")

from itertools import combinations
from collections import Counter

df['Grouped'] = df.groupby('order_id')['product'].transform(lambda x: ','.join(x))
df2 = df[['order_id', 'Grouped']].drop_duplicates()

count = Counter()
for row in df2['Grouped']:
    row_list = row.split(',')
    count.update(Counter(combinations(row_list, 2)))

print("Products frequently sold together:")
for key, value in count.most_common(10):
    print(key, value)

top_selling_product = df.groupby('product')['sales_column'].sum().idxmax()
print(f"The top-selling product is {top_selling_product}.")

least_selling_by_category = df.groupby(['category', 'product'])['sales_column'].sum().idxmin()
least_selling_by_brand = df.groupby(['brand', 'product'])['sales_column'].sum().idxmin()
print(f"The least selling product by category is {least_selling_by_category}.")
print(f"The least selling product by brand is {least_selling_by_brand}.")

plt.figure(figsize=(10,6))
sns.histplot(df['rating'])
plt.title('Ratings Distribution')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

best_rated_brands = df.groupby('brand')['rating'].mean().sort_values(ascending=False)
print("Best rated brands:")
print(best_rated_brands.head())
