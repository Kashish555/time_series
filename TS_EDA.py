import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("/content/TS_OTTDataset_Cleaned.csv")

print("Columns:", data.columns)

data.ffill(inplace=True)

date_columns = ['Membership Start Date', 'Membership End Date']
for col in date_columns:
    data[col] = pd.to_datetime(data[col], format='%d-%m-%Y', errors='coerce')

data['Membership Duration (Days)'] = (data['Membership End Date'] - data['Membership Start Date']).dt.days

data.dropna(subset=['Membership Duration (Days)'], inplace=True)

# Normalize Membership
scaler = MinMaxScaler()
data[['Membership Duration (Days)']] = scaler.fit_transform(data[['Membership Duration (Days)']])

data['Gender'] = data['Gender'].map({'Female': 1, 'Male': 0})

print(data.head())

data.to_csv("Processed_OTT_Dataset.csv", index=False)

# Plot distribution of age
plt.figure(figsize=(10, 5))
sns.histplot(data['Age'], bins=30, kde=True)
plt.title('Age Distribution of Netflix Users')
plt.xlabel('Age')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# Plot gender distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Gender', data=data, palette=['blue', 'pink'])
plt.title('Gender Distribution of Netflix Users')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Male', 'Female'])
plt.grid(True)
plt.show()

# Subscription Plan Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='Subscription Plan', data=data, order=data['Subscription Plan'].value_counts().index)
plt.title('Subscription Plan Distribution')
plt.xlabel('Subscription Plan')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# Renewal Status Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Renewal Status', data=data, order=data['Renewal Status'].value_counts().index)
plt.title('Renewal Status Distribution')
plt.xlabel('Renewal Status')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# Engagement Metrics
engagement_order = ['Low', 'Medium', 'High']
data['Engagement Level'] = pd.Categorical(data['Engagement Level'], categories=engagement_order, ordered=True)

plt.figure(figsize=(8, 5))
engagement_colors = {'Low': 'red', 'Medium': 'blue', 'High': 'green'}
sns.countplot(x='Engagement Level', data=data,palette=engagement_colors, order=engagement_order)
plt.title('Distribution of User Engagement Levels')
plt.xlabel('Engagement Level')
plt.ylabel('Count')
plt.grid(True)
plt.show()

import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

data = pd.read_csv("Processed_OTT_Dataset.csv")

data['Membership Start Date'] = pd.to_datetime(data['Membership Start Date'])
data['Membership End Date'] = pd.to_datetime(data['Membership End Date'])

data['Renewal Month'] = data['Membership End Date'].dt.to_period('M')

renewal_data = data.groupby('Renewal Month')['Subscription Renewed'].apply(lambda x: (x == 'Renewed').sum()).reset_index()
renewal_data['Renewal Month'] = renewal_data['Renewal Month'].astype(str)
renewal_data.set_index('Renewal Month', inplace=True)

renewal_data.index = pd.to_datetime(renewal_data.index)

# RENEWAL TREND OVER TIME
plt.figure(figsize=(12, 6))
plt.plot(renewal_data, marker='o', linestyle='-')
plt.title("Subscription Renewal Trend Over Time")
plt.xlabel("Time (Months)")
plt.ylabel("Number of Renewals")
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

# TREND, SEASONALITY, AND RESIDUALS
decomposition = seasonal_decompose(renewal_data, model='additive', period=12)

plt.figure(figsize=(12, 8))

plt.subplot(411)
plt.plot(decomposition.observed, label='Observed')
plt.legend(loc='upper left')

plt.subplot(412)
plt.plot(decomposition.trend, label='Trend', color='green')
plt.legend(loc='upper left')

plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonality', color='orange')
plt.legend(loc='upper left')

plt.subplot(414)
plt.plot(decomposition.resid, label='Residuals', color='red')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# CHECK FOR STATIONARITY
result = adfuller(renewal_data['Subscription Renewed'])
print("\nAugmented Dickey-Fuller Test Results:")
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

if result[1] < 0.05:
    print("✅ The time series is stationary (No trend).")
else:
    print("❌ The time series is NOT stationary (Has trend or seasonality). Consider differencing.")