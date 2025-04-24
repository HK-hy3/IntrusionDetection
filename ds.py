import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Load Dataset
df = pd.read_csv("road_accident_dataset.csv")  # Update with actual file name

# Convert Date column to datetime format
df['Accident Date'] = pd.to_datetime(df['Accident Date'])
df['Year'] = df['Accident Date'].dt.year
df['Month'] = df['Accident Date'].dt.month
df['DayOfWeek'] = df['Day_of_Week']
df['Hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour  # Handling possible missing times

# --- 1. Frequency of Accidents Over Time ---
print("Total number of accidents:", len(df))
plt.figure(figsize=(12, 6))
sns.countplot(x='Year', data=df, palette='coolwarm')
plt.title("Accident Frequency by Year")
plt.xticks(rotation=45)
plt.show()

# Distribution of accidents by month
plt.figure(figsize=(12, 6))
sns.countplot(x='Month', data=df, palette='coolwarm')
plt.title("Accident Frequency by Month")
plt.show()

# Distribution of accidents by day of the week
plt.figure(figsize=(12, 6))
sns.countplot(x='DayOfWeek', data=df, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title("Accident Frequency by Day of the Week")
plt.show()

# Distribution of accidents by hour of the day
plt.figure(figsize=(12, 6))
sns.histplot(df['Hour'], bins=24, kde=True)
plt.title("Accident Frequency by Hour")
plt.xlabel("Hour of the Day")
plt.show()

# --- 2. Geographical Distribution ---
print("Top 10 accident locations:")
print(df['Local_Authority_(District)'].value_counts().head(10))
plt.figure(figsize=(12, 6))
sns.barplot(x=df['Local_Authority_(District)'].value_counts().index[:10], y=df['Local_Authority_(District)'].value_counts().values[:10], palette='coolwarm')
plt.xticks(rotation=90)
plt.title("Top 10 Locations with Highest Accident Frequency")
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(x=df['Longitude'], y=df['Latitude'], alpha=0.5)
plt.title("Geographical Distribution of Accidents")
plt.show()

# --- 3. Accident Severity Analysis ---
severity_counts = df['Accident_Severity'].value_counts(normalize=True) * 100
print("Accident Severity Distribution:")
print(severity_counts)
sns.countplot(x='Accident_Severity', data=df)
plt.title("Accident Severity Distribution")
plt.show()

# --- 7. Temporal Patterns ---
# Peak accident times during the day
plt.figure(figsize=(12, 6))
sns.histplot(df['Hour'], bins=24, kde=True)
plt.title("Accident Frequency by Hour")
plt.xlabel("Hour of the Day")
plt.show()

# Weekday vs Weekend comparison
if 'DayOfWeek' in df.columns:
    df['Weekend'] = df['DayOfWeek'].apply(lambda x: 'Weekend' if x in ['Saturday', 'Sunday'] else 'Weekday')
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Weekend', data=df, palette='coolwarm')
    plt.title("Accident Frequency: Weekdays vs Weekends")
    plt.show()

# --- 8. Contributing Factors ---
print("Top contributing factors to accidents:")
if 'Junction_Control' in df.columns:
    print("\nJunction Control:")
    print(df['Junction_Control'].value_counts().head(10))
if 'Carriageway_Hazards' in df.columns:
    print("\nCarriageway Hazards:")
    print(df['Carriageway_Hazards'].dropna().value_counts().head(10))
if 'Road_Surface_Conditions' in df.columns:
    print("\nRoad Surface Conditions:")
    print(df['Road_Surface_Conditions'].value_counts().head(10))
if 'Weather_Conditions' in df.columns:
    print("\nWeather Conditions:")
    print(df['Weather_Conditions'].value_counts().head(10))

# --- 9. Injury and Fatality Analysis ---
if 'Number_of_Casualties' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Number_of_Casualties'], bins=10, kde=True)
    plt.title("Distribution of Number of Casualties in Accidents")
    plt.show()

# --- 10. Comparative Analysis ---
sns.countplot(x='Urban_or_Rural_Area', data=df)
plt.title("Urban vs. Rural Accident Distribution")
plt.show()

print("Analysis Complete.")
