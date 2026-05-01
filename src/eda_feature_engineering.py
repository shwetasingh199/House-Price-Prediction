import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create folders
os.makedirs("outputs/graphs", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv("data/raw/housing.csv")

print("\n📊 Dataset Loaded Successfully!\n")

# -------------------------------
# 2. BASIC INFO
# -------------------------------
print("Columns:", df.columns)

# -------------------------------
# 3. DATA CLEANING (IMPORTANT FIX)
# -------------------------------

# Rename columns
df.rename(columns={"total_sqft": "area"}, inplace=True)

# Convert area (handle ranges like "2100-2850")
def convert_sqft(x):
    try:
        if "-" in str(x):
            low, high = x.split("-")
            return (float(low) + float(high)) / 2
        return float(x)
    except:
        return None

df["area"] = df["area"].apply(convert_sqft)

# Extract bedrooms from size column
df["bedrooms"] = df["size"].str.extract(r'(\d+)')
df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors='coerce')

# Rename bath → bathrooms
df.rename(columns={"bath": "bathrooms"}, inplace=True)

# Drop unnecessary columns
df.drop(columns=["society", "availability", "size"], inplace=True, errors="ignore")

# Handle missing values
df = df.ffill()

# Drop remaining nulls
df = df.dropna()

print("\n✅ Data Cleaning Completed!\n")

# -------------------------------
# 4. EDA
# -------------------------------

print("🔍 First 5 Rows:\n", df.head())

print("\n📊 Statistical Summary:\n", df.describe())

# -------------------------------
# 5. VISUALIZATION
# -------------------------------

# Price Distribution
plt.figure()
sns.histplot(df['price'], kde=True)
plt.title("Price Distribution")
plt.savefig("outputs/graphs/price_distribution.png")
plt.close()

# Area Distribution
plt.figure()
sns.histplot(df['area'], kde=True)
plt.title("Area Distribution")
plt.savefig("outputs/graphs/area_distribution.png")
plt.close()

# Bedrooms Count
plt.figure()
sns.countplot(x=df['bedrooms'])
plt.title("Bedrooms Count")
plt.savefig("outputs/graphs/bedrooms_count.png")
plt.close()

# Area vs Price
plt.figure()
sns.scatterplot(x=df['area'], y=df['price'])
plt.title("Area vs Price")
plt.savefig("outputs/graphs/area_vs_price.png")
plt.close()

# Bedrooms vs Price
plt.figure()
sns.boxplot(x=df['bedrooms'], y=df['price'])
plt.title("Bedrooms vs Price")
plt.savefig("outputs/graphs/bedrooms_vs_price.png")
plt.close()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("outputs/graphs/correlation_heatmap.png")
plt.close()

# -------------------------------
# 6. OUTLIER REMOVAL
# -------------------------------
q1 = df['price'].quantile(0.25)
q3 = df['price'].quantile(0.75)
iqr = q3 - q1

df = df[(df['price'] > q1 - 1.5 * iqr) & (df['price'] < q3 + 1.5 * iqr)]

# -------------------------------
# 7. FEATURE ENGINEERING
# -------------------------------

print("\n⚙️ Feature Engineering...\n")

# Price per sqft
df['price_per_sqft'] = df['price'] / df['area']

# Total rooms
df['total_rooms'] = df['bedrooms'] + df['bathrooms']

# Luxury flag
df['is_luxury'] = np.where(df['price'] > df['price'].median(), 1, 0)

# -------------------------------
# 8. INSIGHTS
# -------------------------------

print("📊 Insights:\n")
print("Average Price:", df['price'].mean())
print("Average Area:", df['area'].mean())
print("Most Common Bedrooms:", df['bedrooms'].mode()[0])

# -------------------------------
# 9. SAVE CLEAN DATA
# -------------------------------
df.to_csv("data/processed/cleaned_data.csv", index=False)

print("\n✅ Processed dataset saved!")
print("📁 data/processed/cleaned_data.csv")

print("\n🎯 EDA + Feature Engineering DONE!")