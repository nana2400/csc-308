#!/usr/bin/env python
# coding: utf-8

# In[127]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


# In[128]:


# Step 2: Import and Inspect the Data
# Load the dataset
df = pd.read_csv(r'C:\Users\Administrator\Desktop\shisha\movies.csv')


# In[129]:


# Inspect the dataset
print("Shape:", df.shape)
print("\nInfo:")
print(df.info())
print("\nHead:")
print(df.head(5))
print("\nDescription:")
print(df.describe())
print("\nSummary Statistics:")
df.describe(include='all')


# In[130]:


# Step 3: Clean and Preprocess the Data
# Handle missing values
print(df.isnull().sum)
#drop missingvalues
df.dropna(inplace=True)


# In[131]:


#identify duplicates
print(df.duplicated().sum())


# Since there are no duplicates we will not be required to drop any duplicates.

# In[132]:


# Calculate the mean rating
mean_rating = df['RATING'].mean()
print(mean_rating)


# In[133]:


# Fill missing 'RATING' values with the mean
df['RATING'] = df['RATING'].fillna(mean_rating)

# Handle RunTime: fill missing values with the median
median_runtime = df['RunTime'].median()
df['RunTime'] = df['RunTime'].fillna(median_runtime)


# In[134]:


# Convert 'Gross' to numeric, handling commas and errors
df['Gross'] = df['Gross'].astype(str).str.replace(',', '').str.strip
df['Gross'] = pd.to_numeric(df['Gross'], errors='coerce')
print(df['Gross'].head(10))


# In[135]:


# Convert 'Votes' to numeric, handling commas and errors
df['VOTES'] = df['VOTES'].str.replace(',', '').astype(float)

# Fill missing 'VOTES' values with 0
df['VOTES'] = df['VOTES'].fillna(0)

print("\nMissing values after handling:")
print(df.isnull().sum())


# In[136]:


print(df.columns.tolist())


# In[137]:


# Clean 'YEAR' column
df['YEAR'] = df['YEAR'].astype(str).str.extract(r'(\d{4})', expand=False)
df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce').fillna(0).astype(int)


# In[138]:


# Encode categorical variables
le = LabelEncoder()
df['GENRE'] = le.fit_transform(df['GENRE'])

# Normalize/scale numeric features
scaler = StandardScaler()
df[['VOTES', 'RunTime', 'Gross']] = scaler.fit_transform(df[['VOTES', 'RunTime', 'Gross']])

print("\nCleaned Head:")
print(df.head())


# In[139]:


# Step 4: Conduct Basic EDA
print("\nValue Counts for GENRE:")
print(df['GENRE'].value_counts().head())

print("\nGroupby RATING by GENRE (Mean):")
print(df.groupby('GENRE')['RATING'].mean().head())

print("Correlation matrix:\n", df[['RATING', 'RunTime']].corr())


# In[140]:


# Step 5: Visualize the Data
# Chart 1: Histogram of Ratings (Matplotlib)
plt.figure(figsize=(10, 6))
plt.hist(df['RATING'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()


# The histogram shows the distribution of movie ratings. It appears to be approximately normally distributed, with most ratings between 6 and 8.

# In[141]:


# Chart 2: Scatterplot of Votes vs. Ratings (Matplotlib)
plt.figure(figsize=(10, 6))
plt.scatter(df['VOTES'], df['RATING'], alpha=0.5, color='red')
plt.title('Scatterplot of Votes vs. Ratings')
plt.xlabel('Votes (Scaled)')
plt.ylabel('Rating')
plt.show()


# The scatterplot shows the relationship between the number of votes and the rating. There seems to be a weak positive correlation, suggesting that movies with more votes tend to have slightly higher ratings.

# In[142]:


# Chart 3: Boxplot of Ratings by Genre (Seaborn)
plt.figure(figsize=(12, 6))
sns.boxplot(x='GENRE', y='RATING', data=df)
plt.title('Boxplot of Ratings by Genre')
plt.xlabel('Genre')
plt.ylabel('Rating')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# The boxplot visualizes the distribution of ratings for each genre. This helps in understanding which genres tend to have higher or lower ratings on average.

# In[143]:


df_filtered = df[df['YEAR'] > 0]
numeric_cols = ['RATING', 'VOTES', 'RunTime', 'Gross', 'YEAR']
sns.set(style="ticks", palette="pastel")
pairplot = sns.pairplot(df_filtered[numeric_cols], diag_kind='kde', plot_kws={'alpha':0.5})
pairplot.fig.suptitle('Pairplot of Movie Dataset Numeric features', y=1.02)
plt.show()


# In[144]:


# Chart 5: Line Plot of Average Rating Over Years
yearly_avg_rating = df.groupby('YEAR')['RATING'].mean()
plt.figure(figsize=(12, 6))
plt.plot(yearly_avg_rating.index, yearly_avg_rating.values, marker='o', linestyle='-')
plt.title('Average Movie Rating Over Years')
plt.xlabel('Year')
plt.ylabel('Average Rating')
plt.grid(True)
plt.show()


# This line plot illustrates how the average movie rating has changed over the years. This can help in identifying trends in movie quality or audience preferences over time.

# In[147]:


# Step 6: Apply Machine Learning Analysis
# Select features and target variable
X = df[['GENRE', 'VOTES', 'RunTime', 'YEAR']]
y = df['RATING']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Select a model (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')


# In[148]:


# Performance Summary
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual vs. Predicted Ratings')
plt.show()


# This scatter plot shows the performance of the Random Forest Regressor model by comparing the actual and predicted ratings. The closer the points are to the diagonal line, the better the model's performance.
