import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Download the latest version of the dataset
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print("Path to dataset files:", path)

# Specify the path to the dataset
path2file = path + "/creditcard.csv"
df = pd.read_csv(path2file, sep=",")

# Check data types of each column
print("Data types of each column:")
print(df.dtypes)

# Check for missing values in each column
print("\nMissing values in each column:")
print(df.isnull().sum())

# Show basic statistics of the dataset (mean, std, min, max, etc.)
print("\nDescriptive statistics of the dataset:")
print(df.describe())

# Show a count plot for 'Class' to see the class distribution (fraud vs non-fraud)
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df, palette='viridis')
plt.title('Class Distribution:  Non-Fraud vs Fraud ')
plt.show()

# Visualizing the distribution of the 'Amount' variable (continuous)
plt.figure(figsize=(8, 6))
sns.histplot(df['Amount'], kde=True, color='blue', bins=50)
plt.title('Distribution of Amount')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.show()

# Visualizing the correlation matrix
# Select only numeric columns for the correlation matrix
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numerical_columns].corr()

# Adjust figure size for better readability
plt.figure(figsize=(15, 10))

# Create a heatmap to visualize correlations with a color palette
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, annot_kws={"size": 8})

plt.title('Correlation Matrix of Variables', fontsize=16)
plt.show()

