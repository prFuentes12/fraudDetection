import kagglehub
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

# Show the unique values in each categorical column (if applicable)
print("\nUnique values in categorical columns:")
categorical_columns = df.select_dtypes(include=['object']).columns
print(df[categorical_columns].nunique())

# Show a count plot for 'Class' to see the class distribution (fraud vs non-fraud)
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df, palette='viridis', hue='Class', legend=False)  # Fixed the deprecation warning
plt.title('Class Distribution: Fraud vs Non-Fraud')
plt.show()

# Visualizing the distribution of the 'Amount' variable (continuous)
plt.figure(figsize=(8, 6))
sns.histplot(df['Amount'], kde=True, color='blue', bins=50)
plt.title('Distribution of Amount')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.show()

# Visualizing the correlation matrix
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numerical_columns].corr()

# Adjust figure size for better readability
plt.figure(figsize=(15, 10))

# Create a heatmap to visualize correlations with a color palette
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, annot_kws={"size": 8})

plt.title('Correlation Matrix of Variables', fontsize=16)
plt.show()

# Separate the data into features (X) and target (y)
X = df.drop(columns=['Class'])  # Features (all columns except 'Class')
y = df['Class']  # Target variable (fraud or not)

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling the features
scaler = StandardScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data (using the same scaler)
X_test_scaled = scaler.transform(X_test)

# Optional: Apply PCA for dimensionality reduction
pca = PCA(n_components=30)  # Adjusted to the number of features
X_train_scaled_pca = pca.fit_transform(X_train_scaled)
X_test_scaled_pca = pca.transform(X_test_scaled)

# Create the Support Vector Classifier (SVC) model
svc_model = SVC(class_weight='balanced', kernel='linear', random_state=42, C=1, gamma='scale')

# Train the model
svc_model.fit(X_train_scaled_pca, y_train)

# Make predictions on training and test data
y_train_pred = svc_model.predict(X_train_scaled_pca)
y_test_pred = svc_model.predict(X_test_scaled_pca)

# Calculate metrics for training data
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

# Calculate metrics for test data
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

# Print the results
print("Training Metrics:")
print(f"Accuracy: {train_accuracy:.4f}")
print(f"Precision: {train_precision:.4f}")
print(f"Recall: {train_recall:.4f}")
print(f"F1 Score: {train_f1:.4f}")

print("\nTest Metrics:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1 Score: {test_f1:.4f}")

# Print classification reports
print("\nClassification Report (Train):")
print(classification_report(y_train, y_train_pred))

print("\nClassification Report (Test):")
print(classification_report(y_test, y_test_pred))

# Confusion matrix
print("\nConfusion Matrix (Train):")
print(confusion_matrix(y_train, y_train_pred))

print("\nConfusion Matrix (Test):")
print(confusion_matrix(y_test, y_test_pred))

