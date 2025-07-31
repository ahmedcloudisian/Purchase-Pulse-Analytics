import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load the dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        print("Dataset loaded successfully!")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def explore_data(data):
    """Perform basic data exploration."""
    print("\n--- Dataset Overview ---")
    print(data.head())  # Display the first 5 rows
    print("\n--- Dataset Summary ---")
    print(data.describe())  # Statistical summary
    print("\n--- Missing Values ---")
    print(data.isnull().sum())  # Check for missing values

def analyze_purchase_frequency(data):
    """Analyze purchase frequency by customer."""
    print("\n--- Purchase Frequency Analysis ---")
    purchase_frequency = data['customer_id'].value_counts()
    print(purchase_frequency.describe())  # Summary of purchase frequency

    # Plot purchase frequency distribution
    plt.figure(figsize=(10, 6))
    purchase_frequency.hist(bins=50, color='skyblue', edgecolor='black')
    plt.title('Purchase Frequency Distribution')
    plt.xlabel('Number of Purchases')
    plt.ylabel('Number of Customers')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def analyze_average_purchase_value(data):
    """Calculate and analyze average purchase value by customer."""
    print("\n--- Average Purchase Value Analysis ---")
    # Calculate purchase value for each transaction
    data['purchase_value'] = data['quantity'] * data['price']

    # Calculate average purchase value per customer
    avg_purchase_value = data.groupby('customer_id')['purchase_value'].mean()
    print(avg_purchase_value.describe())  # Summary of average purchase value

    # Plot average purchase value distribution
    plt.figure(figsize=(10, 6))
    avg_purchase_value.hist(bins=50, color='lightgreen', edgecolor='black')
    plt.title('Average Purchase Value Distribution')
    plt.xlabel('Average Purchase Value')
    plt.ylabel('Number of Customers')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def main():
    # File path to the dataset
    file_path = 'purchase_data.csv'

    # Step 1: Load the dataset
    data = load_data(file_path)
    if data is None:
        return  # Exit if data loading fails

    # Step 2: Explore the dataset
    explore_data(data)

    # Step 3: Analyze purchase frequency
    analyze_purchase_frequency(data)

    # Step 4: Analyze average purchase value
    analyze_average_purchase_value(data)

if __name__ == "__main__":
    main()


Data set 

customer_id	product_id	quantity	price	purchase_date	region
1	101	2	10.0	2023-10-01	North
2	102	1	15.0	2023-10-02	South
1	103	3	20.0	2023-10-03	North
3	101	5	5.0	2023-10-04	East
2	104	2	10.0	2023-10-05	South
4	105	1	50.0	2023-10-06	West
5	102	4	15.0	2023-10-07	North
3	103	2	20.0	2023-10-08	East
1	104	3	10.0	2023-10-09	North
2	105	1	50.0	2023-10-10	South
4	101	2	5.0	2023-10-11	West
5	103	1	20.0	2023-10-12	North
3	104	4	10.0	2023-10-13	East
1	105	2	50.0	2023-10-14	North
2	101	3	5.0	2023-10-15	South



Complete Source Code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import urllib.request
import uuid
import os

# Step 1: Load and Preprocess the Dataset
# Load dataset
df = pd.read_csv("online_shoppers_intention.csv")

# Handle missing values (if any)
df = df.dropna()

# Encode categorical variables
le = LabelEncoder()
df['VisitorType'] = le.fit_transform(df['VisitorType'])
df['Month'] = le.fit_transform(df['Month'])
df['Weekend'] = df['Weekend'].astype(int)
df['Revenue'] = df['Revenue'].astype(int)  # Target variable (0 or 1)

# Step 2: Exploratory Data Analysis (EDA)
# Summary statistics
print("Dataset Summary:")
print(df.describe())

# Correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Features')
plt.savefig('correlation_matrix.png')
plt.close()

# Distribution of ProductRelated_Duration
plt.figure(figsize=(8, 6))
sns.histplot(df['ProductRelated_Duration'], bins=30, kde=True)
plt.title('Distribution of Product-Related Page Duration')
plt.xlabel('Duration (seconds)')
plt.ylabel('Count')
plt.savefig('product_duration_distribution.png')
plt.close()

# Purchase rate by VisitorType
plt.figure(figsize=(8, 6))
sns.countplot(x='VisitorType', hue='Revenue', data=df)
plt.title('Purchase Rate by Visitor Type')
plt.xlabel('Visitor Type (Encoded)')
plt.ylabel('Count')
plt.savefig('purchase_by_visitor_type.png')
plt.close()

# Step 3: Data Preprocessing
# Features and target
features = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
            'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues',
            'SpecialDay', 'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType',
            'VisitorType', 'Weekend']
X = df[features]
y = df['Revenue']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Step 6: Save Predictions to CSV
# Create a DataFrame with test features, actual labels, and predictions
results_df = X_test.copy()
results_df['Actual_Revenue'] = y_test
results_df['Predicted_Revenue'] = y_pred

# Path to CSV file
csv_file = 'prediction_results.csv'

# Check if file exists to append or create new
if os.path.exists(csv_file):
    # Append to existing CSV without header
    results_df.to_csv(csv_file, mode='a', header=False, index=False)
else:
    # Create new CSV with header
    results_df.to_csv(csv_file, mode='w', header=True, index=False)

print(f"Predictions saved to {csv_file}")

# Step 7: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.close()

# Step 8: Feature Importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance in Random Forest Model')
plt.savefig('feature_importance.png')
plt.close()

# Step 9: Save Results Summary
results = f"""
Purchase Pulse Analytics Results
==============================
Dataset: Online Shoppers Purchasing Intention (UCI)
Dataset Size: {len(df)} records
Model: Random Forest Classifier
Accuracy: {accuracy:.2f}
Precision: {precision:.2f}
Recall: {recall:.2f}
==============================
Visualizations Generated:
- correlation_matrix.png
- product_duration_distribution.png
- purchase_by_visitor_type.png
- confusion_matrix.png
- feature_importance.png
==============================
Predictions Saved:
- prediction_results.csv
"""
with open('results_summary.txt', 'w') as f:
    f.write(results)

print("\nResults, visualizations, and predictions saved successfully.")
