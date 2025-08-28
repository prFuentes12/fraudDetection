import kagglehub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import mlflow

mlflow.set_tracking_uri("file:./tracking")
mlflow.set_experiment("creditcard_fraud_rf")

# Download the latest version of the dataset
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print("Path to dataset files:", path)

path2file = path + "/creditcard.csv"
df = pd.read_csv(path2file, sep=",")

# Separate the data into features (X) and target (y)
X = df.drop(columns=['Class'])  # Features (all columns except 'Class')
y = df['Class']  # Target variable (fraud or not)


# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

with mlflow.start_run(run_name= "rf_baseline_balanced"):



    # Create the Random Forest model with class_weight='balanced' to handle class imbalance
    model_rf = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)

    # Train the model
    model_rf.fit(X_train, y_train)

    # Make predictions on training and test data
    y_test_pred = model_rf.predict(X_test)

 
    ap = average_precision_score(y_test, y_test_pred)   # PR-AUC 
    auc = roc_auc_score(y_test, y_test_pred)            # ROC-AUC

    mlflow.log_metric("AP", ap)
    mlflow.log_metric("ROC_AUC", auc)


    mlflow.log_params({
        "split": "stratified_80_20",
        "class_weight": "balanced",
        "n_estimators": 100
    })


    # Metrics PR AUC used in desbalanced datasets
    display = PrecisionRecallDisplay.from_predictions(
        y_test, y_test_pred, name="RandomForest", plot_chance_level=True, despine=True
    )
    
    display.ax_.set_title("2-class Precision–Recall curve")
    plt.show()

    #Save model
    mlflow.sklearn.log_model(model_rf, name="model")
    mlflow.sklearn.save_model(model_rf, "my_model")
