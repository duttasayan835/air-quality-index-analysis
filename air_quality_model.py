import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load dataset
print("Loading and preprocessing data...")
file_path = "air quality data.csv"
df = pd.read_csv(file_path)

# Drop rows with missing AQI_Bucket (target)
df = df.dropna(subset=["AQI_Bucket"])

# Define feature columns
features = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]

# Advanced preprocessing
# 1. Handle outliers using IQR method
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df

# 2. Fill missing values with median for each AQI bucket
for feature in features:
    df[feature] = df.groupby('AQI_Bucket')[feature].transform(
        lambda x: x.fillna(x.median())
    )

# Remove outliers
df = remove_outliers(df, features)

# Feature scaling
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Encode target variable
label_encoder = LabelEncoder()
df["AQI_Bucket"] = label_encoder.fit_transform(df["AQI_Bucket"])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df[features], 
    df["AQI_Bucket"], 
    test_size=0.2, 
    random_state=42,
    stratify=df["AQI_Bucket"]
)

# Handle class imbalance using SMOTE
print("Applying SMOTE for class balancing...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Define a simpler model with fixed hyperparameters
print("\nTraining the model...")
xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective='multi:softmax',
    tree_method='hist',
    random_state=42,
    use_label_encoder=False,
    eval_metric="mlogloss"
)

# Train the model with early stopping
eval_set = [(X_test, y_test)]
xgb_model.fit(
    X_train_res, 
    y_train_res,
    eval_set=eval_set,
    verbose=True
)

# Evaluate the model
print("\nEvaluating model performance...")
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Classification Report:")
print(report)

# Save the model and preprocessing objects
print("\nSaving model and preprocessing objects...")
joblib.dump(xgb_model, "best_xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
np.save("label_encoder_classes.npy", label_encoder.classes_)

print("Model training and saving completed successfully!")