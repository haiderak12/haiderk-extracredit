import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from math import radians, sin, cos, asin, sqrt

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute the haversine distance between two points on Earth (in kilometers).
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of Earth in km
    return c * r

def prepare_features(df, label_encoders=None, scaler=None, is_train=True):
    """
    Preprocess the given DataFrame by extracting features and encoding categorical variables.
    """

    # Parse datetime from unix_time
    df['trans_datetime'] = pd.to_datetime(df['unix_time'], unit='s')
    df['hour'] = df['trans_datetime'].dt.hour
    df['day_of_week'] = df['trans_datetime'].dt.dayofweek
    df['day_of_year'] = df['trans_datetime'].dt.dayofyear
    df['weekend'] = (df['trans_datetime'].dt.weekday >= 5).astype(int)

    # Convert dob to datetime and calculate age
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
    df['age'] = (df['trans_datetime'] - df['dob']).dt.days // 365

    # Consider cc_num as user identifier
    user_id_col = 'cc_num'

    # Compute user-level transaction counts and mean amounts
    df['user_tx_count'] = df.groupby(user_id_col)[user_id_col].transform('count')
    df['user_avg_amount'] = df.groupby(user_id_col)['amt'].transform('mean')
    df['amt_diff_from_user_mean'] = df['amt'] - df['user_avg_amount']

    # Compute a derived feature: transaction amount * total user transactions
    df['amt_times_count'] = df['amt'] * df['user_tx_count']

    # Geographical distance between merchant and cardholder
    df['distance'] = df.apply(lambda row: haversine_distance(
        row['lat'], row['long'], row['merch_lat'], row['merch_long']
    ), axis=1)

    # Drop columns that are unlikely to be useful or already processed
    drop_cols = ['trans_date', 'trans_time', 'trans_num', 'unix_time', 
                 'first', 'last', 'street', 'city', 'zip', 
                 'lat', 'long', 'merchant', 'merch_lat', 'merch_long', 'dob',
                 'trans_datetime']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')

    # Categorical columns
    categorical_cols = ['category', 'gender', 'state', 'job']
    # Ensure these columns exist in df
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    # For training, fit LabelEncoders
    if is_train:
        label_encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_cols}
        for col in categorical_cols:
            df[col] = label_encoders[col].transform(df[col])
    else:
        # For test set, just transform using fitted encoders
        for col in categorical_cols:
            df[col] = label_encoders[col].transform(df[col])

    # Select numeric features for scaling
    numeric_cols = ['amt', 'city_pop', 'age', 'user_tx_count', 'user_avg_amount', 
                    'amt_diff_from_user_mean', 'amt_times_count', 'distance']
    # Handle potential missing ages or other numeric anomalies
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    if is_train:
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Final feature list
    feature_cols = categorical_cols + numeric_cols + [
        'hour', 'day_of_week', 'day_of_year', 'weekend'
    ]

    return df[feature_cols], label_encoders, scaler

# Load training data
train_df = pd.read_csv('./data/train.csv')
X_train, label_encoders, scaler = prepare_features(train_df, is_train=True)
y_train = train_df['is_fraud']

# Create a validation set
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.3, stratify=y_train, random_state=42
)

# Define XGBoost model (GPU-enabled)
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.07,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_tr, y_tr)

# Validate the model
y_val_pred = xgb_model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)
conf_matrix = confusion_matrix(y_val, y_val_pred)
class_report = classification_report(y_val, y_val_pred)

print("Validation Accuracy:", accuracy)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Load test data
test_df = pd.read_csv('./data/test.csv')
X_test, _, _ = prepare_features(test_df, label_encoders=label_encoders, scaler=scaler, is_train=False)

# Predict on the test data
test_df['is_fraud'] = xgb_model.predict(X_test)

# Save predictions to submission.csv
test_df.rename(columns={'trans_num': 'id'}, inplace=True)
test_df[['id', 'is_fraud']].to_csv('./data/submission.csv', index=False)
print("Predictions saved to submission.csv")