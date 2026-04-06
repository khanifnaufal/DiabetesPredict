import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

def train_and_export():
    # Load dataset
    data_path = "diabetes.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    
    # Handle zeros in medical features by replacing with median
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zeros:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Resampling (SMOTE)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)
    
    # Best parameters from notebook:
    # criterion='entropy', max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=100
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        criterion='entropy',
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    
    model.fit(X_res, y_res)
    
    # Export model and scaler
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print("Model and Scaler have been exported successfully (model.pkl, scaler.pkl)")

if __name__ == "__main__":
    train_and_export()
