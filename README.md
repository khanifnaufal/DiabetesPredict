# DiaPredict: AI-Powered Diabetes Risk Assessment

DiaPredict is a professional web application designed to predict the risk of diabetes in patients using a high-performance machine learning model. The application leverages a Tuned Random Forest Classifier trained on medical health metrics (PIMA Diabetes Dataset) to provide real-time risk assessments.

## 🚀 Features

-   **High Accuracy Predictions**: Uses a performance-tuned Random Forest model optimized for medical diagnosis (high recall).
-   **Intelligent Data Preprocessing**:
    -   Automated handling of unrealistic zero values in medical metrics.
    -   Feature scaling using `RobustScaler` to manage outliers.
    -   Class balancing using `SMOTE` (Synthetic Minority Over-sampling Technique).
-   **Premium Medical UI**: A modern, responsive, and trustworthy interface designed for ease of use.
-   **Instant Analysis**: Get real-time probability scores and health recommendations based on individual profiles.

## 🛠️ Technology Stack

-   **Backend**: Flask (Python)
-   **Machine Learning**: Scikit-Learn, Pandas, Numpy, Imbalanced-learn
-   **Frontend**: HTML5, CSS3 (Vanilla JS, Glassmorphism design)
-   **Model Management**: Joblib

## 📋 Prerequisites

Ensure you have Python 3.8+ installed. You will need the following libraries:

```bash
pip install flask pandas scikit-learn imbalanced-learn joblib
```

## ⚙️ Installation & Usage

1. **Clone or Download** the project repository.
2. **Prepare the Model** (Optional - model.pkl is provided but can be regenerated):
   ```bash
   python train_model.py
   ```
3. **Start the Application**:
   ```bash
   python app.py
   ```
4. **Access the Website**:
   Open your browser and navigate to `http://127.0.0.1:5000`.

## 📂 Project Structure

```text
├── static/
│   ├── css/
│   │   └── style.css       # Premium medical styling
│   └── js/
│       └── main.js        # Frontend logic and API integration
├── templates/
│   └── index.html         # Main web interface
├── app.py                 # Flask server and prediction API
├── train_model.py         # ML pipeline: preprocessing & training
├── diabetes.csv           # Source dataset
├── model.pkl              # Saved Random Forest model
├── scaler.pkl             # Saved RobustScaler
└── README.md              # Project documentation
```

## 📊 Model Information

The underlying model is a **Random Forest Classifier** tuned with the following hyperparameters:
- `Criterion`: Entropy
- `Max Depth`: 10
- `N Estimators`: 100

The model was trained on data that underwent strict preprocessing to ensure clinical reliability, specifically focusing on minimizing False Negatives (Missed Diagnoses).

---
*Disclaimer: This tool is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.*
