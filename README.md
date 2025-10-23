🏥 Health Insurance Cost Predictor

A Machine Learning web app built with Streamlit that predicts Health Insurance Costs based on personal, lifestyle, and medical details.
The model is trained on a dataset of 50,000 individuals and provides accurate, data-driven insurance cost predictions.

🚀 Features

✅ Predicts health insurance cost in real-time
✅ Simple and interactive Streamlit interface
✅ Supports multiple demographic and health-related parameters
✅ Trained on 50K+ records for improved accuracy
✅ Built and deployed directly from PyCharm

🧠 Model Overview

The machine learning model is trained in Jupyter Notebook using a dataset of 50,000 entries, containing the following features:

Feature	Description
Age	Age of the individual
Number of Dependents	Family members dependent on the user
Income in Lakhs	Annual income in INR (Lakhs)
Genetical Risk	Genetic predisposition risk score
Insurance Plan	Bronze / Silver / Gold / Platinum
Employment Status	Salaried / Self-employed / Unemployed / Retired
Gender	Male / Female / Other
Marital Status	Married / Unmarried / Divorced
BMI Category	Underweight / Normal / Overweight / Obese
Smoking Status	Smoker / No Smoking
Region	Northeast / Northwest / Southeast / Southwest
Medical History	Past medical history (e.g., No Disease, Diabetes, etc.)
🧩 Tech Stack

Model Training:

Python

Pandas, NumPy

Scikit-learn

Matplotlib / Seaborn (for visualization)

Jupyter Notebook

Frontend (App):

Streamlit

PyCharm IDE

⚙️ How It Works

The user enters demographic, health, and lifestyle details in the Streamlit interface.

Data is preprocessed and passed to the trained ML model.

The model predicts the estimated annual insurance cost.

The result is displayed instantly on the app interface.

🧪 Setup Instructions
1️⃣ Clone Repository
git clone https://github.com/RudyMontoo/Health-Care-Premium-Prediction.git
cd health-insurance-cost-predictor

2️⃣ Install Dependencies

Make sure you have Python 3.8+ and pip installed. Then run:

pip install -r requirements.txt


joblib==1.3.2
pandas==2.0.2
streamlit==1.22.0
numpy==1.25.0
scikit-learn==1.3.0
xgboost==2.0.3

3️⃣ Run the App

In your PyCharm terminal or command prompt:

streamlit run app.py

4️⃣ Access the App

Once it starts, open your browser and visit:
👉 http://localhost:8501

🧮 Model Training

To retrain the model on your dataset:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
data = pd.read_csv('insurance_dataset.csv')

# Split data
X = data.drop('insurance_cost', axis=1)
y = data['insurance_cost']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'insurance_model.pkl')


Then, in your Streamlit app, load it as:

import joblib
model = joblib.load('insurance_model.pkl')

📊 Future Enhancements

Add feature importance visualization (e.g., SHAP values)

Add prediction history tracking

Integrate authentication for multiple users

Deploy publicly on Streamlit Cloud or Render

