🏥 Health Insurance Cost Predictor

An intelligent Machine Learning web app that predicts the estimated health insurance cost based on personal, lifestyle, and medical factors.
The model was trained on a dataset of 50,000 individuals and provides accurate and explainable cost predictions.

🚀 Features

✅ Predicts health insurance cost in seconds
✅ User-friendly web interface
✅ Interactive inputs (dropdowns, increment/decrement controls)
✅ Supports multiple demographic and health-related features
✅ Trained on 50K+ data records for better accuracy
✅ Built with modern ML and web technologies

🧠 Model Overview

The model uses supervised machine learning to predict insurance costs based on the following features:

Feature	Description
Age	Age of the individual
Number of Dependents	Number of family members dependent on the individual
Income in Lakhs	Annual income in INR (Lakhs)
Genetical Risk	Genetic risk factor (0–10 scale)
Insurance Plan	Type of plan (Bronze, Silver, Gold, Platinum)
Employment Status	Employment type (Salaried, Self-employed, Unemployed, Retired)
Gender	Male / Female / Other
Marital Status	Married / Unmarried / Divorced
BMI Category	Underweight / Normal / Overweight / Obese
Smoking Status	Smoker / No Smoking
Region	Region of residence (Northeast, Northwest, Southeast, Southwest)
Medical History	Past medical condition (None, Diabetes, Heart Disease, etc.)
🧩 Tech Stack

Machine Learning:

Python

Pandas, NumPy

Scikit-learn / XGBoost / RandomForestRegressor (depending on your model)

Frontend:

React.js (or Streamlit, depending on your app)

Tailwind CSS / ShadCN UI

Backend (optional):

Flask / FastAPI (for model serving)

Deployment:

Vercel / Streamlit Cloud / Render

⚙️ How It Works

Input Data: User enters personal and health-related details.

Data Preprocessing: Model encodes categorical variables and normalizes numerical inputs.

Prediction: Trained ML model computes the estimated insurance cost.

Output: Predicted cost is displayed instantly on the UI.

📈 Example Prediction
Input	Output
Age: 30, Income: 6 LPA, BMI: Overweight, Smoker	💰 Predicted Cost: ₹46,000/year
🧪 Setup Instructions
1️⃣ Clone Repository
git clone https://github.com/your-username/health-insurance-predictor.git
cd health-insurance-predictor

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the App

If using Flask/FastAPI backend:

python app.py


If using Streamlit:

streamlit run app.py

4️⃣ Access the App

Open http://localhost:8501
 (Streamlit) or your specified port for Flask.

🧮 Model Training (optional)

To retrain the model on your dataset:

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'insurance_model.pkl')

📊 Future Enhancements

🔍 Explainable AI integration (SHAP/LIME)

☁️ Cloud deployment with live database

📈 Dashboard for data analytics and insights

🔐 User login & personalized history

👨‍💻 Author

Your Name
📧 rudraeng27@gmail.com

🌐 https://github.com/RudyMontoo
