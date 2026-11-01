# 🌾 Crop Yield Prediction

A machine learning web application that predicts agricultural crop yield using historical agricultural data from Indian states.

The project uses **Python, Pandas, NumPy, Scikit-learn, Joblib, and Streamlit** to build a complete machine learning pipeline from data preprocessing and model training to prediction and deployment.

---

## 🚀 Live Demo

👉 https://crop-yield-prediction-zketoa5sfgplfuzsqr4eho.streamlit.app/

## 💻 GitHub Repository

👉 https://github.com/alinakhan1796/Crop-Yield-Prediction

---

## 📌 Project Overview

Crop yield prediction can help understand how different agricultural factors influence crop productivity.

This project uses historical agricultural data from Indian states and trains a **Random Forest Regression** model to predict crop yield based on:

- Crop
- Crop Year
- Season
- State
- Area
- Annual Rainfall
- Fertilizer Usage
- Pesticide Usage

The application provides an interactive Streamlit interface where users can enter agricultural information and receive an estimated crop yield.

---

## ✨ Features

- 🌾 Crop yield prediction
- 📊 Historical agricultural dataset
- 🧹 Data preprocessing pipeline
- 🔤 Categorical feature encoding using OneHotEncoder
- 🤖 Random Forest Regression model
- 📈 Model evaluation using MAE, RMSE, and R²
- 💾 Saved trained model using Joblib
- 🔮 Real-time predictions
- 📝 Prediction history
- 🌐 Interactive Streamlit web application
- ☁️ Streamlit Community Cloud deployment
- 🔒 Prevention of target leakage by excluding `Production`
- 📁 Clean project structure
- 🐙 Git and GitHub version control

---

## 🧠 Machine Learning Model

### Random Forest Regression

The project uses a Random Forest Regressor from Scikit-learn.

```python
RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
```

### Why Random Forest?

Random Forest was selected because it:

- Handles nonlinear relationships well
- Works effectively with mixed agricultural features
- Is robust to noise
- Can capture complex relationships between input variables
- Provides strong performance without requiring complex feature engineering

---

## 📊 Dataset

The dataset used in this project is:

**Agricultural Crop Yield in Indian States Dataset**

Source:

https://www.kaggle.com/datasets/akshatgupta7/crop-yield-in-indian-states-dataset

### Dataset Information

- Records: **19,689**
- Crops: **55**
- States/UTs: **30**
- Years: **1997–2020**
- Seasons: **6**

### Original Features

| Feature | Description |
|---|---|
| Crop | Name of the crop |
| Crop_Year | Year of cultivation |
| Season | Agricultural season |
| State | Indian state/UT |
| Area | Cultivated area |
| Production | Crop production |
| Annual_Rainfall | Annual rainfall |
| Fertilizer | Fertilizer usage |
| Pesticide | Pesticide usage |
| Yield | Crop yield |

---

## ⚠️ Data Leakage Prevention

A major part of this project was identifying and preventing **target leakage**.

The dataset contains both:

- `Production`
- `Area`
- `Yield`

Since crop yield is mathematically related to production and area, using `Production` as an input feature would allow the model to indirectly access information very closely related to the target.

Therefore, `Production` was deliberately **excluded from the model features**.

### Final Model Features

```text
Crop
Crop_Year
Season
State
Area
Annual_Rainfall
Fertilizer
Pesticide
```

### Target

```text
Yield
```

This makes the prediction pipeline more realistic and demonstrates awareness of an important machine learning problem: **target leakage**.

---

## 🔄 Machine Learning Workflow

The project follows a complete machine learning workflow:

```text
Raw Dataset
     ↓
Data Loading
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Selection
     ↓
Target Leakage Analysis
     ↓
Train/Test Split
     ↓
Categorical Encoding
     ↓
Feature Transformation
     ↓
Random Forest Training
     ↓
Model Evaluation
     ↓
Model Serialization
     ↓
Prediction Pipeline
     ↓
Streamlit Application
     ↓
Cloud Deployment
```

---

## 🧹 Data Preprocessing

The preprocessing pipeline includes:

### 1. Data Cleaning

Categorical columns were cleaned using whitespace removal.

```python
df["Crop"] = df["Crop"].str.strip()
df["Season"] = df["Season"].str.strip()
df["State"] = df["State"].str.strip()
```

### 2. Feature Selection

The following features were selected:

```text
Crop
Crop_Year
Season
State
Area
Annual_Rainfall
Fertilizer
Pesticide
```

### 3. Target Selection

```text
Target = Yield
```

### 4. Train/Test Split

The dataset was divided into:

- 80% training data
- 20% testing data

Using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### 5. Categorical Encoding

Categorical features were converted into numerical representations using:

```python
OneHotEncoder(
    handle_unknown="ignore"
)
```

The preprocessing pipeline generated **96 processed features**.

---

## 📈 Model Performance

The Random Forest Regression model achieved the following results on the test dataset:

| Metric | Score |
|---|---:|
| MAE | 9.54 |
| RMSE | 126.38 |
| R² Score | 0.9801 |

### R² Score

```text
R² ≈ 0.98
```

This indicates that the model explains approximately **98% of the variance** in the test dataset.

### Important Note

The dataset contains extreme yield values, particularly for certain crop categories such as Coconut. These outliers contribute significantly to the RMSE.

The outliers were not simply removed because they may represent genuine agricultural observations.

---

## 🌐 Streamlit Application

The trained machine learning model is integrated into a Streamlit web application.

Users can enter:

- Crop
- Season
- State
- Crop Year
- Area
- Annual Rainfall
- Fertilizer
- Pesticide

The application then processes the inputs using the saved preprocessing pipeline and generates a predicted crop yield.

### Application Features

- Professional dashboard-style interface
- Interactive input controls
- Input validation
- Real-time prediction
- Prediction history
- Clear history option
- Model performance information
- Dataset information
- Explanation of data leakage prevention

---

## 📁 Project Structure

```text
Crop-Yield-Prediction/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── data/
│   └── crop_yield.csv
│
├── models/
│   ├── crop_yield_model.pkl
│   └── preprocessor.pkl
│
├── src/
│   ├── preprocessing.py
│   └── predict.py
│
└── venv/
```

### File Description

| File/Folder | Purpose |
|---|---|
| `app.py` | Streamlit web application |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Files excluded from Git |
| `data/` | Dataset storage |
| `models/` | Trained model and preprocessing objects |
| `src/preprocessing.py` | Data preprocessing and model training |
| `src/predict.py` | Standalone prediction script |
| `venv/` | Python virtual environment |

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Data Analysis

- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn

### Model Serialization

- Joblib

### Web Application

- Streamlit

### Version Control

- Git
- GitHub

### Deployment

- Streamlit Community Cloud

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/amanawasthi-lang/Crop-Yield-Prediction.git
```

### 2. Navigate to the Project

```bash
cd Crop-Yield-Prediction
```

### 3. Create Virtual Environment

```bash
python -m venv venv
```

### 4. Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### macOS/Linux

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit Application

Run:

```bash
streamlit run app.py
```

The application will open in the browser.

Usually:

```text
http://localhost:8501
```

---

## 🔮 Prediction Pipeline

The prediction process works as follows:

```text
User Input
    ↓
Create Input DataFrame
    ↓
Load Preprocessor
    ↓
Transform Input Features
    ↓
Load Random Forest Model
    ↓
Generate Prediction
    ↓
Display Crop Yield
```

The same preprocessing logic used during model training is reused during prediction to maintain consistency.

---

## 💾 Model Serialization

The trained model and preprocessing pipeline are saved using Joblib.

```text
models/crop_yield_model.pkl
models/preprocessor.pkl
```

This allows the Streamlit application to load the trained objects without retraining the model every time.

The trained Random Forest model was also compressed during serialization to significantly reduce its file size and make GitHub deployment practical.

---

## 🔍 Error Analysis

Model predictions were analyzed using actual-vs-predicted values and error analysis.

One important observation was the presence of extreme yield values in the dataset.

These extreme values can increase:

- MAE
- RMSE

especially when the model makes larger errors on rare observations.

Instead of blindly removing these observations, they were retained because they may represent real agricultural conditions.

---

## 📚 Key Learning Outcomes

This project demonstrates practical understanding of:

- Python programming
- Pandas data manipulation
- NumPy
- Exploratory Data Analysis
- Data cleaning
- Feature selection
- Target leakage
- Train/test splitting
- Categorical feature encoding
- Scikit-learn pipelines
- Random Forest Regression
- Regression evaluation metrics
- MAE
- RMSE
- R² Score
- Model serialization with Joblib
- Streamlit application development
- Git and GitHub
- Cloud deployment
- Machine learning project structuring

---

## 🚀 Future Improvements

Possible future improvements include:

- Hyperparameter tuning
- Cross-validation
- Feature importance visualization
- More advanced regression models
- XGBoost/Gradient Boosting comparison
- Improved outlier analysis
- Model explainability using SHAP
- More detailed agricultural visualizations
- Prediction confidence/range estimation
- Additional environmental features
- Larger and more recent datasets

---

## 👨‍💻 Author

**Aman Awasthi**

GitHub:  
https://github.com/amanawasthi-lang

LinkedIn:  
https://www.linkedin.com/in/aman-awasthi-b10886275/

---

## ⭐ Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📌 Disclaimer

This project is developed for educational and demonstration purposes.

The predicted crop yield should not be considered a guaranteed agricultural outcome. Real-world crop yield depends on many additional factors including weather conditions, soil characteristics, irrigation, farming practices, crop variety, pests, diseases, and other environmental factors.
