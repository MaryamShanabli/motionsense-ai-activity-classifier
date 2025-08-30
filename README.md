# MotionSense AI Project

## Project Overview
This project is part of the **MotionSense AI Competition**, aimed at analyzing human activity data collected via smartphones and wearables. The goal is to classify activities (SEDENTARY, STANDING, ACTIVE) and provide insights for healthier lifestyle recommendations.

**The project uses:**
- Data preprocessing and feature engineering
- Machine Learning models (Random Forest, Gradient Boosting, etc.)
- Hyperparameter tuning and model evaluation
- Python (NumPy, Pandas, Scikit-learn)
- Streamlit for interface integration

---

## Folder Structure
```
MotionSense_Project/
│
├── data/
│   ├── train_data_ready.csv
│   ├── test_data_ready.csv
│   ├── X_train_sel.npy
│   ├── X_val_sel.npy
│   ├── X_test_sel.npy
│   ├── y_train_app.npy
│   ├── y_val_app.npy
│   ├── y_test_app.npy
│   └── final_features.csv
│
├── models/
│   ├── best_gradient_boosting_model.joblib
│   └── robust_scaler.joblib
│
├── notebooks/
│   └── MotionSense_Project.ipynb
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md
```
---

## Setup Instructions

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

**3. Install dependencies**
```
pip install -r requirements.txt
```

---

## Usage

**Run the Jupyter Notebook**

```bash
jupyter notebook notebooks/MotionSense_Project.ipynb
```
**Run the Streamlit App**
```
streamlit run app/app.py
```

---

## Model Information

•	**Best Model: Gradient Boosting Classifier**

•	**Hyperparameters:**
```
{
    "learning_rate": 0.2,
    "max_depth": 4,
    "min_samples_split": 5,
    "n_estimators": 500,
    "subsample": 0.8
}
```
•	**Validation Accuracy: 0.9946**

•	**Test Accuracy: 0.9647**

---


## Feature Importance (Top 10)
```
  1.	tBodyGyroJerk-std()-Y
	2.	tGravityAcc-arCoeff()-Y,3
	3.	fBodyAccJerk-entropy()-X
	4.	fBodyAcc-bandsEnergy()-1,24.1
	5.	tGravityAcc-mean()-Z
	6.	tBodyGyroJerk-min()-X
	7.	fBodyAcc-bandsEnergy()-1,24
	8.	tBodyAccJerk-std()-X
	9.	tBodyGyroJerk-mad()-Z
	10.	angle(X,gravityMean)
```

---

## Notes 

- The data folder contains preprocessed and scaled features ready for modeling.
- The models folder contains the trained Gradient Boosting model and scaler for feature preprocessing.
- The app folder contains the Streamlit interface code (app.py) to run the demo.
- requirements.txt contains all Python dependencies to replicate the environment.

---
