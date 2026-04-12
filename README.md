# Health Risk Predictor

A machine learning application that predicts patient health risk levels (Low, Medium, High) from structured clinical data. The project includes a complete ML pipeline with data ingestion, transformation, model training, and a FastAPI web service for predictions.

![UI Preview](ui-design.png)

## Quick Start

```bash
git clone https://github.com/ariful59/ml-pipeline-heath-risk_predictor.git
cd ml-pipeline-heath-risk_predictor
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

## Dataset

The model is trained on 1,500 synthetic patient records containing:

- **Demographics**: Age, Gender
- **Clinical Information**: Symptoms (comma-separated), Medical History, Lab Reports
- **Treatment**: Current Medications, Diagnosis
- **Lifestyle**: Activity level and habits

**Target Variable**: Risk_Level with three classes (Low, Medium, High)

The dataset is automatically loaded from Hugging Face during training or can be placed in `artifacts/data.csv`.

## Exploratory Data Analysis

The `src/notebook/` directory contains Jupyter notebooks for data exploration:

**EDA_dataset.ipynb**
- Distribution analysis of features and target variable
- Missing value analysis
- Correlation between features
- Visualization of risk levels across different demographics

**model_training.ipynb**
- Model experimentation and comparison
- Hyperparameter tuning
- Performance metrics evaluation
- Feature importance analysis

Run notebooks to understand the data before training:

```bash
jupyter notebook src/notebook/EDA_dataset.ipynb
```

## ML Pipeline

The training pipeline consists of three main components:

**1. Data Ingestion** (`src/components/data_ingestion.py`)
- Loads dataset from Hugging Face or local CSV
- Splits data into training and test sets
- Saves splits to `artifacts/train.csv` and `artifacts/test.csv`

**2. Data Transformation** (`src/components/data_transformation.py`)
- Numeric feature scaling
- Categorical feature encoding
- Feature engineering (creates Symptom_Count from comma-separated symptoms)
- Saves fitted preprocessor to `artifacts/preprocessor.pkl`

**3. Model Training** (`src/components/model_trainer.py`)
- Trains multiple models: Random Forest, XGBoost, Decision Tree, Logistic Regression, AdaBoost
- Evaluates each model on test set
- Saves best performing model to `artifacts/model.pkl`

**Run the complete pipeline:**

```bash
python src/pipeline/training_pipeline.py
```

This executes all three components sequentially and generates artifacts for deployment.

## FastAPI Application

The application is built with FastAPI and serves predictions through a web interface and REST API.

**main.py** - Main application file containing:
- Web UI route that renders an HTML form
- Form submission endpoint for browser-based predictions
- JSON API endpoint for programmatic access
- Automatic API documentation

**Prediction Pipeline** (`src/pipeline/predict_pipeline.py`)
- Loads trained model and preprocessor
- Applies same transformations used during training
- Returns risk level prediction

**Available Endpoints:**

- `GET /` - Web interface with form
- `POST /predict` - Form submission from web UI
- `POST /api/predict` - JSON API endpoint
- `GET /docs` - Interactive API documentation



## Deployment to Azure

The repository includes a GitHub Actions workflow for automated deployment to Azure App Service. The workflow is configured at `.github/workflows/main_health-predictor.yml` and triggers on push to the main branch.

---

## Project Structure

```
ml-pipeline-heath-risk_predictor/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   ├── training_pipeline.py
│   │   └── predict_pipeline.py
│   └── notebook/
│       ├── EDA_dataset.ipynb
│       └── model_training.ipynb
├── artifacts/
│   ├── data.csv
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessor.pkl
│   └── model.pkl
├── templates/
│   └── index.html
├── static/
├── main.py
└── requirements.txt
```
