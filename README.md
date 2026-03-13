# SentinelX Safety Engine
### AI-Powered Accident Severity Prediction System

SentinelX Safety Engine is an end-to-end machine learning system designed to analyze traffic accident data and estimate accident severity using environmental and temporal features.

The project demonstrates how to move beyond simple ML notebooks and build a **production-style machine learning pipeline** with modular architecture, reproducible training workflows, and scalable data processing.

---

# Project Overview

Road traffic accidents are a major global safety concern. Predicting the severity of accidents can help authorities understand risk factors, improve emergency response, and design safer road systems.

SentinelX Safety Engine analyzes historical accident data and trains machine learning models to predict the severity of an accident based on environmental conditions such as:

- Weather conditions
- Visibility
- Time of accident
- Infrastructure indicators
- Traffic patterns

The system implements an **end-to-end ML pipeline** including:

- Data ingestion
- Data preprocessing
- Feature engineering
- Model training
- Evaluation
- Prediction pipeline

---

# Key Features

- Modular ML pipeline architecture
- Automated data ingestion and preprocessing
- Feature engineering and transformation pipeline
- Random Forest based severity prediction model
- Serialized preprocessing and model artifacts
- Reproducible training workflow
- Scalable pipeline capable of handling large datasets

---

# Tech Stack

Programming Language  
Python

Machine Learning  
Scikit-learn  
NumPy  
Pandas  

Data Processing  
ColumnTransformer  
Feature Scaling  
One-Hot Encoding  

Model  
Random Forest Regressor

Development Tools  
Jupyter Notebook  
Git  
GitHub

---

# Project Architecture

The system follows a modular architecture commonly used in production ML systems.
Raw Dataset
↓
Data Ingestion
↓
Data Validation
↓
Data Transformation
↓
Feature Engineering
↓
Model Training
↓
Model Evaluation
↓
Serialized Model
↓
Prediction Pipeline

---

# Machine Learning Pipeline

The pipeline consists of the following components:

### Data Ingestion
Downloads and loads the dataset while performing basic validation checks.

### Data Transformation
Preprocesses the raw dataset using a transformation pipeline:

Numerical Features  
- Median Imputation  
- Standard Scaling  

Categorical Features  
- Mode Imputation  
- One-Hot Encoding  

### Model Training
Trains a Random Forest Regressor and performs hyperparameter tuning.

### Model Evaluation
Evaluates the model using metrics such as:

- R² Score
- Mean Squared Error

### Prediction Pipeline
Loads serialized preprocessing objects and trained models to generate predictions on new data.

---

# Dataset

The project uses the **US Accidents Dataset**, a large dataset containing millions of accident records collected from traffic monitoring systems.

Important features include:

- Weather conditions
- Visibility
- Temperature
- Road infrastructure indicators
- Time of accident
- Traffic signal presence

---

# Project Structure
SentinelX-Safety-Engine
│
├── notebooks
│ ├── notebook1.ipynb
│
├── src
│ ├── components
│ │ ├── data_ingestion.py
│ │ ├── data_transformation.py
│ │ ├── model_trainer.py
│ │
│ ├── pipeline
│ │ ├── training_pipeline.py
│ │ ├── prediction_pipeline.py
│ │
│ ├── utils
│ │ ├── logger.py
│ │ ├── exception.py
│
├── artifacts
│ ├── model.pkl
│ ├── preprocessor.pkl
│
├── requirements.txt
├── README.md
└── main.py

---

# Installation

Clone the repository:
git clone https://github.com/Rohit-03-19/SentinelX-Safety-Engine.git

cd SentinelX-Safety-Engine


Create a virtual environment:


python -m venv venv


Activate the environment:

Mac/Linux


source venv/bin/activate


Windows


venv\Scripts\activate


Install dependencies:


pip install -r requirements.txt


---

# Running the Project

Run the training pipeline:


python main.py


This will:

- ingest the dataset
- preprocess the data
- train the model
- store trained artifacts

---

# Making Predictions

To run inference:


python prediction_pipeline.py


The prediction pipeline loads:

- saved preprocessing pipeline
- trained model

and generates severity predictions for new input data.

---

# Example Output

Example model prediction:


Input:
Weather = Rain
Visibility = 3 miles
Hour = 18
Traffic Signal = Yes

Output:
Predicted Severity Score = 2.31


---

# Results

Model Performance

R² Score: 0.0447

Although the score is relatively low, it reflects the complexity of predicting accident severity because many factors such as driver behavior and vehicle condition are not included in the dataset.

The model therefore serves as a **baseline predictive system** for environmental risk estimation.

---

# Engineering Challenges

During development, several engineering challenges were encountered:

Memory bottlenecks during model training  
Solution: implemented dataset sampling strategy

Feature dimension mismatch after encoding  
Solution: applied NumPy reshaping and alignment

Pipeline interface mismatches  
Solution: improved modular pipeline architecture

---

# Future Improvements

Possible future enhancements include:

- Implementing XGBoost or LightGBM models
- Advanced feature engineering
- Hyperparameter optimization
- Real-time API deployment using FastAPI
- Web dashboard for prediction visualization

---

# Author

Rohit Parida

Data Science / Machine Learning Enthusiast

---

# License

This project is released under the MIT License.
