# novapay-fraud-detection
Machine Learning Fraud Detection Project for NovaPay Fintech Transactions


Fraud Detection ML Platform
Project Overview

This project is an end-to-end Machine Learning fraud detection platform developed to identify potentially fraudulent financial transactions in near real time. The solution combines data engineering, machine learning, application development, and containerisation to deliver a production-ready analytical system.

The platform uses transactional data to predict whether a transaction is legitimate or fraudulent using supervised machine learning techniques. The solution includes both a Streamlit user interface for interactive risk assessment and a FastAPI REST API for production-style model serving.

Objectives

The objectives of this project were to:

Build an end-to-end fraud detection solution.
Analyse transaction patterns and identify fraud indicators.
Engineer meaningful features to improve model performance.
Develop and compare multiple machine learning models.
Deploy the solution through Streamlit and FastAPI.
Containerise the application using Docker.
Prepare the solution for cloud deployment.
Technologies Used
Programming
Python
SQL
Data Science & Machine Learning
Pandas
NumPy
Scikit-learn
XGBoost
Data Visualisation
Matplotlib
Seaborn
Application Development
Streamlit
FastAPI
Containerisation
Docker
Version Control
Git
GitHub
Cloud Deployment (Ready)
AWS ECS
AWS Fargate
Amazon ECR
Project Architecture
Raw Transaction Data
          ↓
Data Cleaning
          ↓
Exploratory Data Analysis
          ↓
Feature Engineering
          ↓
Model Development
(Logistic Regression, Random Forest, XGBoost)
          ↓
Model Evaluation
          ↓
Streamlit Application
          ↓
FastAPI REST API
          ↓
Docker Container
          ↓
Cloud Deployment
Machine Learning Models

The following models were developed and evaluated:

Logistic Regression
Random Forest
XGBoost

Performance was assessed using:

Accuracy
Precision
Recall
F1-Score
ROC-AUC
Confusion Matrix
Precision-Recall Curves

The Random Forest model was selected for deployment.

Streamlit Application

The Streamlit application enables users to:

Input transaction information.
Calculate fraud risk in real time.
View fraud probability scores.
Identify suspicious transactions.

Features include:

Interactive sidebar inputs
Automated feature encoding
Risk prediction
Fraud probability scoring

Run locally:

streamlit run app/streamlit_app.py
FastAPI REST API

The FastAPI application exposes the model through REST endpoints.

Available endpoints:

GET /
GET /health
POST /predict

Run locally:

uvicorn api.fastApi:app --reload

API documentation:

http://localhost:8000/docs
Docker Containerisation

The application was containerised using Docker to ensure consistent execution across development and production environments.

Build image:

docker build -t fraud-detection-app .

Run Streamlit:

docker run -p 8501:8501 fraud-detection-app

Run FastAPI:

docker build -f Dockerfile.api -t fraud-detection-api .

docker run -p 8000:8000 fraud-detection-api
Key Outcomes
Built an end-to-end fraud detection platform.
Developed and compared multiple machine learning models.
Created interactive Streamlit and FastAPI applications.
Containerised applications using Docker.
Implemented a cloud deployment-ready architecture.
Demonstrated principles of MLOps and production deployment.
Future Enhancements
Deploy to AWS ECS and Fargate.
Integrate CI/CD pipelines.
Add model monitoring and alerting.
Implement automated retraining pipelines.
Introduce explainability using SHAP.
Author



Batholomew Ohanme

MSc Data Science | Data Engineer | Machine Learning Practitioner

GitHub: https://github.com/Barth40
streamlit run app/streamlit_app.py