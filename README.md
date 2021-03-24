# Udacity Capstone Project - Using Machine Learning to Predict Survival of Patients with Heart Failure


In this project, we have created two models: one using Automated ML (AutoML) and one customized model whose hyperparameters are tuned using HyperDrive. We then compare the performance of both the models and deploy the best performing model. We have used Patient Heart failure data to predict survival, source of data is from publicly available <a href="https://www.kaggle.com">Kaggle</a> dataset. We consume the deployed model using rest endpoint URL. 

Below is overall task flow of this project:

![Project Workflow](images/Capstone_Project.jpg?raw=true "Project Workflow") 

## Project Set Up and Installation
This is project used Azure Machine Learning studio provided by Udacity Lab. We have used following Jupyter notebook files to completes this project.

In order to run the project in Azure Machine Learning Studio, we will need the two Jupyter Notebooks files, dataset file and other supporting files to build models.

- `automl.ipynb`: AutoML model;
- `hyperparameter_tuning.ipynb`: HyperDrive model.
- `heart_failure_clinical_records_dataset.csv`: the dataset file.

- Additionally, Models are created under Python 3.6 SDK environment and xgboost with verion 1.3.3 was installed. I removed version 1.3.3 and installed version xgboost==0.90 version. This we need to get best run detail detail.

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.

Cardiovascular diseases (CVDs) are the number one of the most cause of death globally, taking an estimated 17.9 million lives each year.
Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.
Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.
People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

This dataset has been taken from kaggle site - https://www.kaggle.com/andrewmvd/heart-failure-clinical-data?select=heart_failure_clinical_records_dataset.csv

Below are (13) clinical features in this dataset:

- age: age of the patient (years)
- anaemia: decrease of red blood cells or hemoglobin (boolean)
- high blood pressure: if the patient has hypertension (boolean)
- creatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L)
- diabetes: if the patient has diabetes (boolean)
- ejection fraction: percentage of blood leaving the heart at each contraction (percentage)
- platelets: platelets in the blood (kiloplatelets/mL)
- sex: woman or man (binary)
- serum creatinine: level of serum creatinine in the blood (mg/dL)
- serum sodium: level of serum sodium in the blood (mEq/L)
- smoking: if the patient smokes or not (boolean)
- time: follow-up period (days)
- [target] death event: if the patient deceased during the follow-up period (boolean)

![Dataset details](images/Dataset_detail.png?raw=true "Dataset detail") 


### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
