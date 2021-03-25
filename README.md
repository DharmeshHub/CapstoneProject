# Capstone Project - Using Machine Learning to Predict Survival of Patients with Heart Failure

In this project, we have created two models: one using Automated ML (AutoML) and one customized model whose hyperparameters are tuned using HyperDrive. We then compare the performance of both the models and deploy the best performing model. We have used Patient Heart failure data to predict survival, source of data is from publicly available <a href="https://www.kaggle.com">Kaggle</a> dataset. We consume the deployed model using rest endpoint URL. 

Below is overall task flow of this project:

![Project Workflow](images/Capstone_Project.jpg?raw=true "Project Workflow") 

## Project Set Up and Installation
This is project used Azure Machine Learning studio provided by Udacity Lab. We have used following Jupyter notebook files to completes this project.

In order to run the project in Azure Machine Learning Studio, we will need the two Jupyter Notebooks files, dataset file and other supporting files to build models.

- `automl.ipynb`: AutoML model;
- `hyperparameter_tuning.ipynb`: HyperDrive model.
- `heart_failure_clinical_records_dataset.csv`: the dataset file.

- Additionally, Models are created under Python 3.6 SDK environment and xgboost with verion 1.3.3 was installed. I removed version 1.3.3 and installed version xgboost==0.90 version. This we need to get best run detail.

## Dataset

### Overview

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
The model we build using Heart failure dataset classify whether patient have chance of survival or not. Model use above list features and result in target column (Death event) with value 0 (no) or 1 (yes).


### Access

I downloaded dataset file (.csv) from Kaggle site and added to my GitHub respository. I access dataset directly from my GitHub repository and registered the dataset into workspace. I have also added dataset file in my Jupyter notebook directory to access data during training (train.py)

***Create Dataset:*** _URI source_

![Create Dataset Registered](images/CreateDataset.png?raw=true "Create Dataset") 

***Registered datasets:*** _heart-failure-prediction_

![Dataset Registered](images/RegisteredDataset.png?raw=true "Dataset Registered") 

***Registered datasets Source:*** _URI_

![Dataset URI](images/RegisteredDatasetURI.png?raw=true "Dataset URI") 

 
## Automated ML

Below is the AutoML setting and configuration for this project.

![AutoML Configuration](images/AutoMLConfiguration.png?raw=true "AutoML Configuration") 

_experiment_timeout_minutes=20_

This is an exit criterion and is used to define how long (in minutes), the experiment should continue to run. To help avoid experiment time out failures, I used the minimum of 20 minutes.

_max_concurrent_iterations_: 4

It represents the maximum number of iterations that would be executed in parallel.

_task='classification'_

This defines the experiment type which in this case is classification.

_primary_metric='accuracy'_

I chose accuracy as the primary metric for this classification model.

_n_cross_validations=2_

This parameter sets how many cross validations to perform, based on the same number of folds (subsets). Two folds for cross-validation are defined. So, two different trainings, each training using 1/2 of the data, and each validation using 1/2 of the data with a different holdout fold each time.

_enable_early_stopping=True_

Early stopping helps in performance, it terminates poor performing run and fully run good performing run.

_featurization=auto_

Featurization is done automatically, i.e. normalization technique are applied to your data. This help certain algorithms that are sensitive to features on different scales.

_lable_column_name = "DEATH_EVENT"
The name of the label (target) column. This parameter is applicable to training_data and validation_data parameters.

### Results

Below is the best model run for AutoML model.
run_id = AutoML_c6507e5f-cd46-4860-a977-7732236037f7_16
Accuracy - 0.852841163310962
Algorithm - VotingEnsemble


Following are screenshot of AutoML experiment run.

***AutoML Completed:***

![AutoML Completed](images/AutoML_Exp_Completed.png?raw=true "AutoML Completed")

***Data Guardrials:***

![Data Guardrials](images/DataGuardrials.png?raw=true "Data Guardrials") 

***AutoML Best Model:***

![AutoML Best Model](images/AutoML_Exp_BestModel1.png?raw=true "AutoML Best Model") 

***AutoML RunDetails Widget:***

![AutoML RunDetails](images/AutoML_Exp_Completed_SDK.png?raw=true "AutoML RunDetails") 

***Best model top features:***

![Best Model features](images/BestModel_Topfeatures.png?raw=true "Best Model features")

***Best model metrics:***

![Best Model metrics](images/AutoML_BestModel_Metrics.png?raw=true "Best Model metrics")

***AutoML runs:***

![AutoML Runs](images/Auto_RunDetails.png?raw=true "AutoML Runs") 

***Best model run details:***

![Best Runs](images/BestRunDetails.png?raw=true "Best Run")

![Best Runs2](images/BestRunDetails1.png?raw=true "Best Run1") 

![Best Runs3](images/BestRunDetails2.png?raw=true "Best Run2")

***Best model application insights:***

![Best Runs3](images/BestModel_App_Insights.png?raw=true "Best Run2")


## Hyperparameter Tuning

For Hyperparameter tuning model I choose Scikit-learn Logistic Regression model. 
Sampling method chosen is random sampling as parameters, it supports discrete and continuous hyperparameters. It supports early termination of low-performance runs. It is computationally less expensive as it takes subset of combinations and it's faster unlike GridParameterSampling. Some users do an initial search with random sampling and then refine the search space to improve results. In random sampling, hyperparameter values are randomly selected from the defined search space. You can also specify the maximum number of runs that you want the module to execute. This option is useful when you want to increase model performance by using the metrics of your choice but still conserve computing resources. GridParameterSampling utilize more resources compare to RandomParameterSampling. 

Here I choose discrete values with _choice_ for both parameters, _C_ and _max_iter_. _C_ is the Regularization and _max_iter_ is the maximum number of iterations. 
This option trains a model by using a set number of iterations. You specify a range of values to iterate over, and the module uses a randomly chosen subset of those values. Values are chosen with replacement, meaning that numbers previously chosen at random are not removed from the pool of available numbers. So the chance of any value being selected stays the same across all passes.

For more aggressive savings, used Bandit Policy with a smaller allowable slack or Truncation Selection Policy with a larger truncation percentage. Any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated. This means that with this policy, the best performing runs will execute until they finish.

![Hyperparameter](images/HD_Parameters.png?raw=true "Hyperparameter")

_evaluation_interval_: (optional) the frequency for applying the policy. Each time the training script logs the primary metric counts as one interval.

_slack_factor_: The amount of slack allowed with respect to the best performing training run. This factor specifies the slack as a ratio.

_delay_evaluation: (optional) delays the first policy evaluation for a specified number of intervals.


### Results

***HyperDrive run Completed:***

![Hyperdrive Completed](images/HD_Exp_Completed_SDK.png?raw=true "Hyperdrive Completed")

![Hyperdrive Completed](images/HD_Exp_Completed.png?raw=true "Hyperdrive Completed")

![Hyperdrive Child runs](images/HD_Childrun_SDK.png?raw=true "Hyperdrive Child runs")

***HyperDrive Best run:***

![Hyperdrive Bestrun](images/HD_BestRun.png?raw=true "Hyperdrive Bestrun")

![Hyperdrive Bestrun](images/HD_Bestrun_SDK.png?raw=true "Hyperdrive Bestrun")

![Hyperdrive Bestrun](images/HD_Bestrun_SDK1.png?raw=true "Hyperdrive Bestrun")

![Hyperdrive childruns](images/HD_Childrun_SDK.png?raw=true "Hyperdrive childruns")

## Model Deployment

Below are both model results. The difference is the accuracy as above between two models. AutoML runs data agains multiple alogorithm so we get best model with higher accuracy compare to Hyperdrive model where we are working with one model. In every AutoML experiment, automatic scaling and normalization techniques are applied to your data by default. These techniques are types of featurization that help certain algorithms that are sensitive to features on different scales. You can enable more featurization, such as missing-values imputation, encoding, and transforms, in case of HyperDrive model we have to tune model with multiple runs.

<b>AutoML Model:</b></br>
run_id = AutoML_c6507e5f-cd46-4860-a977-7732236037f7_16</br>
Accuracy - 0.852841163310962</br>
Algorithm - VotingEnsemble</br>

<b>HyperDrive Model:</b></br>
run_id - HD_ce211f3b-14b4-486a-afa9-7daf5673efba_0</br>
Accuracy - 0.8166666666666667</br>
Parameter sampling - Random</br>
Termination Policy - BANDIT</br>

***Deployment steps of model:***

Register the model
Prepare an inference configuration
Prepare an entry script
Choose a compute target.
Deploy the model to the compute target.
Test the resulting web service.


***Registered AutoML model:***

![Registered AutoML](images/AutoML_model_download.png?raw=true "Registered AutoML")

![Registered AutoML](images/AutoML_Model_Registered.png?raw=true "Registered AutoML")

***Deployed AutoML model:***

![Deployed AutoML](images/AutoML_Deployed_SDK.png?raw=true "Deployed AutoML")
![Deployed AutoML](images/AutoML_Deployed_Model_SDK.png?raw=true "Deployed AutoML")

- Inference configuration: The inference configuration describes how to configure the model to make predictions. It references your scoring script (entry_script) and is used to locate all the resources required for the deployment. Inference configurations use Azure Machine Learning environments to define the software dependencies needed for your deployment.

- Entry script (scoring_file_v_1_0_0.py): loads the trained model, processes input data from requests, does real-time inferences, and returns the result. The designer automatically generates a scoring.py entry script file when the Train Model module completes.

- Compute Target: As compute target, I chose the Azure Container Instances (ACI) service, which is used for low-scale CPU-based workloads that require less than 48 GB of RAM. The AciWebservice Class represents a machine learning model deployed as a web service endpoint on Azure Container Instances. The deployed service is created from the model, script, and associated files, as I explain above. The resulting web service is a load-balanced, HTTP endpoint with a REST API. We can send data to this API and receive the prediction returned by the model.

***Service is healthy and the endpoint is available:***

![Deployed AutoML](images/AutoML_Endpoint_details.png?raw=true "Deployed AutoML")

***Consume End point:***
Created the endpoint.py script to interact with the trained model. scoring_uri is REST endpoint point URL and Key is the authentication key generated while deploying the model using Enable Authentication.

![Consume End point](images/AutoML_BestModel_Consume.png?raw=true "Consume End point")

![Run End point](images/AutoML_BestModel_Consume_SDK.png?raw=true "Run End point")

![End point test](images/Endpoint_Test.png?raw=true "Test End point")

***[Optional] Swagger Documentation:***

Swagger is a set of open-source tools built around the OpenAPI Specification that can help you design, build, document and consume REST APIs. We have used Swagger UI tool to generate documentation to interact with API. Azure provide swagger.json file URL for deployed models. We downloaded this file in swagger folder.

	- Created swagger.sh script downloaded the latest swagger container and run swagger UI. Swagger is running on local host at port 9000.
	- Created serve.py started the Python server on port 8000.

![Swagger](images/swagger.png?raw=true "Swagger")

***Service deleted:***

![Service delete](images/ServiceDeleted.png?raw=true "Delete Service")


## Screen Recording

The screen recording can be found [here](https://youtu.be/6ch4_8pHnuA) and it shows the project in action.

More specifically, the screencast demonstrates:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Comments and Future Improvements
- Need to explore more on Hyperdrive parameter and want to achieve higher accurarcy compare to AutoML mode, but with low volume dataset and time constraint it can't be achived here.
- More on time limit, I saw some other dataset on kaggle which required lots of encoding and cleaning so finally used what is suggested in project video. Tried one big dataset but accuracy was very low for all models and timing out.
- We can explore more on using customize featurization settings to ensure that the data and features that are used to train your ML model result in relevant predictions.
- For HyperDrive model, a range of different optimization algorithms can be used, like grid search as we did random search. Define a new model and set the hyperparameter values of the model to the values found by different search. Then fit the model on all available data and use the model to start making predictions on new data.

