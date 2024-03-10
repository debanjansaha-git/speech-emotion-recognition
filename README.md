# End-to-End MLOps Pipeline for Emotion Detection

In today's digital age, understanding human emotions from various sources like audio samples has become imperative for applications ranging from customer service to mental health support. In this project, we embark on a journey to develop an end-to-end Emotion Detection MLOps pipeline leveraging the power of Google Cloud Platform (GCP).


## Table of Contents

- [End-to-End MLOps Pipeline for Emotion Detection](#end-to-end-mlops-pipeline-for-emotion-detection)
  - [Table of Contents](#table-of-contents)
  - [Folder Structure](#folder-structure)
  - [Datasets](#datasets)
  - [Key Objectives](#key-objectives)
  - [Data Preprocessing](#data-preprocessing)
  - [Instructions](#instructions)
    - [Core ML Module](#core-ml-module)
    - [Airflow Pipeline](#airflow-pipeline)
    - [MLFlow Pipeline](#mlflow-pipeline)
  - [Tools \& Technologies](#tools--technologies)
  - [Contributions](#contributions)
  - [Expected Outcomes](#expected-outcomes)
  - [Conclusion](#conclusion)
  - [License](#license)
  - [Internal Notes by the Team](#internal-notes-by-the-team)


## Folder Structure 

The project files are organised as follows:
[Organisation](tree.txt)
  

## Datasets

Our dataset comprises a diverse collection of audio samples sourced from renowned databases like the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS), Toronto Emotional Speech Set (TESS), and augmented with custom audio data to enhance the model's robustness across various demographics and contexts.


## Key Objectives

1. **Data Preprocessing**: We will explore techniques to preprocess the audio data, including feature extraction and normalization, to prepare it for model training.

2. **Model Development**: Employing state-of-the-art deep learning architectures, we will design and train a robust Emotion Detection model capable of accurately categorizing emotional states from audio samples.

3. **Model Deployment**: Leveraging GCP's scalable infrastructure, we will deploy the trained model as a service, ensuring seamless integration with production environments and real-time inference capabilities.

4. **MLOps Integration**: Implementing best practices in MLOps, we will establish a streamlined workflow for model versioning, monitoring, and continuous integration/continuous deployment (CI/CD) to maintain model performance and reliability over time.


## Data Preprocessing



## Instructions

Below contains guidance on how to execute the modules.


### Core ML Module


  > [!WARNING]
  > All codes/commands have to be executed from the **dags** folder.
   - Move to the **dags** in project directory

  > [!CAUTION]
   - In order to execute this module you will need to *build* and *install* the module, otherwise you will get the error
  ```bash
  ModuleNotFoundError: No module named 'mlcore'
  ```

   - Start with installing the dependencies first.

   ```python
   pip install -r requirements.txt
   ```

   - Build the module binaries

   ```python
   python setup.py sdist
   ```

   - Install the build

   ```python
   python setup.py install
   ```

  > [!IMPORTANT]
  Every time you make some changes in the configurations, you will have to **build the module** manually, so that your changes are reflected. We will later automate this using GitHub Workflows / other CICD options.
That means you need to run the previous command from the root folder.


- In order to run any of the data scripts, use commands:
```python
python stage_03_data_transformation.py  
```


### Airflow Pipeline

- Move to the **pipeline** directory
- With the **pipeline** directory as your working directory, run the following command to start airflow in docker one after the other:

```bash
docker compose up airflow-init
```
This command initializes the database and services needed to start and run the airflow webserver, scheduler, etc.

```bash
docker compose up
```
This service starts the airflow services in various containers.
To open the Airflow UI, open the following link in your browser:
[http://0.0.0.0:8080/home](http://0.0.0.0:8080/home)


### MLFlow Pipeline

- Move to the **$HOME** directory
- Create your virtual environment and install all the dependencies

```python
pip install -r ./src/mlflow/requirements.txt
```

- Start MLFlow Web UI:

```bash
mlflow ui --port=5001
```
This will fire up the MLFlow UI, and open the following link in your browser:
[http://0.0.0.0:5001](http://0.0.0.0:5001)

- In order to log experiments using MLFlow execute the Trainer module from the **$HOME** directory:

```python
python ./src/mlflow/mlflow_trainer.py
```

## Tools & Technologies

**Google Cloud Platform (GCP)**: Compute Engine, Cloud Storage, BigQuery and Vertex AI Platform for scalable infrastructure, data storage, model training, and deployment. The entire project is planned to implemented in **GCP only** with Google Kubernetes Engine (GKE), however, the project is cloud agnostic, so it can be easily integrated into Microsoft Azure, AWS or others providers and platforms like MinIO.

**Airflow**: Orchestration of data preprocessing and transformation, notification pipelines.

**TensorFlow-Keras / PyTorch**: Utilized for developing and training deep learning models for Emotion Detection.

**MLflow**: Integrated for managing machine learning workflows and experimentation.

**DVC**: Integration of DVC for version controlling the data, and tracking the data provenance.

**Optuna**: Integration of Optuna for hyperparameter tuning and tracking via MLflow.

**Apache Beam**: [Planned] Will transfer our data preprocessing and transformation pipelines to Apache Beam for scalability.

OR

**KubeFlow**: [Planned] The entire pipeline will be transferred to KubeFlow for orchestration


**KServe**: [Planned] We plan to serve the inferencing pipeline using KServe (Kubernetes on GKE)


## Contributions

Create a fork of the repository and submit a pull request after adding your changes


## Expected Outcomes

- A production-ready Emotion Detection model capable of accurately categorizing emotional states from audio samples.

- A scalable MLOps pipeline on GCP for model training, deployment, and monitoring, ensuring robustness and reliability in real-world applications.


## Conclusion

Through this project, we aim to demonstrate the efficacy of leveraging GCP's infrastructure and MLOps practices to develop and deploy sophisticated Emotion Detection solutions, with potential applications spanning diverse domains such as mental health support, customer sentiment analysis, and human-computer interaction.

## License

This project is licensed under the [MIT License](LICENSE).


## Internal Notes by the Team

**1. How to contribute to the the pipeline (How to develop your own dags)**

The dags, logs, plugins and secrets folders are copied into the container and are updated in real-time. Any changes you make to the files inside these folders will reflect inside the container.

So by modifying the code inside the dags folder, we can add and develop varous dags. For instance, we can add a new dag which can download the data and preprocess it inside the dags folder and it will automatically be picked up airflow's services.

