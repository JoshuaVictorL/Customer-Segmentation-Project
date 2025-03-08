# Customer-Segmentation-Project

## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the app.py
10. Update the dvc.yaml



# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/JoshuaVictorL/Customer-Segmentation-Project
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n mlproj python=3.10 -y
```

```bash
conda activate mlproj
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```

## DVC

1. dvc init
2. dvc repro
3. dvc dag



## MLflow

[Documentation](https://mlflow.org/docs/latest/index.html)


##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/JoshuaVictorL/Customer-Segmentation-Project.mlflow \
MLFLOW_TRACKING_USERNAME=JoshuaVictorL \
MLFLOW_TRACKING_PASSWORD=ba3573d2f61c38b6a9def21419a4bc92446acfbb \
python main.py

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/JoshuaVictorL/Customer-Segmentation-Project.mlflowmlflow

export MLFLOW_TRACKING_USERNAME=JoshuaVictorL 

export MLFLOW_TRACKING_PASSWORD=ba3573d2f61c38b6a9def21419a4bc92446acfbb

```