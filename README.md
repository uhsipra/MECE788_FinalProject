# ECG Signal Classification Project

## Overview
This repository contains code for an ECG signal classification project using the MIT-BIH and PTB-XL datasets. The goal of this project is to develop machine learning models to classify ECG signals into different categories.

## Dataset
- **MIT-BIH Dataset**: [Link to dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
- **PTB-XL Dataset**: [Link to dataset](https://physionet.org/content/ptb-xl/1.0.3)

## Repository Structure
- **Best Model**: This folder contains the training and evaluation code for the best performing model achieved in the project.
  - **Train_finetuned_per_label.ipynb**: Notebook file for training the best model.
  - **Evaluate_finetuned_all_label.ipynb**: Notebook file for testing the best model.
  - **Preprocessed PTB-XL dataset**: We have added a link to the [preprocessed PTB-XL dataset](https://drive.google.com/drive/folders/1gspRgYy7IqPXUGf9qrZv3zyfNWuYgk6y?usp=share_link) used in training and testing the best model. 
  
- **Research**: This folder contains all file related to the data preprocessing and various models tested in the project.
  - **Preprocessing**: Code for preprocessing the PTB-XL dataset. The MIT dataset comes preprocessed by default via kaggle.
  - **Baseline**: Code for baseline models including KNN, SVM, and XGBoost.
  - **MPA**: Code for models based on the Marine Predators algorithm optimization including KNN, SVM, and Random Forest. MPA-kNN, MPA-RF, and MPA-SVM. Each model has                been tested on two distinct datasets, the MIT-BIH Arrhythmia Database and the PTB-XL Electrocardiography Database, resulting in six individual Jupyter                  Notebook (.ipynb) files.

    - The structure of each notebook is consistent and divided into nine key sections:
      * CUDA Environment Setup: This initial cell ensures that the CUDA environment is properly configured for GPU acceleration, essential for running the                      machine learning models efficiently.
      * Logging: The second cell sets up logging, capturing the workflow's detailed performance and outputting it to log files for subsequent analysis and                               record-keeping.
      * Parameter Definition: Here, we define the necessary parameters that set the boundaries for the hyperparameter optimization process, crucial for the                     success of the following Machine Learning Particle Swarm Optimization (MPA) algorithm.
      * Data Preparation: The fourth cell deals with loading the training dataset. It includes resampling to address class imbalance issues and splitting the                   data into training and validation sets.
      * Utility Functions: This section includes functions for Levy Flight (used in optimization), initialization routines, fitness evaluation, and plotting                    confusion matrices. These functions are key components that support the optimization process.
      * MPA Function Definition: The core MPA function is defined here. This function outlines various optimization scenarios that the MPA algorithm will                       execute to fine-tune the model's hyperparameters.
      * MPA Function Execution: In this cell, we invoke the MPA function to begin the hyperparameter optimization process. The MPA seeks to find the best                       hyperparameter settings for the machine learning models.
      * Model Evaluation: Finally, we load the test dataset and evaluate the trained model against it. This cell is crucial for understanding the model's                       performance on unseen data and includes various metrics for a thorough evaluation.

    To run these notebooks successfully, one must only modify the directory paths for loading the training and test datasets. Once the paths are set to the                 correct locations where the datasets are stored, the notebooks can be executed without any additional modifications.
              Each notebook is crafted to provide a seamless experience, from setting up the computing environment to the final assessment of the models' performance.
  - **Deep Learning**: Code for deep learning model used to train and test MIT dataset.

- **Model Card**: This folder contains the model card for the best performing model achieved in the project.  

## Usage
1. Navigate to the `Best Model` directory:

2. Download the preprocessed dataset via the following [link](https://drive.google.com/drive/folders/1gspRgYy7IqPXUGf9qrZv3zyfNWuYgk6y?usp=share_link)

3. Open and run the Jupyter Notebook files to train and evaluate the model.

4. For quick testing, we have saved the best models for each label [here](https://drive.google.com/drive/folders/19qW2sTTGEgtGelORTO6E72BxPzmWY3uS?usp=share_link).


## License
This project is licensed under the [MIT License](LICENSE).
