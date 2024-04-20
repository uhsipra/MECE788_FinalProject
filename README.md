# ECG Signal Classification Project

## Overview
This repository contains code for an ECG signal classification project using the MIT-BIH and PTB-XL datasets. The goal of this project is to develop machine learning models to classify ECG signals into different categories.

## Dataset
- **MIT-BIH Dataset**: [Link to dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
- **PTB-XL Dataset**: [Link to dataset](https://physionet.org/content/ptb-xl/1.0.3)

## Repository Structure
- **Best Model**: This folder contains the training and evaluation code for the best performing model achieved in the project.
  - **Best_Model_Training.ipynb**: Notebook file for training the best model.
  - **Best_Model_Testing.ipynb**: Notebook file for testing the best model.
  
- **Research**: This folder contains all file related to the data preprocessing and various models tested in the project.
  - **Preprocessing**: Code for preprocessing the PTB-XL dataset. The MIT dataset comes preprocessed by default via kaggle.
  - **Baseline**: Code for baseline models including KNN, SVM, and XGBoost.
  - **MPA**: Code for models based on the Marine Predators algorithm optimization including KNN, SVM, and Random Forest.
  - **Deep Learning**: Code for deep learning model used to train and test MIT dataset.

- **Model Card**: This folder contains the model card for the best performing model achieved in the project.  

## Usage
1. Navigate to the `Best Model` directory:

2. Open and run the Jupyter Notebook files to train and evaluate the model.

## Results
- **Best Model Performance**: Insert performance metrics (e.g., accuracy, precision, recall, F1-score) achieved by the best model.

## License
This project is licensed under the [MIT License](LICENSE).

## Contact
For any questions or inquiries, please contact [Your Name](mailto:your_email@example.com).
