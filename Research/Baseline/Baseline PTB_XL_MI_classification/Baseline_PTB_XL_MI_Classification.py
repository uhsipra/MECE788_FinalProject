import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np
import os

# Prepare directory for outputs
output_dir = "PTB_XL_MI_classification"
os.makedirs(output_dir, exist_ok=True)
# Load training dataset with specific sheet name
train_df = pd.read_csv("E:\Esraa\ML project\Test.csv")

# Load testing dataset with specific sheet name
test_df = pd.read_csv("E:\Esraa\ML project\Train.csv")

# Extract features and target variable from training dataset
X_train = train_df.iloc[:, 0:193]

# Extract features and target variable from testing dataset
X_test = test_df.iloc[:, 0:193]  # Include all columns from 1 to the end

# Apply the function to classify the target variable
y_train = train_df['MI']
y_test = test_df['MI']
# Define the predefined classes
predefined_classes = ['NORM', 'MI', 'HYP', 'CD', 'STTC']
class_labels = predefined_classes

# Define classifiers
classifiers = {
    'SVM': SVC(probability=True),
    'kNN': KNeighborsClassifier(),
    'XGBoost': XGBClassifier()
}

# Define parameter grids for grid search
param_grids = {
    'SVM': {'C': [1, 10, 100], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf']},
    'kNN': {'n_neighbors': [3, 5], 'metric': ['euclidean', 'manhattan']},
    'XGBoost': {'n_estimators': [100, 200, 300], 'max_depth': [3, 4, 5], 'learning_rate': [0.1, .01, 0.001]}
}

# Initialize empty dictionary to store best models
best_models = {}
for clf_name, clf in classifiers.items():
    model_path = os.path.join(output_dir, f'MI_best_model_{clf_name}.pkl')
    if os.path.exists(model_path):
        print(f"Loading existing model for {clf_name}...")
        best_models[clf_name] = joblib.load(model_path)
    else:
        print(f"Performing Grid Search for {clf_name}...")
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grids[clf_name], cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_models[clf_name] = grid_search.best_estimator_
        joblib.dump(best_models[clf_name], model_path)
        print(f"Best parameters for {clf_name}: {grid_search.best_params_}")

# Convert target variable to binary format (necessary for ROC-AUC calculation)
num_classes = len(np.unique(y_train))
y_train_bin = label_binarize(y_train, classes=np.arange(num_classes))
y_test_bin = label_binarize(y_test, classes=np.arange(num_classes))

# Evaluate each best model
for clf_name, best_model in best_models.items():
    y_pred = best_model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred, normalize='true') * 100  # Convert to percentage
    class_report = classification_report(y_test, y_pred, zero_division=1)

    # Print and save confusion matrix
    print(f"MI _ {clf_name}:\n", conf_matrix)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='.2f')
    plt.title(f'MI Matrix for {clf_name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(os.path.join(output_dir, f'MI_confusion_matrix_{clf_name}.png'))  # Save confusion matrix plot
    plt.close()

    # Print classification report
    print(f"\nClassification Report for {clf_name}:\n", class_report)
