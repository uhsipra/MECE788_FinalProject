import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, auc, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import joblib
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np
import os
from PIL import Image
import io

matplotlib.use('Agg')

# Load datasets
# Load training dataset with specific sheet name
train_df = pd.read_csv("E:\Esraa\ML project\Train_new.csv")

# Load testing dataset with specific sheet name
test_df = pd.read_csv("E:\Esraa\ML project\Test_new.csv")

# Extract features and target variable from training dataset
X_train = train_df.iloc[:, 0:414]

# Extract features and target variable from testing dataset
X_test = test_df.iloc[:, 0:414]  # Include all columns from 1 to the end

# Define the predefined classes
predefined_classes = ['NORM', 'MI', 'HYP', 'CD', 'STTC']
class_labels = predefined_classes


# Function to classify target variable based on predefined classes
def classify_target_class(target):
    for idx, class_label in enumerate(predefined_classes):
        if class_label in target:
            return idx  # Return the index of the class label
    # If none of the predefined classes match, return a default label
    return len(predefined_classes)  # Assign a default label


# Apply the function to classify the target variable
y_train = train_df['diagnostic_superclass'].apply(classify_target_class)
y_test = test_df['diagnostic_superclass'].apply(classify_target_class)

# Define classifiers and their parameters for grid search
classifiers = {
    'SVM': SVC(probability=True),
    'kNN': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

param_grids = {
    'SVM': {'C': [1, 10, 100], 'gamma': [0.001, 0.01, 0.1], 'kernel': ['linear', 'rbf']},
    'KNN': {'n_neighbors': [3, 5], 'metric': ['euclidean', 'manhattan']},
    'XGBoost': {'n_estimators': [100, 200, 300], 'max_depth': [3, 4, 5], 'learning_rate': [0.1, .01, 0.001]},
  }

# Prepare directory for outputs
output_dir = "PTB_XL_classification"
os.makedirs(output_dir, exist_ok=True)

# Perform grid search and evaluate models
best_models = {}
for clf_name, clf in classifiers.items():
    model_path = os.path.join(output_dir, f'best_model_{clf_name}.pkl')
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
    y_proba = best_model.predict_proba(X_test)  # Ensure model can predict probabilities
    y_test_binarized = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
    # define the confution matrix
    conf_matrix = confusion_matrix(y_test, y_pred, normalize='true')
    # define the classification report
    #class_report = classification_report(y_test, y_pred, target_names=class_labels, zero_division=1)
    class_report = classification_report(y_test, y_pred, zero_division=1)
    roc_auc_ovo = roc_auc_score(y_test, y_proba, multi_class='ovo')
    roc_auc_ovr = roc_auc_score(y_test, y_proba, multi_class='ovr')
    report_content = f"PTB_xl multi class classification report for {clf_name}\n"
    report_content += f"ROC-AUC-OVO: {roc_auc_ovo:.2f}\n"
    report_content += f"ROC-AUC-OVR: {roc_auc_ovr:.2f}\n\n"
    report_content += class_report
    print(f"\nClassification Report for {clf_name}:\n", report_content)
    # Save classification report as an image
    # Create an image of the classification report
    fig, ax = plt.subplots(figsize=(12, 4))  # Adjust the size according to your text
    ax.text(0.5, 0.5, class_report, fontsize=12, va='center', ha='center')
    ax.axis('off')
    plt.savefig(os.path.join(output_dir, f'PTB_xl multi_class_classification_report_{clf_name}.png'))
    plt.close()

    # Plot and save normalized confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='.2%', xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'PTB_xl Confusion Matrix for {clf_name} (Percentages)')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{clf_name}_percentages.png'))
    plt.close()

    # Plot ROC-AUC for all classes
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{label} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'PTB_xl ROC Curve for {clf_name}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, f'roc_curve_{clf_name}.png'))
    plt.close()

