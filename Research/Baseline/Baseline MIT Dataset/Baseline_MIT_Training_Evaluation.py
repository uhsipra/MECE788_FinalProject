import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import  GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score,auc,roc_curve
import joblib
import numpy as np
import os
import matplotlib

os.environ["OPENBLAS_NUM_THREADS"] = "0"

# Use non-interactive backend for matplotlib
matplotlib.use('Agg')

# Create a directory for saving results best model,graphs and metrix
output_dir = "MIT classification"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load training and testing datasets
train_df = pd.read_csv("E:/Esraa/ML project/mitbih_train_new.csv")
test_df = pd.read_csv("E:/Esraa/ML project/mitbih_test_new.csv")

# Define features and target
X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]
X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

# Convert target variable to binary format (necessary for ROC-AUC calculation)
num_classes = len(np.unique(y_train))
y_train_bin = label_binarize(y_train, classes=np.arange(num_classes))
y_test_bin = label_binarize(y_test, classes=np.arange(num_classes))

# Define classifiers and their hyperparameters
classifiers = {
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(eval_metric='mlogloss')
}

param_grids = {
    'SVM': {'C': [1, 10, 100], 'gamma': [0.001, 0.01, 0.1], 'kernel': ['linear', 'rbf']},
    'KNN': {'n_neighbors': [3, 5], 'metric': ['euclidean', 'manhattan']},
    'XGBoost': {'n_estimators': [100, 200, 300], 'max_depth': [3, 4, 5], 'learning_rate': [0.1, .01, 0.001]},
  }

# Perform grid search and evaluate models
best_models = {}
for clf_name, clf in classifiers.items():
    model_path = os.path.join(output_dir, f'best_model_{clf_name}.pkl')
    #check if best model is already exist in the file bath
    if os.path.exists(model_path):
        print(f"Loading existing model for {clf_name}...")
        best_models[clf_name] = joblib.load(model_path)
    else:
        # perform grid search for each model with parameter tunning searching for the best the model
        print(f"Performing Grid Search for {clf_name}...")
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grids[clf_name], cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_models[clf_name] = grid_search.best_estimator_
        joblib.dump(best_models[clf_name], model_path)
        print(f"Best parameters for {clf_name}: {grid_search.best_params_}")

# Evaluation and visualization of each model
for clf_name, model in best_models.items():
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    else:
        y_score = model.decision_function(X_test)
    #generate a confusion matrix for the predection classes
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Normalize the confusion matrix to percentages
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
    y_proba = best_models[clf_name].predict_proba(X_test) if hasattr(best_models[clf_name], 'predict_proba') else \
    best_models[clf_name].decision_function(X_test)

    class_report = classification_report(y_test, y_pred, zero_division=1)
    roc_auc_ovo = roc_auc_score(y_test, y_proba, multi_class='ovo')
    roc_auc_ovr = roc_auc_score(y_test, y_proba, multi_class='ovr')
    report_content = f"MIT multi class classification report for {clf_name}\n"
    report_content += f"ROC-AUC-OVO: {roc_auc_ovo:.2f}\n"
    report_content += f"ROC-AUC-OVR: {roc_auc_ovr:.2f}\n\n"
    report_content += class_report

    # Create an image of the classification report
    fig, ax = plt.subplots(figsize=(12, 4))  # Adjust the size according to your text
    ax.text(0.5, 0.5, report_content, fontsize=12, va='center', ha='center')
    ax.axis('off')
    plt.savefig(os.path.join(output_dir, f'MIT multi_class_classification_report_{clf_name}.png'))
    plt.close()

    # Define custom labels for the classes
    class_labels = ['Q', 'F', 'V', 'S', 'N']

    # Plot and save normalized confusion matrix with custom labels
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt=".1f", cmap='Blues', xticklabels=class_labels,
                yticklabels=class_labels)
    plt.title(f'MIT multi class classification Confusion Matrix for {clf_name} (Percentage)')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{clf_name}_percentage.png'))
    plt.close()

    # ROC-AUC calculation for multi-class if applicable
    if num_classes > 2:
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curve for a specific class
        plt.figure()
        for i in range(num_classes):
            plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {class_labels[i]} (area = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('MIT multi class classification ROC Curves for {}'.format(clf_name))
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, f'roc_curve_{clf_name}.png'))
        plt.close()
