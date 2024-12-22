import os
import sys
import dill
from src.exception import CustomException
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import joblib

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(e, sys)

def plot_conf_mat(y_true, y_pred, labels):
    conf_mat = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()  # Ensure everything fits without overlapping
    plt.show()  # Use plt.show() to render the plot in interactive environments
    # Remove plt.close() to avoid closing the plot prematurely
    return plt

def plot_classification_report(y_true, y_pred, labels):
    class_report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    df_class_report = pd.DataFrame(class_report).iloc[:-1, :].T  # Convert to DataFrame and transpose
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_class_report, annot=True, cmap='Blues')
    plt.title('Classification Report')
    plt.tight_layout()  # Ensure everything fits without overlapping
    plt.show()  # Use plt.show() to render the plot in interactive environments
    # Remove plt.close() to avoid closing the plot prematurely
    return plt

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluate Train and Test dataset
            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)