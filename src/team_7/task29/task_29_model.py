# %%
# Team 7 - Task 29

# %%
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from IPython.display import display, HTML

# %%
# Ignore all warnings
warnings.filterwarnings("ignore")

# %%
# Function to load and preprocess the dataset
def load_and_preprocess(file_path):
    df_fixed = pd.read_excel(file_path)
    medication_columns = ['1183', '2188', '630', '2791', '6737', '2624', '5913', 
                          '2606', '1443', '4437', '6718', '4328', '2043', '6720', 
                          '37', '3381', '643', '3459', '577', '4677']
    return df_fixed, medication_columns

# %%
# Function to encode categorical target
def encode_target(df, target):
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])
    return df, le.classes_

# %%
# Function to prepare data for training and testing
def prepare_data(df, features, target):
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Function to train and evaluate models
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy, f1, report, y_pred

# %%
# Function to run GridSearchCV with hyperparameter tuning
def run_grid_search(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# %%
# Function to compare multiple models
def compare_models(df, features, target, models_params):
    X_train, X_test, y_train, y_test = prepare_data(df, features, target)
    results = []
    best_model = None
    best_y_pred = None
    best_accuracy = 0
    best_model_name = None
    
    for name, (model, param_grid) in models_params.items():
        if param_grid:
            model = run_grid_search(model, param_grid, X_train, y_train)
        accuracy, f1, report, y_pred = train_and_evaluate(model, X_train, X_test, y_train, y_test)
        
        # Track the best model based on accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_y_pred = y_pred
            best_model_name = name
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'F1 Score': f1,
            'Classification Report': report
        })
    
    return pd.DataFrame(results), best_model, X_test, y_test, best_y_pred, best_model_name

# %%
# Function to visualize results as bar plots
def visualize_results(results, title):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    results.plot(kind='bar', x='Model', y='Accuracy', ax=ax[0], color='skyblue', legend=False)
    ax[0].set_title(f'Accuracy Comparison - {title}')
    ax[0].set_xlabel('Model')
    ax[0].set_ylabel('Accuracy')
    
    results.plot(kind='bar', x='Model', y='F1 Score', ax=ax[1], color='lightgreen', legend=False)
    ax[1].set_title(f'F1 Score Comparison - {title}')
    ax[1].set_xlabel('Model')
    ax[1].set_ylabel('F1 Score')
    
    plt.tight_layout()
    plt.show()

# %%
# Function to display the results as an HTML table
def display_html_table(results):
    display(HTML(results[['Model', 'Accuracy', 'F1 Score']].to_html(index=False)))

# %%
# Function to plot confusion matrix
def plot_confusion_matrix(best_model_name, X_test, y_test, y_pred, title, classes):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {title} ({best_model_name})')
    plt.xlabel('Predicted Category')
    plt.ylabel('True Category')
    plt.show()

# %%
# Define models and hyperparameters for GridSearchCV
def define_models_params():
    return {
        'Logistic Regression': (
            LogisticRegression(max_iter=1000),
            {'C': [0.01, 0.1, 1, 10, 100]}
        ),
        'Decision Tree': (
            DecisionTreeClassifier(),
            {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}
        ),
        'Random Forest': (
            RandomForestClassifier(),
            {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'bootstrap': [True, False]}
        ),
        'Gradient Boosting': (
            GradientBoostingClassifier(),
            {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 0.2]}
        ),
        'Support Vector Machine': (
            SVC(),
            {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        ),
        'XGBoost': (
            xgb.XGBClassifier(),
            {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 0.2]}
        ),
        'KNN': (
            KNeighborsClassifier(),
            {'n_neighbors': [3, 10, 15, 40], 'weights': ['uniform', 'distance']}
        ),
        'Neural Network': (
            MLPClassifier(max_iter=1000),
            {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['tanh', 'relu'], 'solver': ['adam', 'sgd'], 'learning_rate': ['constant', 'adaptive']}
        )
    }

# %%
# Main function to run the entire workflow
def run_workflow(file_path):
    df_fixed, medication_columns = load_and_preprocess(file_path)
    
    # Encode the target variables for each duration category
    df_fixed, classes_16 = encode_target(df_fixed, 'Duration_Category_16')
    df_fixed, classes_17 = encode_target(df_fixed, 'Duration_Category_17')
    df_fixed, classes_18 = encode_target(df_fixed, 'Duration_Category_18')
    
    models_params = define_models_params()
    
    # Run comparison for Duration_Category_16
    results_16, best_model_16, X_test_16, y_test_16, y_pred_16, best_model_name_16 = compare_models(df_fixed, medication_columns, 'Duration_Category_16', models_params)
    visualize_results(results_16[['Model', 'Accuracy', 'F1 Score']], 'Duration_Category_16')
    display_html_table(results_16)
    plot_confusion_matrix(best_model_name_16, X_test_16, y_test_16, y_pred_16, 'Duration_Category_16', classes_16)
    
    # Run comparison for Duration_Category_17
    results_17, best_model_17, X_test_17, y_test_17, y_pred_17, best_model_name_17 = compare_models(df_fixed, medication_columns, 'Duration_Category_17', models_params)
    visualize_results(results_17[['Model', 'Accuracy', 'F1 Score']], 'Duration_Category_17')
    display_html_table(results_17)
    plot_confusion_matrix(best_model_name_17, X_test_17, y_test_17, y_pred_17, 'Duration_Category_17', classes_17)
    
    # Run comparison for Duration_Category_18
    results_18, best_model_18, X_test_18, y_test_18, y_pred_18, best_model_name_18 = compare_models(df_fixed, medication_columns, 'Duration_Category_18', models_params)
    visualize_results(results_18[['Model', 'Accuracy', 'F1 Score']], 'Duration_Category_18')
    display_html_table(results_18)
    plot_confusion_matrix(best_model_name_18, X_test_18, y_test_18, y_pred_18, 'Duration_Category_18', classes_18)

# Example usage:
if __name__ == "__main__":
    file_path = 'task_29_dataset_final.xlsx'
    run_workflow(file_path)

# %%
