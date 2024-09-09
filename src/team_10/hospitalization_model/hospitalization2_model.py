# ====================== Hospitalization2 Model ======================

# =========== Imports ===========
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import classification_report, confusion_matrix

from src.team_10.hospitalization_model.NNHospitalizationPredictor import NNHospitalizationPredictor


# ========== Path Config ==========
load_path = '../../data/hospitalization2_Team_10.csv'


# =========== Functions ===========
def add_diagnosis_to_cats(cats, diag):
    if ',' in diag:
        split_diag = diag.split(',')
        for value in split_diag:
            value = value.strip()
            if value not in cats and value != '':
                cats.append(value)
    elif diag.strip() not in cats and diag.strip() != '':
        cats.append(diag.strip())


def get_diag_categories(df_cp, column_name):
    cats = []
    for i in range(len(df_cp[column_name])):
        if df_cp[column_name][i] != -1:
            diag = str(df_cp[column_name][i])
            add_diagnosis_to_cats(cats, diag)
    return cats


def create_count_dict_for_cat_list(cats):
    count_dict = {}
    count_dict = dict(zip(cats, [0] * len(cats)))
    return count_dict


def add_value_to_cat_dict(diag, cats, cat_dict_cp):
    if ',' in diag:
        split_diag = diag.split(',')
        for value in split_diag:
            value = value.strip()
            if value in cats and value != '':
                cat_dict_cp[value] += 1
    elif diag.strip() in cats and diag.strip() != '':
        cat_dict_cp[diag.strip()] += 1


def get_diag_counts_for_each_category(df_cp, cats, column_name):
    er_counts_dict = create_count_dict_for_cat_list(cats)
    for i in range(len(df_cp[column_name])):
        if df_cp[column_name][i] != -1:
            diag = str(df_cp[column_name][i])
            add_value_to_cat_dict(diag, cats, er_counts_dict)
    return er_counts_dict


def check_and_replace_values_in_column_by_list(df_cp, desired_values_list, column_name, replacing_val):
    for i in range(len(df_cp[column_name])):
        diag = str(df_cp[column_name][i])
        if ',' in diag:
            split_diag = diag.split(',')
            for value in split_diag:
                replace = True
                value = value.strip()
                if value in desired_values_list and value != '' and not value.isspace():
                    df_cp.loc[i, column_name] = value.strip()
                    replace = False
                    break;
                if replace:
                    df_cp.loc[i, column_name] = replacing_val.strip()
        elif diag.strip() not in desired_values_list and diag.strip() != '' and not diag.isspace():
            df_cp.loc[i, column_name] = replacing_val.strip()
        elif diag.strip() == '':
            df_cp.loc[i, column_name] = replacing_val.strip()
        else:
            df_cp.loc[i, column_name] = diag.strip()


def calculate_optimal_split(df, column_name, number_of_quartiles):
    quantiles = np.linspace(0, 100, number_of_quartiles + 1)
    quartiles = np.percentile(df[column_name], quantiles)

    bin_edges = [df[column_name].min()] + list(quartiles[1:-1]) + [
        df[column_name].max()]

    return bin_edges


def calculate_input_size(df_copy, numeric_columns, categorical_columns):
    # Initialize the transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(sparse_output=False)  # One-hot encoding

    # Create a column transformer to process numeric and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ]
    )

    # Fit and transform the data (without doctor column)
    X_processed = preprocessor.fit_transform(df_copy)

    # Return the number of features after preprocessing
    return X_processed.shape[1]


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="viridis", xticks_rotation="vertical")
    plt.title("Confusion Matrix")
    plt.show()


# =========== Dataset loading ===========
df = pd.read_csv(load_path)

df_copy = df.copy()

# =========== Pre-Process Data For Model ===========
df_copy.drop(columns=['Admission_Entry_Date', 'Release_Date', 'Admission_Entry_Date2', 'Release_Date2'], inplace=True)

diag_in_rel_categories = get_diag_categories(df_copy, 'Diagnosis_In_Release')

diag_in_rec_categories = get_diag_categories(df_copy, 'Diagnosis_In_Reception')

counts_dict_diag_in_rel = get_diag_counts_for_each_category(df_copy, diag_in_rel_categories, 'Diagnosis_In_Release')
counts_dict_diag_in_rec = get_diag_counts_for_each_category(df_copy, diag_in_rec_categories, 'Diagnosis_In_Reception')

sorted_counts_dict_diag_in_rel = dict(sorted(counts_dict_diag_in_rel.items(), key=lambda x:x[1], reverse=True))
print(sorted_counts_dict_diag_in_rel, '\n')

sorted_counts_dict_diag_in_rec = dict(sorted(counts_dict_diag_in_rec.items(), key=lambda x:x[1], reverse=True))
print(sorted_counts_dict_diag_in_rec, '\n')

most_common_diags_rel = []
most_common_diags_rec = []
for i in range(3):
    most_common_diags_rel.append(list(sorted_counts_dict_diag_in_rel.keys())[i])
    most_common_diags_rec.append(list(sorted_counts_dict_diag_in_rec.keys())[i])

print(most_common_diags_rel)
print(most_common_diags_rec, '\n')

check_and_replace_values_in_column_by_list(df_copy, most_common_diags_rel, 'Diagnosis_In_Release', 'Other')

check_and_replace_values_in_column_by_list(df_copy, most_common_diags_rec, 'Diagnosis_In_Reception', 'Other')

print(df_copy['Diagnosis_In_Release'].unique())
print(df_copy['Diagnosis_In_Reception'].unique(), '\n')

print(df_copy['Diagnosis_In_Release'].value_counts(), '\n')
print(df_copy['Diagnosis_In_Reception'].value_counts(), '\n')

period_between_addmissions = df_copy['Period_Between_Admissions'].copy()
doctor_col = df_copy['Releasing_Doctor'].copy()
df_copy.drop(columns=['Releasing_Doctor', 'Period_Between_Admissions'], inplace=True)

categorical_cols = ['unitName1', 'unitName2' ,'Entry_Type', 'Patient_Origin', 'Release_Type', 'Diagnosis_In_Reception', 'Diagnosis_In_Release', 'ct']
numerical_cols = ['Admission_Days', 'Admission_Days2']

num_doctors = len(set(doctor_col))

input_size = calculate_input_size(df_copy, numerical_cols, categorical_cols)

predictor = NNHospitalizationPredictor(input_size=input_size, num_doctors=num_doctors, embedding_dim=10, num_epochs=3000)

X_processed, doctor_encoded, y_processed = predictor.preprocess_data(df_copy, period_between_addmissions, doctor_col, numeric_columns=numerical_cols, categorical_columns=categorical_cols)

X_train, X_test, doctor_train, doctor_test, y_train, y_test = predictor.split_data(X_processed, doctor_encoded, y_processed)

predictor.train(X_train, doctor_train, y_train)
print()

# Evaluate the model
predicted, true_labels = predictor.evaluate(X_test, doctor_test, y_test)

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate classification report (which includes precision, recall, F1-score)
print("Classification Report:")
print(classification_report(true_labels, predicted), '\n')

class_names = ['short', 'mid', 'long']
plot_confusion_matrix(true_labels, predicted, class_names)
