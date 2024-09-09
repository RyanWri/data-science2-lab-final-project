# ====================== Hospitalization2 Cleaning ======================

# =========== Imports ===========
from src.team_10.utils import *
from src.team_1.utils import *

# ========== Path Config ==========
load_path = '../../data/rehospitalization.xlsx'
sheet = 'hospitalization2'
export_path = f'../../data/{sheet}_Team_10.csv'
data_dir_path = 'src/data'

# =========== Functions ===========


def count_nulls_in_reception_vs_not_null_release(df_cp):
    count = 0
    for i in range(len(df_cp['Diagnosis_In_Reception'])):
        if str(df_cp['Diagnosis_In_Reception'][i]) in str(df_cp['Diagnosis_In_Release'][i]):
            count += 1
    print(f'Number of values of \'Diagnosis_In_Reception\' that exist in \'Diagnosis_In_Release\': {count}')
    print(f'Percentage of values contained by \'Diagnosis_In_Release\' column: {round((count / len(df_cp['Diagnosis_In_Reception'])) * 100, 2)}%\n')


def replace_na_vals_in_reception_diagnosis(df_cp):
    for i in range(len(df_cp['Diagnosis_In_Reception'])):
        if df_cp['Diagnosis_In_Reception'][i] == -1 and df_cp['Diagnosis_In_Release'][i] != -2:
            df_cp.loc[i, 'Diagnosis_In_Reception'] = df_cp['Diagnosis_In_Release'][i]
    print(f'Number of missing values left in \'Diagnosis_In_Reception\' column: {df_cp['Diagnosis_In_Reception'].value_counts()[-1]}\n')


def replace_missing_values_diagnosis_in_release(df_cp):
    for i in range(len(df_cp['Diagnosis_In_Release'])):
        if df_cp['Diagnosis_In_Reception'][i] != -1 and df_cp['Diagnosis_In_Release'][i] == -2:
            df_cp.loc[i, 'Diagnosis_In_Release'] = df_cp['Diagnosis_In_Reception'][i]
        elif df_cp['Diagnosis_In_Reception'][i] == -1 and df_cp['Diagnosis_In_Release'][i] == -2:
            df_cp.loc[i, 'Diagnosis_In_Release'] = -1
    print(f'Number of missing values left in \'Diagnosis_In_Release\' column: {df_cp['Diagnosis_In_Release'].value_counts()[-1]}\n')


def count_missing_doctors_with_no_diagnosis(df_cp):
    count = 0
    for i in range(len(df_cp['Releasing_Doctor'])):
        if df_cp['Diagnosis_In_Reception'][i] == -1 and df_cp['Diagnosis_In_Release'][i] == -1 and df_cp['Releasing_Doctor'][i] == -1:
            count += 1
    print(f'Number of releasing doctors with no diagnosis: {count}\n')


def add_diagnosis_to_cats(cats, diag):
    if ',' in diag:
        split_diag = diag.split(',')
        for value in split_diag:
            value = value.strip()
            if value not in cats and value != '':
                cats.append(value)
    elif diag.strip() not in cats and diag.strip() != '':
        cats.append(diag.strip())


def get_diag_categories_for_missing_doctors(df_cp):
    cats = []
    for i in range(len(df_cp['Releasing_Doctor'])):
        if df_cp['Releasing_Doctor'][i] == -1:
            if df_cp['Diagnosis_In_Release'][i] != -1:
                diag = str(df_cp['Diagnosis_In_Release'][i])
                add_diagnosis_to_cats(cats, diag)
            elif df_cp['Diagnosis_In_Reception'][i] != -1:
                diag = str(df_cp['Diagnosis_In_Reception'][i])
                add_diagnosis_to_cats(cats, diag)
    return cats


def create_doc_to_cat_dict(df_cp, cats):
    doc_to_cats_dict = {}
    cats_dict = dict(zip(cats, [0] * len(cats)))
    for doc in df_cp['Releasing_Doctor'].unique():
        doc_to_cats_dict[doc] = cats_dict.copy()
    return doc_to_cats_dict


def add_value_to_doc_cat_dict(diag, doc, cats, doc_to_cat_dict_cp):
    if ',' in diag:
        split_diag = diag.split(',')
        for value in split_diag:
            value = value.strip()
            if value in cats and value != '':
                doc_to_cat_dict_cp[doc][value] += 1
    elif diag.strip() in cats and diag.strip() != '':
        doc_to_cat_dict_cp[doc][diag.strip()] += 1


def get_doctor_counts_dict_for_each_category(df_cp, cats):
    doc_to_cat_dict_cp = create_doc_to_cat_dict(df_cp, cats)
    for i in range(len(df_cp['Releasing_Doctor'])):
        doc = df_cp['Releasing_Doctor'][i]
        if df_cp['Diagnosis_In_Release'][i] != -1:
            diag = str(df_cp['Diagnosis_In_Release'][i])
            add_value_to_doc_cat_dict(diag, doc, cats, doc_to_cat_dict_cp)
        elif df_cp['Diagnosis_In_Reception'][i] != -1:
            diag = str(df_cp['Diagnosis_In_Reception'][i])
            add_value_to_doc_cat_dict(diag, doc, cats, doc_to_cat_dict_cp)
    return doc_to_cat_dict_cp


def get_doc_with_most_diag_cases_dict(doc_to_diag_dict, cats, df_cp):
    most_diag_to_doc_dict = {}
    for cat in cats:
        most_diag_to_doc_dict[cat] = {}
        best_doc = ''
        max_val = -1
        for doc in df_cp['Releasing_Doctor'].unique():
            if doc_to_diag_dict[doc][cat] > max_val and doc != -1:
                best_doc = doc
                max_val = doc_to_diag_dict[doc][cat]
        most_diag_to_doc_dict[cat][best_doc] = max_val
    return most_diag_to_doc_dict


def get_diag(diag):
    if ',' in diag:
        split_diag = diag.split(',')
        for value in split_diag:
            value = value.strip()
            if value != '':
                return value
    elif diag.strip() != '':
        return diag.strip()


def fill_releasing_doctor(df_cp, cat_doc_most_cp):
    for i in range(len(df_cp['Releasing_Doctor'])):
        if df_cp['Releasing_Doctor'][i] == -1:
            if df_cp['Diagnosis_In_Release'][i] != -1:
                diag = get_diag(str(df_cp['Diagnosis_In_Release'][i]))
                df_cp.loc[i, 'Releasing_Doctor'] = list(cat_doc_most_cp[diag].keys())[0]
            elif df_cp['Diagnosis_In_Reception'][i] != -1:
                diag = get_diag(str(df_cp['Diagnosis_In_Reception'][i]))
                df_cp.loc[i, 'Releasing_Doctor'] = list(cat_doc_most_cp[diag].keys())[0]


def remove_comma(df_cp):
    for i in range(len(df_cp['Diagnosis_In_Release'])):
        if df_cp['Diagnosis_In_Release'][i] != -1:
            diag = str(df_cp['Diagnosis_In_Release'][i])
            if diag.strip().startswith(','):
                diag = diag.strip()[1:]
            if diag.strip().endswith(','):
                diag = diag.strip()[0:len(diag.strip()) - 1]
            df_cp.loc[i, 'Diagnosis_In_Release'] = diag
        if df_cp['Diagnosis_In_Reception'][i] != -1:
            diag = str(df_cp['Diagnosis_In_Reception'][i])
            if diag.strip().startswith(','):
                diag = diag.strip()[1:]
            if diag.strip().endswith(','):
                diag = diag.strip()[0:len(diag.strip()) - 1]
            df_cp.loc[i, 'Diagnosis_In_Reception'] = diag


# =========== Dataset loading & initial fix ===========
print('=========== Dataset loading & initial fix ===========')
df = pd.read_excel(load_path, sheet_name=sheet)

df_copy = df
df_copy.rename(columns={'סוג קבלה': 'Entry_Type'}, inplace=True)
df_copy.rename(columns={'מהיכן המטופל הגיע': 'Patient_Origin'}, inplace=True)
df_copy.rename(columns={'רופא משחרר': 'Releasing_Doctor'}, inplace=True)
df_copy.rename(columns={'ימי אשפוז': 'Admission_Days2'}, inplace=True)
df_copy.rename(columns={'אבחנות בקבלה': 'Diagnosis_In_Reception'}, inplace=True)
df_copy.rename(columns={'אבחנות בשחרור': 'Diagnosis_In_Release'}, inplace=True)
df_copy.rename(columns={'מחלקות מייעצות': 'Advisory_Departments'}, inplace=True)

# We check the info regarding each column in the Dataframe.
print(df.info, '\n')

# =========== Data Completion ===========
print('=========== Data Completion ===========')
print(np.sum(df_copy.isnull(), axis=0), '\n')

# Clean 'Diagnosis_In_Reception' & 'Diagnosis_In_Release' columns.
print('========= Clean \'Diagnosis_In_Reception\' & \'Diagnosis_In_Release\' columns =========')
df_copy.fillna({'Diagnosis_In_Reception': -1}, inplace=True)
df_copy.fillna({'Diagnosis_In_Release': -2}, inplace=True)
print(np.sum(df_copy.isnull(), axis=0), '\n')

count_nulls_in_reception_vs_not_null_release(df_copy)

replace_na_vals_in_reception_diagnosis(df_copy)
count_nulls_in_reception_vs_not_null_release(df_copy)

replace_missing_values_diagnosis_in_release(df_copy)

# Clean 'Releasing_Doctor' column
print('=========== Clean \'Releasing_Doctor\' column ===========')
df_copy.fillna({'Releasing_Doctor': -1}, inplace=True)
print(np.sum(df_copy.isnull(), axis=0), '\n')

count_missing_doctors_with_no_diagnosis(df_copy)

diag_categories = get_diag_categories_for_missing_doctors(df_copy)
print('Number of categories with missing releasing doctor:', len(diag_categories), '\n')

df_copy['Releasing_Doctor'] = df_copy['Releasing_Doctor'].astype(int)
print('Number of optional releasing doctors:', len(df_copy['Releasing_Doctor'].unique()) - 1, '\n')

doc_to_cat_dict = get_doctor_counts_dict_for_each_category(df_copy, diag_categories)

cat_doc_most = get_doc_with_most_diag_cases_dict(doc_to_cat_dict, diag_categories, df_copy)
print(cat_doc_most, '\n')

fill_releasing_doctor(df_copy, cat_doc_most)

print(df_copy['Releasing_Doctor'].value_counts()[-1], '\n')

# Clean 'Entry_Type' column
print('=========== Clean \'Entry_Type\' column ===========')
print('Value Counts Before Fill:', df_copy['Entry_Type'].value_counts(), '\n')

df_copy.fillna({'Entry_Type': 'דחוף'}, inplace=True)
print(np.sum(df_copy.isnull(), axis=0), '\n')

print('Value Counts After Fill:', df_copy['Entry_Type'].value_counts(), '\n')

# Translate hebrew categories into english.
translation_dict = {'דחוף': 'urgent', 'מוזמן': 'scheduled', 'אשפוז יום': 'day hospitalization'}
translate_column(df_copy, translation_dict, 'Entry_Type')

translation_dict = {'ממרפאה': 'medical clinic', 'מבית חולים אחר': 'different hospital', 'ממוסד': 'institude', 'מביתו': 'home', 'אחר': 'other'}
translate_column(df_copy, translation_dict, 'Patient_Origin')

translation_dict = {'שוחרר לביתו': 'home', 'שוחרר למוסד': 'institude'}
translate_column(df_copy, translation_dict, 'Release_Type')

# Drop Advisory_departments column as it is not relevant for the research question and too many null values.
df_copy = df_copy.drop(columns=['Advisory_Departments'])
print(np.sum(df_copy.isnull(), axis=0), '\n')

# Remove ',' at beginning of 'Diagnosis_In_Reception' and 'Diagnosis_In_Release' string value.
remove_comma(df_copy)

rows_to_drop = df_copy.index[(df_copy['Releasing_Doctor'] == -1) & (df_copy['Diagnosis_In_Reception'] == -1) & (df_copy['Diagnosis_In_Release'] == -1)].tolist()
print('Rows To Drop:', rows_to_drop, '\n')

df_copy['Admission_Entry_Date'] = pd.to_datetime(df_copy['Admission_Entry_Date'], format='%d/%m/%Y %H:%M')
df_copy['Release_Date'] = pd.to_datetime(df_copy['Release_Date'], format='%d/%m/%Y %H:%M:%S')

df_copy['Admission_Days'] = (df_copy['Release_Date'] - df_copy['Admission_Entry_Date']).dt.round('D').abs().dt.days

df_copy['Admission_Entry_Date2'] = pd.to_datetime(df_copy['Admission_Entry_Date2'], format='%d/%m/%Y %H:%M')
df_copy['Days_Between_Admissions'] = (df_copy['Admission_Entry_Date2'] - df_copy['Release_Date']).dt.round('D').abs().dt.days

duration_counts = df_copy['Days_Between_Admissions'].value_counts().sort_index()

plot_basic_histogram(duration_counts, 'Duration (Days)', 'Count', 'Histogram of Duration in Days')

number_of_quartiles = 3
colors = ['red', 'green', 'blue']  # Make sure the number of colors matches the number of quartiles!

bin_edges = calculate_optimal_split(df_copy, 'Days_Between_Admissions', number_of_quartiles)

plot_multi_color_basic_histogram_for_optimal_split(df_copy, 'Days_Between_Admissions', bin_edges, colors,
                                                   'Duration (Days)', 'Count',
                                                   f'Histogram of Duration in Days Divided into {number_of_quartiles} Groups')


print_optimal_groups(bin_edges, number_of_quartiles)

# Encode the target column into categories according to the 3 quartiles
df_copy['Period_Between_Admissions'] = 'long'
df_copy.loc[df_copy['Days_Between_Admissions'] < bin_edges[1], 'Period_Between_Admissions'] = 'short'
df_copy.loc[(df_copy['Days_Between_Admissions'] >= bin_edges[1]) & (df_copy['Days_Between_Admissions'] < bin_edges[2]), 'Period_Between_Admissions'] = 'mid'
df_copy.loc[df_copy['Days_Between_Admissions'] >= bin_edges[2], 'Period_Between_Admissions'] = 'long'

df_copy.drop(columns=['Patient', 'Admission_Medical_Record', 'Admission_Medical_Record2', 'Days_Between_Admissions'], inplace=True)

df_copy.drop(rows_to_drop, inplace=True)

# Export clean Data to csv file.
print('=========== Exporting Clean Data to CSV File ===========')
df_copy.to_csv(export_path, index=False)
print(f'File Path: {data_dir_path}/{sheet}.csv')
print('=========== File Exported Successfully! ===========')
