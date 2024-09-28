# ====================== Clustering GeneralData ======================

# =========== Imports ===========
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import math

# ========== Path Config ==========
load_path = '../../data/rehospitalization.xlsx'
sheet = 'GeneralData'

# =========== Functions ===========


def calculate_bmi(weight, height):
    height /= 100.0
    return round(weight / math.pow(height, 2), 2)


def replace_bmi(df_cp):
    for i in range(len(df_cp['BMI'])):
        df_cp.loc[i, 'BMI'] = calculate_bmi(df_cp['Weight'][i].copy(), df_cp['Height'][i].copy())

# =========== Dataset loading & initial fix ===========
df = pd.read_excel(load_path, sheet_name=sheet)

# Translate column names from hebrew to english.
df.rename(columns={'גורם משלם': 'Paying_institute'}, inplace=True)
df.rename(columns={'משקל': 'Weight'}, inplace=True)
df.rename(columns={'גובה': 'Height'}, inplace=True)
df.rename(columns={'מחלות כרוניות': 'Chronic_diseases'}, inplace=True)
df.rename(columns={'מספר ילדים': 'Number_of_children'}, inplace=True)
df.rename(columns={'מצב משפחתי': 'Marital_status'}, inplace=True)
df.rename(columns={'השכלה': 'Education'}, inplace=True)
df.rename(columns={'תרופות קבועות': 'Regular_medications'}, inplace=True)

# =========== Basic Cleaning ===========
print('==================== GeneralData sheet basic cleaning ====================')
print(df.info, '\n')

print(np.sum(df.isnull(), axis=0), '\n')

weight_mean = round(df['Weight'].mean(), 2)
height_mean = round(df['Height'].mean(), 2)
print(f'Weight Mean: {weight_mean}')
print(f'Height Mean: {height_mean}\n')

# Fill missing Weight and Height with mean.
df.fillna({'Weight': weight_mean}, inplace=True)
df.fillna({'Height': height_mean}, inplace=True)

replace_bmi(df)

print(np.sum(df.isnull(), axis=0), '\n')

print(df['Marital_status'].value_counts(), '\n')

df.fillna({'Marital_status': 'נשוי'}, inplace=True)

df['Number_of_children'].value_counts()

# Find the uncommon values indexes.
print(np.where(df['Number_of_children'] == 'אין'))
print(np.where(df['Number_of_children'] == '1 + 2 (נפטרו)'))
print(np.where(df['Number_of_children'] == '4           4'))
print(np.where(df['Number_of_children'] == 'נכדה  אחת'))
print(np.where(df['Number_of_children'] == '2( ילדים בחוץ לארץ)'), '\n')

# Replace uncommon values with number.
df.loc[2467, 'Number_of_children'] = 0
df.loc[478, 'Number_of_children'] = 1
df.loc[588, 'Number_of_children'] = 4
df.loc[1978, 'Number_of_children'] = 1
df.loc[1292, 'Number_of_children'] = 2

df['Number_of_children'] = df['Number_of_children'].astype(float)

# Most families have 3 children, so we will fill the missing values with 3.
df.fillna({'Number_of_children': 3.0}, inplace=True)

df['Number_of_children'] = df['Number_of_children'].astype(int)

df.fillna({'Regular_medications': -1}, inplace=True)

print(np.sum(df.isnull(), axis=0), '\n')

label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

df['Marital_status'] = label_encoder.fit_transform(df['Marital_status'])

relevant_features = ['age', 'Gender', 'BMI', 'Chronic_diseases', 'Number_of_children', 'Marital_status']

df_relevant = df[relevant_features]

df_relevant['Chronic_diseases'] = df['Chronic_diseases'].astype(int)

cols_to_scale = []
for col in df_relevant.columns:
    if df_relevant[col].dtype == 'int64' or df_relevant[col].dtype == 'float64':
        cols_to_scale.append(col)

print(cols_to_scale)

df_test = df_relevant.copy()

scaler = StandardScaler()
df_test[['age', 'BMI']] = scaler.fit_transform(df_relevant[['age', 'BMI']])

print('==================== GeneralData Cleaning Completed ====================')

kmeans = KMeans(n_clusters=3, random_state=42)  # Change 'n_clusters' as needed
label = kmeans.fit_predict(df_test)
centroids = kmeans.cluster_centers_
df_relevant['Cluster'] = label

# Perform PCA to reduce dimensionality to 2 components for visualization
pca = PCA(n_components=2)
pca_features = pca.fit_transform(df_test)

# Assuming 'label' contains the cluster assignments
u_labels = np.unique(label)

# Plotting the results in 2D using the principal components
plt.figure(figsize=(8, 6))
for i in u_labels:
    plt.scatter(pca_features[label == i, 0], pca_features[label == i, 1], label=f'Cluster {i}')

plt.legend()
plt.title("K-Means Clustering with PCA (2D Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()









