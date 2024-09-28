# ====================== erDoctor EDA ======================

# =========== Imports ===========
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ========== Path Config ==========
load_path = '../../data/rehospitalization.xlsx'
sheet = 'erDoctor'

# =========== Dataset loading & initial fix ===========
df = pd.read_excel(load_path, sheet_name=sheet)

df.rename(columns={'תאריך': 'Date'}, inplace=True)
df.rename(columns={'קוד רופא': 'Doctor_ID'}, inplace=True)
df.rename(columns={'כמות מטופלים': 'Patient_count'}, inplace=True)

# =========== Perform EDA ===========
print(df.info(), '\n')

print(np.sum(df.isnull(), axis=0), '\n')

days_count = df['Doctor_ID'].value_counts()
print(days_count, '\n')

# Group by doctor and sum the number of patients for each doctor
doctor_patient_totals = df.groupby('Doctor_ID')['Patient_count'].sum().reset_index()
print(doctor_patient_totals, '\n')

# Merge doctor counts with grouped data
merged = pd.merge(days_count, doctor_patient_totals, on='Doctor_ID')
merged.rename(columns={'count': 'Days_Count'}, inplace=True)
print(merged, '\n')

# Calculate the average number of doctors per day of duration
merged['Patient_count'] = merged['Patient_count'].astype(float)
merged['average_releases_per_day'] = merged['Patient_count'] / merged['Days_Count']
print(merged, '\n')

# Round 'average_releases_per_day' to the nearest 0.1
merged['rounded_average_releases_per_day'] = (merged['average_releases_per_day'] * 10).round() / 10

# Count the occurrences of each rounded value
counts = merged['rounded_average_releases_per_day'].value_counts().sort_index()

# Calculate the mean and standard deviation of 'average_releases_per_day'
mean_value = merged['rounded_average_releases_per_day'].mean()
std_dev = merged['rounded_average_releases_per_day'].std()

# Define the range for N standard deviations from the mean
N_std = 3
lower_bound = mean_value - N_std * std_dev
upper_bound = mean_value + N_std * std_dev

# Filter the values within this range
filtered_array = merged[(merged['rounded_average_releases_per_day'] <= lower_bound) |
                        (merged['rounded_average_releases_per_day'] >= upper_bound)]['rounded_average_releases_per_day'].values

print("\nAverage releases per day evaluation:\nSD = ", std_dev, " Mean = ", mean_value)
print("lower_bound = ", lower_bound, "\nupper_bound = ", upper_bound)
# Print the filtered array
print(f"\nDoctor Average releases per day -- outside {N_std} SDs (Doctor_ID: Average_Releases_Per_Day):")

# Filter the doctor codes where 'average_releases_per_day' is above the threshold
doctors_above_Nsd = merged[merged['average_releases_per_day'] > upper_bound]['Doctor_ID']
doctors_above_Nsd = [str(ele) for ele in doctors_above_Nsd]
releases_outside_of_SD_dict = dict(zip(doctors_above_Nsd, filtered_array))
print(releases_outside_of_SD_dict, '\n')

# Plot the histogram (bar chart)
plt.figure(figsize=(18, 12))
plt.bar(counts.index, counts.values, width=0.2, color='skyblue')
# only one line may be specified; full height
plt.axvline(x=upper_bound, color='r', label='Upper Bound')

# Set labels and title
plt.xlabel('Average Releases Per Day (Rounded to Nearest 0.1)')
plt.ylabel('Count')
plt.title('Distribution of Rounded Average Releases Per Day')

# Customize x-axis ticks
plt.xticks(counts.index, rotation=90, ha='right')

# Show grid for y-axis
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.rcParams.update({'font.size': 15})

# Adjust layout for better spacing
plt.tight_layout()

plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')

# Show the plot
plt.show()
