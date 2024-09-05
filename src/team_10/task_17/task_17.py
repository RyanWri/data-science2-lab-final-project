# ====================== Finding Optimal Split ======================

# =========== Imports ===========
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========== Path Config ==========
load_path = '../../data/hospitalization2_Team_10.csv'

# =========== Load Dataset ===========
df = pd.read_csv(load_path)

print(df.info(), '\n')

df['Admission_Entry_Date'] = pd.to_datetime(df['Admission_Entry_Date'], format='%Y-%m-%d %H:%M:%S.%f')
df['Release_Date'] = pd.to_datetime(df['Release_Date'], format='%Y-%m-%d %H:%M:%S')

# Calculate duration in days, rounded
df['Admission_days'] = (df['Release_Date'] - df['Admission_Entry_Date']).dt.round('D').abs().dt.days

# Create an array that counts the occurrences of each duration
duration_counts = df['Admission_days'].value_counts().sort_index()

# Plot the histogram
plt.figure(figsize=(15, 8))
plt.bar(duration_counts.index, duration_counts.values, width=0.5, color='blue')
plt.xlabel('Duration (Days)')
plt.ylabel('Count')
plt.title('Histogram of Duration in Days')
plt.xticks(duration_counts.index)  # Ensure each duration day is a tick
plt.xticks(fontsize=11)
plt.show()

# Calculate the quartiles
quartiles = np.percentile(df['Admission_days'], [25, 50, 75])

# Define bin edges for 4 equal-sized groups
bin_edges = [df['Admission_days'].min(), quartiles[0], quartiles[1], quartiles[2], df['Admission_days'].max()]

colors = ['red', 'green', 'blue', 'orange']  # One color for each quartile

plt.figure(figsize=(15, 8))
n, bins, patches = plt.hist(df['Admission_days'], bins=bin_edges, edgecolor='black')

for i, patch in enumerate(patches):
    patch.set_facecolor(colors[i % len(colors)])  # Assign color from the list

plt.xlabel('Duration (Days)')
plt.ylabel('Count')
plt.title('Histogram of Duration in Days Divided into 4 Equal-Sized Groups with Different Colors')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=8)

plt.show()

# Create labels for the groups based on bin edges
bin_labels = [
    f"Group 1: {int(bin_edges[0])} - {int(bin_edges[1])}",
    f"Group 2: {int(bin_edges[1])} - {int(bin_edges[2])}",
    f"Group 3: {int(bin_edges[2])} - {int(bin_edges[3])}",
    f"Group 4: {int(bin_edges[3])} - {int(bin_edges[4])}"
]

# Print the group ranges
print("Groups Distribution:")
for label in bin_labels:
    print(f"{label} days in admission2")
