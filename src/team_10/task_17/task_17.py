# ====================== Finding Optimal Split ======================

# =========== Imports ===========
import pandas as pd
from src.team_10.utils import *

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

plot_basic_histogram(duration_counts, 'Duration (Days)', 'Count', 'Histogram of Duration in Days')

colors = ['red', 'green', 'blue']  # One color for each quartile

number_of_quartiles = 3

bin_edges = calculate_optimal_split(df, 'Admission_days', number_of_quartiles)

plot_multi_color_basic_histogram_for_optimal_split(df, 'Admission_days', bin_edges, colors,
                                                   'Duration (Days)', 'Count',
                                                   f'Histogram of Duration in Days Divided into {number_of_quartiles} Groups')


print_optimal_groups(bin_edges, number_of_quartiles)
