#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import statsmodels.api as sm
import matplotlib.pyplot as plt

# Convert to datetime if not already
GeneralData['Admission_Entry_Date'] = pd.to_datetime(GeneralData['Admission_Entry_Date'])
GeneralData['Release_Date'] = pd.to_datetime(GeneralData['Release_Date'])

# Filter and group data for each department (1 to 5)
departments = [1, 2, 3, 4, 5]

# Function to create time series plots for admissions and releases
def create_time_series_plot(df, date_col, group_col, title):
    monthly_counts = df.groupby([pd.Grouper(key=date_col, freq='M'), group_col]).size().reset_index(name='Count')
    
    # Create time series plot
    fig = px.line(monthly_counts, x=date_col, y='Count', color=group_col,
                  title=title,
                  labels={date_col: 'Date', 'Count': 'Number of Admissions/Releases'})
    fig.update_layout(title_font_size=30)
    fig.show()

# Function to perform seasonal decomposition for a given department
def seasonal_decomposition(df, department_number):
    # Filter data for the department
    department_data = df[df['מחלקה מאשפזת1'] == department_number]
    
    # Admissions time series
    department_admissions = department_data.set_index('Admission_Entry_Date').resample('M').size()
    
    # Decomposition
    decomposition = sm.tsa.seasonal_decompose(department_admissions, model='additive')
    fig = decomposition.plot()
    fig.set_size_inches(14, 7)
    plt.suptitle(f'Seasonal Decomposition of Admissions for Department {department_number}', fontsize=16)
    plt.show()

# Loop through each department (1 to 5) for time series plots and decomposition
for department in departments:
    # Filter data for the department
    department_data = GeneralData[GeneralData['מחלקה מאשפזת1'] == department]
    
    # Time series plots for admissions and releases
    create_time_series_plot(department_data, 'Admission_Entry_Date', 'מחלקה מאשפזת1', f'Admissions Over Time for Department {department}')
    create_time_series_plot(department_data, 'Release_Date', 'מחלקה מאשפזת1', f'Releases Over Time for Department {department}')
    
    # Seasonal decomposition for admissions
    seasonal_decomposition(GeneralData, department)

