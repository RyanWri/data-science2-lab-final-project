import pandas as pd
import plotly.express as px
import plotly.io as pio

# Function to create and customize a histogram plot 
def create_plot(df: pd.DataFrame, x_col: str, title: str):
    fig1 = px.histogram(df, x=x_col, barmode="group"). \
        update_layout(
            yaxis_title="Count",         # Set the y-axis title
            title=title,                 # Set the plot title
            title_font_size=30           # Set the font size for the title
        )
    fig1.update_xaxes(tickfont_size=20, title=" ")  
    fig1.update_yaxes(title_font={"size": 20})      
    fig1.update_traces(opacity=0.9)                 
    fig1.show()                                     

# Load data 
data = pd.read_excel(r'src\data\rehospitalization.xlsx', sheet_name='erBeforeHospitalization2')

# Task 14:
# Print the data types of the columns in the dataset
print('Data Types: ', data.dtypes)

# Print the number of unique patients in the dataset
print('Unique number of patients: ', len(data['Patient'].unique()))

# Calculate the admission time in days for the first hospitalization event
data['admission_time_1'] = (data['Release_Date'] - data['Admission_Entry_Date']).dt.days

# Histograms
create_plot(data, 'admission_time_1', 'Histogram of Admission Time (days)')
create_plot(data, 'מחלקה מאשפזת1', 'Patients by Admission Department')

# Calculate the ER time in hours for the second hospitalization event
data['admission_time_2'] = (data['ev_Release_Time'] - data['ev_Admission_Date']).dt.total_seconds() / 3600

# Histograms
create_plot(data, 'admission_time_2', 'Histogram of ER Time (hours)')
create_plot(data, 'דרך הגעה למיון', 'ER Arrival Histogram')
create_plot(data, 'מיון', 'ER Type Histogram')
