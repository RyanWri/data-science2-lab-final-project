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

# Function to convert a continuous variable into categories based on percentiles
def create_categories_by_precentile(data: pd.DataFrame, groups_num: int):
    data['range_days_between_admissions'] = pd.qcut(data['days_between_admissions'], q=groups_num).astype('str')  # Create percentile-based bins for the 'days_between_admissions' column
    data['range_days_between_admissions'] = data['range_days_between_admissions'].str.slice(start=1, stop=-1)
    bins_map = {value: idx for idx, value in enumerate(data['range_days_between_admissions'].unique())}
    data['category_days_between_admissions'] = data['range_days_between_admissions'].map(bins_map) # Map the ranges to their corresponding category index
    return data

if __name__ == "__main__":
    # Load data 
    data = pd.read_excel(r'src\data\rehospitalization.xlsx', sheet_name='erBeforeHospitalization2')

    # Task 16:
    # Calculate the number of days between the first and second admissions
    data['days_between_admissions'] = (data['Admission_Entry_Date2'] - data['Release_Date']).dt.days

    # Histogram 
    create_plot(data, 'days_between_admissions', 'Histogram of Days Passed Between Admissions')
    # Convert the continuous 'days_between_admissions' into categories based on percentiles
    data = create_categories_by_precentile(data, 5)
    # Histogram
    create_plot(data, 'category_days_between_admissions', 'Histogram of Days Passed Between Admissions by Category')
    print('Categories created by days:', data['range_days_between_admissions'].unique())