import pandas as pd


class Task13:
    TRANSLATOR = {'תאריך': 'Date','קוד רופא': 'Doctor Code', 'כמות מטופלים': 'Amount Of Patients'}

    @staticmethod
    def sort_duplicates_first(df, duplicate_column, sort_column):
        """
        Sort a DataFrame to group duplicates of a specified column first and
        sort these groups by another specified column.

        Parameters:
        df (pd.DataFrame): The DataFrame to sort.
        duplicate_column (str): The column where duplicates should be prioritized.
        sort_column (str): The column to sort within each group of duplicates.

        Returns:
        pd.DataFrame: The sorted DataFrame.
        """
        # Identify the duplicated rows based on the specified column
        duplicated_df = df[df.duplicated(duplicate_column, keep=False)]

        # Sort duplicates first by the duplicate column, then by the secondary sort column
        sorted_df = duplicated_df.sort_values(by=[duplicate_column, sort_column])

        return sorted_df

    @staticmethod
    def calculate_average_releases(hdoctor_df, hospitalization_df, doctor_code_col, date_col, release_date_cols,
                                   discharge_doctor_col):
        """
        Calculate the average number of releases per doctor per date.

        Parameters:
        hdoctor_df (pd.DataFrame): The DataFrame containing doctor data.
        hospitalization_df (pd.DataFrame): The DataFrame containing patient release data.
        doctor_code_col (str): The column name for doctor codes in the hdoctor DataFrame.
        date_col (str): The column name for dates in the hdoctor DataFrame.
        release_date_cols (list of str): The column names for release dates in the hospitalization DataFrame.
        discharge_doctor_col (str): The column name for discharging doctors in the hospitalization DataFrame.

        Returns:
        pd.DataFrame: The DataFrame with average releases added.
        """
        # Convert date columns to datetime
        hdoctor_df[date_col] = pd.to_datetime(hdoctor_df[date_col])
        for col in release_date_cols:
            hospitalization_df[col] = pd.to_datetime(hospitalization_df[col], errors='coerce')

        # Combine release date columns into one for easier processing
        release_data = pd.melt(
            hospitalization_df,
            id_vars=[discharge_doctor_col],
            value_vars=release_date_cols,
            var_name='Num_Of_Release',
            value_name='Release_Date_New'
        ).dropna(subset=['Release_Date_New'])

        release_data.rename(columns={'Release_Date_New': 'Release_Date'}, inplace=True)

        # Count releases per doctor per date
        release_counts = release_data.groupby([discharge_doctor_col, 'Release_Date']).size().reset_index(
            name='Releases')

        # Sort hdoctor_df for finding the latest previous date
        hdoctor_df_sorted = hdoctor_df.sort_values(by=[doctor_code_col, date_col])

        # Create a dictionary to hold the amount of patients per doctor and date
        patient_count_dict = {}

        # Populate the dictionary with the patient count
        for idx, row in hdoctor_df_sorted.iterrows():
            doctor = row[doctor_code_col]
            date = row[date_col]
            patients = row['Amount Of Patients']
            patient_count_dict[(doctor, date)] = patients

        # Function to find the latest previous date's patient count
        def find_latest_patient_count(doctor, current_date):
            earlier_dates = [date for (doc, date) in patient_count_dict.keys() if
                             doc == doctor and date <= current_date]
            if earlier_dates:
                latest_date = max(earlier_dates)
                return patient_count_dict[(doctor, latest_date)]
            else:
                return None

        # Calculate the average releases
        release_counts['Amount Of Patients'] = release_counts.apply(
            lambda x: find_latest_patient_count(x[discharge_doctor_col], x['Release_Date']), axis=1
        )

        release_counts['Amount Of Patients'].fillna(0, inplace=True)    # it could be None for not yet assigned doctors

        release_counts['Average Releases'] = release_counts['Releases'] / release_counts['Amount Of Patients']
        release_counts['Average Releases'].fillna(0, inplace=True)  # could happen if the amount of patients is 0


        # Merge with the original hdoctor DataFrame to add the new average releases column
        result_df = hdoctor_df_sorted.merge(
            release_counts[[discharge_doctor_col, 'Release_Date', 'Average Releases']],
            left_on=[doctor_code_col, date_col],
            right_on=[discharge_doctor_col, 'Release_Date'],
            how='left'
        )

      # it could be that he dates do not match perfectly so we will need to fix that 

        return result_df[[date_col, doctor_code_col, 'Amount Of Patients', 'Average Releases']]

    if __name__ == '__main__':
        hospitaliztion_df = pd.read_excel(r'C:\DS2-final_project\data-science2-lab-final-project'
                                          r'\src\team_11\task7\hospitalization2_translated_clean.xlsx')
        hospitaliztion_df.dropna()
        hdoctor_df = pd.read_csv(r'C:\DS2-final_project\data-science2-lab-final-project\src\team_11'
                                 r'\task13\hdoctor_translated.csv')
        hdoctor_df.dropna()

        calculate_average_releases(hdoctor_df=hdoctor_df, hospitalization_df=hospitaliztion_df,
                                   doctor_code_col='Doctor Code',date_col='Date',
                                   release_date_cols=['Release_Date','Release_Date2'],
                                   discharge_doctor_col='Discharging Doctor')