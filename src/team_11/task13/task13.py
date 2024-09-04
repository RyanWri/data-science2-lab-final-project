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
    def calculate_releases_and_averages(hdoctor_df, hospitalization_df):
        # Create a copy of the hdoctor DataFrame to avoid modifying the original
        hdoctor_copy = hdoctor_df.copy()

        # Convert date columns to datetime for comparison
        hdoctor_copy['Date'] = pd.to_datetime(hdoctor_copy['Date'])
        hospitalization_df['Release_Date'] = pd.to_datetime(hospitalization_df['Release_Date'])
        hospitalization_df['Release_Date2'] = pd.to_datetime(hospitalization_df['Release_Date2'])

        # Initialize the new columns
        hdoctor_copy['Average Num Of Releases Made'] = 0

        # Create a DataFrame with all releases and their counts per doctor and date
        releases_counts = pd.concat([
            hospitalization_df.groupby(['Discharging Doctor', 'Release_Date']).size().reset_index(name='Count'),
            hospitalization_df.groupby(['Discharging Doctor', 'Release_Date2']).size().reset_index(name='Count').rename(
                columns={'Release_Date2': 'Release_Date'})
        ])

        releases_counts = releases_counts.groupby(['Discharging Doctor', 'Release_Date']).sum().reset_index()
        releases_counts.rename(columns={'Release_Date': 'Date'}, inplace=True)

        # Merge the releases count data with the hdoctor data
        merged_df = pd.merge(hdoctor_copy, releases_counts, how='left', left_on=['Doctor Code', 'Date'],
                             right_on=['Discharging Doctor', 'Date'])

        # Fill missing release counts by finding the latest earlier date
        merged_df['Count'] = merged_df.groupby('Doctor Code')['Count'].ffill().fillna(0)
        merged_df['Average Num Of Releases Made'] = merged_df['Count']

        # Keep only the relevant columns for the final result
        final_df = merged_df[
            ['Date', 'Doctor Code', 'Amount Of Patients', 'Average Num Of Releases Made']]

        return final_df

    if __name__ == '__main__':
        hospitaliztion_df = pd.read_excel(r'C:\DS2-final_project\data-science2-lab-final-project'
                                 r'\src\team_11\task7\hospitalization2_translated_clean.xlsx',
                                   sheet_name='hospitalization2')
        hdoctor_df = pd.read_csv(r'C:\DS2-final_project\data-science2-lab-final-project\src\team_11\task13'
                                 r'\hdoctor_translated_sorted.csv')

        final_df = calculate_releases_and_averages(hdoctor_df, hospitaliztion_df)

        pass