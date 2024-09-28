import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Task13:
    TRANSLATOR = {'תאריך': 'Date','קוד רופא': 'Doctor Code', 'כמות מטופלים': 'Amount Of Patients'}

    @staticmethod
    def sort_data_by_columns(dataframe, primary_column, secondary_column):
        """
        Sort a DataFrame by a primary column and then by a secondary column.

        :param dataframe: The DataFrame to sort.
        :param primary_column: The column to sort by first.
        :param secondary_column: The column to sort by second.
        :return: A sorted DataFrame.
        """
        # Ensure the secondary column is in datetime format if it's a date
        if pd.api.types.is_datetime64_any_dtype(dataframe[secondary_column]):
            dataframe[secondary_column] = pd.to_datetime(dataframe[secondary_column])

        # Sort the DataFrame by the given columns
        sorted_dataframe = dataframe.sort_values(by=[primary_column, secondary_column])
        sorted_dataframe = sorted_dataframe.reset_index(drop=True)

        return sorted_dataframe

    @staticmethod
    def plot_bar_chart_by_best_doctors(dataframe, date_column, value_column, doctor_column, resample=None,
                                       num_of_best_doctors=None):
        """
        Plot a bar chart for a specified value against dates for the top doctors with the highest average patients,
        dynamically adjusting the time axis scale according to the resampling scale.

        :param dataframe: The DataFrame containing the data.
        :param date_column: The column name for dates.
        :param value_column: The column name for the values to plot.
        :param doctor_column: The column name for the doctor codes.
        :param resample: Optional parameter for resampling the time series data (e.g., 'M' for monthly).
        :param num_of_best_doctors: Integer specifying the number of doctors with the highest average patients to plot.
        """
        # Ensure the date column is in datetime format
        dataframe[date_column] = pd.to_datetime(dataframe[date_column])

        # Group the data by the doctor column and calculate the average number of patients for each doctor
        avg_patients_per_doctor = dataframe.groupby(doctor_column)[value_column].mean()

        # Select the top N doctors with the highest average number of patients
        if num_of_best_doctors:
            top_doctors = avg_patients_per_doctor.nlargest(num_of_best_doctors).index
            dataframe = dataframe[dataframe[doctor_column].isin(top_doctors)]

        # Group the filtered data by the doctor column
        grouped = dataframe.groupby(doctor_column)

        # Plotting each doctor's time series
        plt.figure(figsize=(12, 6))

        for doctor_code, group in grouped:
            # If resampling is requested, apply it
            if resample:
                group = group.set_index(date_column).resample(resample).mean().reset_index()
                # if data is missing in specific timestamp - we will say that the doctor hasn't released anybody
                group.fillna(0, inplace=True)

            # Plot the bar chart for the current doctor
            plt.bar(group[date_column], group[value_column], label=f'Doctor {doctor_code}', alpha=0.7, width=15)

            # Adjust the time axis format dynamically
        if resample:
            if resample == 'D':
                plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            elif resample == 'W':
                plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%W'))
            elif resample == 'M':
                plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
            elif resample == 'Q':
                plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-Q%q'))
            elif resample == 'Y':
                plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))

        # Make labels and title bold
        plt.xlabel('Date', fontweight='bold')
        plt.ylabel(value_column, fontweight='bold')
        plt.title(f'Bar Chart of {value_column} by Top {num_of_best_doctors} Doctors', fontweight='bold')

        plt.legend(title=doctor_column, title_fontsize='13', fontsize='10')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.show()

    @staticmethod
    def find_upper_outliers(df, column):
        """
        Finds the upper bound for outliers and the indices of upper outliers
        in a specified column of a DataFrame based on the IQR method.

        Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name to analyze for outliers.

        Returns:
        float: The upper bound for outliers.
        list: The indices of upper outliers.
        """
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        # Calculate the Interquartile Range (IQR)
        IQR = Q3 - Q1

        # Calculate the upper bound for outliers
        upper_bound = Q3 + 1.5 * IQR

        # Find the indices of the upper outliers
        upper_outliers_indices = df[df[column] > upper_bound].index.tolist()

        return upper_bound, upper_outliers_indices

    @staticmethod
    def plot_histogram_and_95th_percentile(df, column):
        # Calculate the 95th percentile of the 'Amount Of Patients' column
        percentile_95 = df[column].quantile(0.95)

        # Plot a histogram to visualize the distribution
        plt.figure(figsize=(10, 6))
        plt.hist(df[column], bins=30, color='skyblue', edgecolor='black')
        plt.axvline(percentile_95, color='red', linestyle='dashed', linewidth=1,
                    label=f'95th Percentile: {percentile_95:.2f}')
        plt.title(f'Distribution of {column}')
        plt.xlabel(f'{column}')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()


    @staticmethod
    def calculate_average_patients_per_doctor(dataframe, doctor_column='Doctor Code',
                                              patient_column='Amount Of Patients'):
        """
        Calculates the average number of patients per doctor and returns a dataframe
        with columns 'Doctor Code' and 'Average Amount of Patients', rounded to two decimal places.

        Parameters:
        - dataframe (pd.DataFrame): The input dataframe containing doctor and patient information.
        - doctor_column (str): The column name for doctors. Default is 'Doctor Code'.
        - patient_column (str): The column name for the amount of patients. Default is 'Amount Of Patients'.

        Returns:
        - pd.DataFrame: A dataframe with columns 'Doctor Code' and 'Average Amount of Patients' rounded to two decimal places.
        """
        # Group by the doctor column and calculate the mean of the patient column
        average_patients_per_doctor = dataframe.groupby(doctor_column)[patient_column].mean().reset_index()
        # Rename columns to 'Doctor Code' and 'Average Amount of Patients'
        average_patients_per_doctor.columns = ['Doctor Code', 'Average Amount Of Releases']
        # Round the 'Average Amount of Patients' to two decimal places
        average_patients_per_doctor['Average Amount Of Releases'] = average_patients_per_doctor[
            'Average Amount Of Releases'].round(2)

        return average_patients_per_doctor



if __name__ == "__main__":
    df = pd.read_csv(r'./hdoctor_translated_sorted.csv')
    Task13.plot_bar_chart_by_best_doctors(dataframe=df, date_column='Date', value_column='Amount Of Patients',
                                          doctor_column='Doctor Code',resample='D',num_of_best_doctors=3)
    outliers = Task13.find_upper_outliers(df, 'Amount Of Patients')
    average_releases_per_doctor_df = Task13.calculate_average_patients_per_doctor(dataframe=df,
                                                                                  doctor_column='Doctor Code',
                                                                                  patient_column='Amount Of Patients')






































