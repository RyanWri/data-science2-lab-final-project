import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from overrides import overrides
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from src.team_1.hospital_models.generic_torch_model import NeuralNetworkClassifier, ClassificationPipeline
import torch.nn as nn
import torch.optim as optim

class Task21:
    Translator = {'תאריך': 'Date', 'מחלקה': 'Unit', 'כמות קבלות': 'Amount Of Admissions', 'שיעור תפוסה': 'Occupation Rate',
                  'כמות שוהים': 'Amount Of Patients',}
    @staticmethod
    def calculate_time_differences_between_columns(column1, column2, direction=2, scale='D'):
        """
        This method calculates the time difference between two datetime columns
        Args:
            column1: column for the difference
            column2: column for the difference
            direction: direction of difference (2 for 2-1, 1 for 1-2, default is 2)
            scale: sets a scale for the difference: 'D': for days, 's': for seconds, 'm': for minutes,
            'H': for hours, 'M': for months, 'Y': for years, 'W': for weeks

        Returns:

        """
        column1_date_obj = pd.to_datetime(column1)
        column2_date_obj = pd.to_datetime(column2)

        if direction != 1 and direction != 2:
            raise ValueError("Direction must be 1 or 2")

        elif direction == 1:
            difference =  column1_date_obj - column2_date_obj

        else:
            difference =  column2_date_obj - column1_date_obj

            if scale == 'D':
                difference /= pd.Timedelta(days=1)

            elif scale == 's':
                difference /= pd.Timedelta(seconds=1)

            elif scale == 'm':
                difference /= pd.Timedelta(minutes=1)

            elif scale == 'H':
                difference /= pd.Timedelta(hours=1)

            elif scale == 'M':
                difference /= pd.Timedelta(months=1)

            elif scale == 'Y':
                difference /= pd.Timedelta(years=1)

            elif scale == 'W':
                difference /= pd.Timedelta(weeks=1)

            else:
                raise ValueError(f'{scale} is not a valid scale')

            return difference

    @staticmethod
    def merge_two_columns_if_identical(df,column1,column2,target_column):
        if not df[column1].equals(df[column2]):
            print(f"{column1} and {column2} are not identical, no merge was done")

        else:
            df[target_column] = df[column1]
            position_column_1 = df.columns.get_loc(column1)
            df.drop([column1,column2], axis=1, inplace=True)
            df.insert(position_column_1, target_column, df.pop(target_column))

    @staticmethod
    def label_data(df,column_to_label,labels):
        copy_df = df.copy()
        num_labels = len(labels)
        quantile_step = 1 / num_labels
        desired_quantiles = []
        quantiles_limits = []

        for i in range(num_labels):
            desired_quantiles.append(quantile_step * (i + 1))
            quantiles_limits.append(copy_df[column_to_label].quantile(desired_quantiles[i]))

        # first limit
        copy_df.loc[copy_df[column_to_label] <= quantiles_limits[0],column_to_label] = labels[0]

        # intermediate limits
        for i in range(1, len(quantiles_limits) ,1):
            for value in copy_df[column_to_label]:
                if not isinstance(value, str):
                    if quantiles_limits[i - 1] < value <= quantiles_limits[i]:
                        copy_df.loc[copy_df[column_to_label] == value,column_to_label] = labels[i]

        return copy_df

    @staticmethod
    def add_release_rate_column(hospitalization_df, admissions_df, discharging_unit_col, release_date_cols):
        """
        Adds a 'Release_Rate' column to a copy of the hospitalization dataframe. The release rate is calculated
        as the number of patients released on a given date divided by the number of admissions on that date.

        Parameters:
        - hospitalization_df: DataFrame containing hospitalization data with at least a 'Discharging Unit' column
          and one or more columns representing release dates.
        - admissions_df: DataFrame containing admissions data with columns for 'Unit', 'Date', and 'Amount Of Admissions'.
        - discharging_unit_col: Name of the column in the hospitalization_df that represents the discharging unit.
        - release_date_cols: List of columns in the hospitalization_df that represent the release dates.

        Returns:
        - A new DataFrame with the 'Release_Rate' column added.
        """
        # Create a copy of the input dataframes
        hospitalization_df_copy = hospitalization_df.copy()
        admissions_df_copy = admissions_df.copy()

        # Convert release date columns to datetime and normalize in the copied dataframe
        for col in release_date_cols:
            hospitalization_df_copy[col] = pd.to_datetime(hospitalization_df_copy[col], errors='coerce').dt.normalize()

        # Convert date in admissions dataframe copy to datetime and normalize
        admissions_df_copy['Date'] = pd.to_datetime(admissions_df_copy['Date']).dt.normalize()

        # Group by unit and date in the admissions table to get the number of admissions per date and unit
        admissions_per_unit_date = admissions_df_copy.groupby(['Unit', 'Date']).agg(
            {'Amount Of Admissions': 'sum'}).reset_index()

        # Create a mapping from (unit, date) to admissions count
        admissions_dict = admissions_per_unit_date.set_index(['Unit', 'Date'])['Amount Of Admissions'].to_dict()

        # Count the number of releases for each unit and date
        releases_count_df = pd.concat([
            hospitalization_df_copy.groupby([discharging_unit_col, col]).size().reset_index(name='Release_Count')
            for col in release_date_cols
        ]).groupby([discharging_unit_col, col], as_index=False)['Release_Count'].sum()

        # Rename columns properly
        releases_count_df.columns = [discharging_unit_col, 'Release_Date', 'Release_Count']

        # Create a mapping from (unit, date) to release count
        releases_dict = releases_count_df.set_index([discharging_unit_col, 'Release_Date'])['Release_Count'].to_dict()

        # Function to calculate the Release Rate for each row
        def calculate_release_rate(row):
            unit = row[discharging_unit_col]
            total_releases = 0
            total_admissions = 0

            # Sum releases and admissions for all release dates (if available)
            for col in release_date_cols:
                release_date = row[col]
                if pd.notna(release_date):
                    total_releases += releases_dict.get((unit, release_date), 0)
                    total_admissions += admissions_dict.get((unit, release_date), 0)

            # Calculate release rate if there are admissions
            if total_admissions > 0:
                release_rate = round(total_releases / total_admissions, 2)
            else:
                release_rate = 0.0

            return release_rate

        # Apply the function to calculate the Release Rate and insert it after the Discharging Unit column
        hospitalization_df_copy.insert(
            hospitalization_df_copy.columns.get_loc(discharging_unit_col) + 1,
            'Release_Rate',
            hospitalization_df_copy.apply(calculate_release_rate, axis=1)
        )

        return hospitalization_df_copy

    @staticmethod
    def add_occupation_and_patient_columns(
            hospitalization_df,
            occupancy_df,
            discharging_unit_col,
            release_date_cols,
            occupancy_unit_col,
            occupancy_date_col,
            occupancy_rate_col,
            amount_patients_col):
        """
        Adds 'Occupation_Rate' and 'Amount of Patients' columns to a copy of the hospitalization dataframe.
        The 'Occupation_Rate' is based on the data from the 'UnitsOccupancyRate' table for each discharging unit
        on the given release dates, and 'Amount of Patients' represents the number of patients in each unit on that date.

        Parameters:
        - hospitalization_df: DataFrame containing hospitalization data with at least a 'Discharging Unit' column and release date columns.
        - occupancy_df: DataFrame containing occupancy rate data with columns for 'Unit', 'Date', 'Occupancy Rate', and 'Amount of Patients'.
        - discharging_unit_col: Name of the column in the hospitalization_df that represents the discharging unit.
        - release_date_cols: List of columns in the hospitalization_df that represent the release dates.
        - occupancy_unit_col: Name of the column in the occupancy_df that represents the unit.
        - occupancy_date_col: Name of the date column in the occupancy_df.
        - occupancy_rate_col: Name of the column in the occupancy_df that represents the occupancy rate.
        - amount_patients_col: Name of the column in the occupancy_df that represents the amount of patients.

        Returns:
        - A new DataFrame with the 'Occupation_Rate' and 'Amount of Patients' columns added after the discharging unit column.
        """
        # Create a copy of the input dataframes
        hospitalization_df_copy = hospitalization_df.copy()
        occupancy_df_copy = occupancy_df.copy()

        # Convert date columns to datetime and normalize
        for col in release_date_cols:
            hospitalization_df_copy[col] = pd.to_datetime(hospitalization_df_copy[col], errors='coerce').dt.normalize()
        occupancy_df_copy[occupancy_date_col] = pd.to_datetime(occupancy_df_copy[occupancy_date_col],
                                                               errors='coerce').dt.normalize()

        # Create mappings from (unit, date) to occupancy rate and amount of patients
        occupancy_rate_dict = occupancy_df_copy.set_index([occupancy_unit_col, occupancy_date_col])[
            occupancy_rate_col].to_dict()
        amount_patients_dict = occupancy_df_copy.set_index([occupancy_unit_col, occupancy_date_col])[
            amount_patients_col].to_dict()

        # Function to calculate the Occupation Rate and Amount of Patients for each row
        def calculate_occupation_and_patients(row):
            unit = row[discharging_unit_col]
            occupation_rate = 0.0
            amount_patients = 0

            # Check for each release date column
            for col in release_date_cols:
                release_date = row[col]
                if pd.notna(release_date):
                    # Get the occupation rate and amount of patients for the unit and date
                    occupation_rate = occupancy_rate_dict.get((unit, release_date), 0.0)
                    amount_patients = amount_patients_dict.get((unit, release_date), 0)
                    if occupation_rate > 0.0 or amount_patients > 0:
                        break  # If a valid occupation rate or patient count is found, break the loop

            return round(occupation_rate, 2), amount_patients

        # Apply the function to calculate the Occupation Rate and Amount of Patients
        results = hospitalization_df_copy.apply(calculate_occupation_and_patients, axis=1)
        hospitalization_df_copy['Occupation_Rate'] = results.apply(lambda x: x[0])
        hospitalization_df_copy['Amount of Patients'] = results.apply(lambda x: x[1])

        # Insert the new columns right after the Discharging Unit column
        hospitalization_df_copy.insert(
            hospitalization_df_copy.columns.get_loc(discharging_unit_col) + 1,
            'Occupation_Rate',
            hospitalization_df_copy.pop('Occupation_Rate')
        )

        hospitalization_df_copy.insert(
            hospitalization_df_copy.columns.get_loc(discharging_unit_col) + 2,
            'Amount of Patients',
            hospitalization_df_copy.pop('Amount of Patients')
        )

        return hospitalization_df_copy


    @staticmethod
    def analyze_discharging_unit_relationship(data):
        # Use LabelEncoder to encode the categorical rehospitalization time for statistical testing
        le = LabelEncoder()
        data['Days_To_Rehospitalization_encoded'] = le.fit_transform(data['Days_To_Rehospitalization'])

        # Inverse transform to get the original labels back
        rehospitalization_labels =['Long','Medium','Short']

        # Plot the distribution of rehospitalization time across different discharging units using boxplot and barplot
        plt.figure(figsize=(12, 6))

        # Box plot showing the spread of rehospitalization time for each discharging unit, with correct labels and colors
        plt.subplot(1, 2, 1)
        sns.boxplot(x='Days_To_Rehospitalization', y='Discharging Unit', data=data, palette="Set2",hue='Days_To_Rehospitalization',
                    legend=False)
        plt.title('Discharging Unit vs Rehospitalization Time')
        plt.ylabel('Rehospitalization Time')
        plt.xticks([0, 1, 2], rehospitalization_labels)

        # Bar plot showing the count of each rehospitalization time category per discharging unit, with correct labels
        plt.subplot(1, 2, 2)
        sns.countplot(x='Discharging Unit', hue='Days_To_Rehospitalization', data=data, palette="Set2")
        plt.title('Rehospitalization Time Distribution by Discharging Unit')
        plt.legend(title='Rehospitalization Time', loc='upper right', labels=rehospitalization_labels)

        plt.tight_layout()
        plt.show()

        # Perform ANOVA to see if there are significant differences between rehospitalization time across discharging units
        anova_result = stats.f_oneway(
            data[data['Discharging Unit'] == 1]['Days_To_Rehospitalization_encoded'],
            data[data['Discharging Unit'] == 2]['Days_To_Rehospitalization_encoded'],
            data[data['Discharging Unit'] == 3]['Days_To_Rehospitalization_encoded'],
            data[data['Discharging Unit'] == 4]['Days_To_Rehospitalization_encoded'],
            data[data['Discharging Unit'] == 5]['Days_To_Rehospitalization_encoded']
        )

        # Display the ANOVA result
        print(f"ANOVA Test Result: F-statistic = {anova_result.statistic}, p-value = {anova_result.pvalue}")

        # Conclusion based on the ANOVA result
        if anova_result.pvalue < 0.05:
            print("Conclusion: The discharging unit has a statistically significant effect on rehospitalization time.")
        else:
            print("Conclusion: No statistically significant effect of the discharging unit on rehospitalization time.")




class LogisticClassifier(NeuralNetworkClassifier):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LogisticClassifier, self).__init__(input_size, hidden_size, num_classes)
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
            nn.Softmax(dim=1)
        )

class LogisticClassificationPipeline(ClassificationPipeline):
    def __init__(
        self, df, features, target, hidden_size=10, test_size=0.3, random_state=42
    ):
        super(LogisticClassificationPipeline, self).__init__(df, features, target, hidden_size, test_size, random_state)


    @overrides
    def train_model(
        self,
        X_train,
        y_train,
        input_size,
        num_classes,
        num_epochs=100,
        learning_rate=0.01,
    ):
        """
        Train the neural network model.

        Parameters:
        -----------
        X_train : torch.Tensor
            The training data features.
        y_train : torch.Tensor
            The training data labels.
        input_size : int
            The number of input features.
        num_classes : int
            The number of output classes.
        num_epochs : int
            The number of epochs for training.
        learning_rate : float
            The learning rate for the optimizer.

        Returns:
        --------
        model : NeuralNetworkClassifier
            The trained model.
        """
        model = LogisticClassifier(
            input_size=input_size, hidden_size=self.hidden_size, num_classes=num_classes
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            outputs = model.forward(X_train)
            loss = criterion(outputs, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

        return model


class ResNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ResNet, self).__init__()

        # Define the model layers using nn.Sequential
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Pass through the sequential layers
        residual = self.model[:2](x)  # Output after first two layers
        out = self.model[2:](residual)  # Output after second two layers

        # Add the skip connection
        out = out + residual

        # Pass through the final output layer
        out = self.output_layer(out)
        return out

class ResnetClassificationPipeline(ClassificationPipeline):
    def __init__(
        self, df, features, target, hidden_size=10, test_size=0.3, random_state=42
    ):
        super(ResnetClassificationPipeline, self).__init__(df, features, target, hidden_size, test_size, random_state)


    @overrides
    def train_model(
        self,
        X_train,
        y_train,
        input_size,
        num_classes,
        num_epochs=100,
        learning_rate=0.01,
    ):
        """
        Train the neural network model.

        Parameters:
        -----------
        X_train : torch.Tensor
            The training data features.
        y_train : torch.Tensor
            The training data labels.
        input_size : int
            The number of input features.
        num_classes : int
            The number of output classes.
        num_epochs : int
            The number of epochs for training.
        learning_rate : float
            The learning rate for the optimizer.

        Returns:
        --------
        model : NeuralNetworkClassifier
            The trained model.
        """
        model = ResNet(
            input_size=input_size, hidden_size=self.hidden_size, num_classes=num_classes
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            outputs = model.forward(X_train)
            loss = criterion(outputs, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

        return model



















