import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class AgeToGenderModel:
    def __init__(self, df):
        """
        Initialize with the rehospitalization DataFrame containing patient data.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing rehospitalization information including columns like:
            'patient_id', 'age', 'gender', 'duration_classification', etc.
        """
        self.df = df

    def analyze_gender_age_duration(self):
        """
        Analyze the connection between gender, age, and rehospitalization duration classification.

        This function groups by rehospitalization duration classification and shows
        the average age and gender distribution within each classification (short, medium, long).

        Returns:
        --------
        pd.DataFrame : A DataFrame showing the mean age and gender distribution in each duration classification.
        """
        # Group by duration classification and calculate the mean age and gender distribution
        duration_analysis = (
            self.df.groupby("duration_classification")
            .agg({"age": "mean", "gender": lambda x: x.value_counts(normalize=True)})
            .reset_index()
        )

        return duration_analysis

    def plot_gender_age_duration(self):
        """
        Visualize the connection between gender, age, and rehospitalization duration classification.

        This function creates bar plots showing the average age and gender distribution across
        short, medium, and long rehospitalization duration classifications.
        """
        # Plot average age by rehospitalization duration classification
        plt.figure(figsize=(10, 6))
        sns.barplot(x="duration_classification", y="age", data=self.df)
        plt.title("Average Age in Each Duration Classification")
        plt.show()

        # Plot gender distribution across duration classifications
        plt.figure(figsize=(10, 6))
        sns.countplot(x="duration_classification", hue="gender", data=self.df)
        plt.title("Gender Distribution Across Duration Classifications")
        plt.show()
