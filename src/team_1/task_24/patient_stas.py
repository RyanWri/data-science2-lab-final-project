import pandas as pd


def calculate_rehospitalizations(df, patient_col):
    """
    Calculates how many times each patient has been rehospitalized.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing patient and hospitalization data.
    patient_col : str
        The name of the column representing patient IDs.

    Returns:
    --------
    pd.DataFrame
        DataFrame with added column `hospitalization_count` indicating the number of hospitalizations for each patient.
    """
    df["hospitalization_count"] = df.groupby(patient_col)[patient_col].transform(
        "count"
    )
    return df


def calculate_duration_between_hospitalizations(
    df, patient_col, admission_col, release_col
):
    """
    Calculates the duration between each hospitalization for each patient based on the
    Release_Date of the previous hospitalization and the Admission_Entry_Date of the next one.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing patient and hospitalization data.
    patient_col : str
        The name of the column representing patient IDs.
    admission_col : str
        The name of the column representing admission entry dates.
    release_col : str
        The name of the column representing release dates.

    Returns:
    --------
    pd.DataFrame
        DataFrame with an added column `duration_between_hospitalizations` indicating
        the number of days between hospitalizations.
    """
    # Sort by patient_id and Admission_Entry_Date
    df = df.sort_values(by=[patient_col, admission_col])

    # Shift Release_Date to calculate time difference between the previous release and next admission
    df["previous_release"] = df.groupby(patient_col)[release_col].shift(1)

    # Calculate the duration between consecutive hospitalizations
    df["duration_between_hospitalizations"] = (
        df[admission_col] - df["previous_release"]
    ).dt.days

    return df


def classify_duration(df, duration_col):
    """
    Classifies the duration between hospitalizations into three quartiles: short, medium, and long.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing duration between hospitalizations.
    duration_col : str
        The name of the column representing the duration between hospitalizations.

    Returns:
    --------
    pd.DataFrame
        DataFrame with an added column `duration_classification` categorizing durations into short, medium, or long.
    """
    df_duration = df.dropna(subset=[duration_col])
    quartiles = df_duration[duration_col].quantile([0.33, 0.66])

    def classify_duration_value(duration):
        if duration <= quartiles[0.33]:
            return "short"
        elif duration <= quartiles[0.66]:
            return "medium"
        else:
            return "long"

    df_duration["duration_classification"] = df_duration[duration_col].apply(
        classify_duration_value
    )
    return df_duration


def process_rehospitalization_data(df, patient_col, admission_col, release_col):
    """
    Main function to process rehospitalization data, which calculates:
    1) How many times each patient was rehospitalized.
    2) Duration between each hospitalization.
    3) Classification of each duration between rehospitalization.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing patient and hospitalization data.
    patient_col : str
        The name of the column representing patient IDs.
    admission_col : str
        The name of the column representing admission entry dates.
    release_col : str
        The name of the column representing release dates.

    Returns:
    --------
    pd.DataFrame
        Processed DataFrame with the following columns:
        - hospitalization_count
        - duration_between_hospitalizations
        - duration_classification (short, medium, long)
    """
    # Convert admission and release dates to datetime
    df[admission_col] = pd.to_datetime(df[admission_col])
    df[release_col] = pd.to_datetime(df[release_col])

    # Step 1: Calculate the number of times each patient was rehospitalized
    df = calculate_rehospitalizations(df, patient_col)

    # Step 2: Calculate the duration between each hospitalization based on release and next admission
    df = calculate_duration_between_hospitalizations(
        df, patient_col, admission_col, release_col
    )

    # Step 3: Classify each duration between hospitalizations into quartiles
    df = classify_duration(df, "duration_between_hospitalizations")

    return df[
        [
            patient_col,
            "hospitalization_count",
            "duration_between_hospitalizations",
            "duration_classification",
        ]
    ]
