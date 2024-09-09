import pandas as pd

def count_pairs(df, urgency_column, mode_columns):
    """
    Count the occurrences of each (urgency level, mode of arrival) pair.
    
    Parameters:
    - df: The DataFrame containing the data.
    - urgency_column: The name of the column that represents the urgency level.
    - mode_columns: A list of column names that represent the modes of arrival.

    Returns:
    - A DataFrame with the counts of each (urgency level, mode of arrival) pair.
    """
    # Get unique urgency levels
    unique_urgency_levels = df[urgency_column].unique()
    
    # Initialize an empty DataFrame for the counts
    pair_counts = pd.DataFrame(0, index=unique_urgency_levels, columns=mode_columns)
    
    # Iterate over the rows in the original DataFrame
    for index, row in df.iterrows():
        urgency = row[urgency_column]
        for mode in mode_columns:
            if row[mode] == 1:  # Check if the way of arrival is active in this row
                pair_counts.at[urgency, mode] += 1
    
    return pair_counts


def count_active_pairs(df, category_columns, indicator_columns):
    """
    Count the occurrences of each active (category, indicator) pair.
    
    Parameters:
    - df: The DataFrame containing the data.
    - category_columns: A list of column names representing different categories (e.g., urgency levels).
    - indicator_columns: A list of column names representing different indicators (e.g., modes of arrival).

    Returns:
    - A DataFrame with the counts of each active (category, indicator) pair.
    """
    # Initialize an empty DataFrame for the counts
    pair_counts = pd.DataFrame(0, index=category_columns, columns=indicator_columns)
    
    # Iterate over the rows in the original DataFrame
    for index, row in df.iterrows():
        # Find which category is active
        active_category = None
        for category in category_columns:
            if row[category] == 1:
                active_category = category
                break  # Stop after finding the active category
        
        # Find which indicator is active
        active_indicator = None
        for indicator in indicator_columns:
            if row[indicator] == 1:
                active_indicator = indicator
                break  # Stop after finding the active indicator
        
        # If both are found, increment the corresponding count
        if active_category and active_indicator:
            pair_counts.at[active_category, active_indicator] += 1
    
    return pair_counts

# Function to reverse the characters in a string
def reverse_string(s):
    return s[::-1]
