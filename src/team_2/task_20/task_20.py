
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_and_correlate(df, categorical_columns, numerical_columns, method='one-hot'):
    """
    Encode categorical columns and compute their correlation with numerical columns.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    categorical_columns (list): List of categorical column names to encode.
    numerical_columns (list): List of numerical column names to check correlation with.
    method (str): Encoding method - 'one-hot' for One-Hot Encoding, 'label' for Label Encoding.

    Returns:
    pd.DataFrame: DataFrame with correlations between encoded categorical columns and numerical columns.
    """
    results = {}
    
    if method == 'one-hot':
        # Apply One-Hot Encoding
        df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
        
        for num_column in numerical_columns:
            correlations = df_encoded.corr()[num_column]
            results[num_column] = correlations.filter(like='_').to_dict()
    
    elif method == 'label':
        for cat_column in categorical_columns:
            le = LabelEncoder()
            df[cat_column + '_encoded'] = le.fit_transform(df[cat_column])
            
            for num_column in numerical_columns:
                correlation = df[[cat_column + '_encoded', num_column]].corr().iloc[0, 1]
                if num_column not in results:
                    results[num_column] = {}
                results[num_column][cat_column] = correlation
    
    else:
        raise ValueError("Method must be 'one-hot' or 'label'")
    
    # Convert results to DataFrame for easier readability
    results_df = pd.DataFrame(results)
    return results_df
