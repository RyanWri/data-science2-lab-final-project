
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_and_correlate(df, categorical_column, numerical_column, method='one-hot'):
    """
    Encode a categorical column and compute its correlation with a numerical column.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    categorical_column (str): The name of the categorical column to encode.
    numerical_column (str): The name of the numerical column to check correlation with.
    method (str): Encoding method - 'one-hot' for One-Hot Encoding, 'label' for Label Encoding.

    Returns:
    pd.Series or float: Correlation value(s) between the encoded categorical column and the numerical column.
    """
    if method == 'one-hot':
        # One-Hot Encoding
        df_encoded = pd.get_dummies(df, columns=[categorical_column], drop_first=True)
        
        # Compute correlation between all encoded columns and the numerical column
        correlations = df_encoded.corr()[numerical_column]
        
    elif method == 'label':
        # Label Encoding
        le = LabelEncoder()
        df[categorical_column + '_encoded'] = le.fit_transform(df[categorical_column])
        
        # Compute correlation between the label-encoded column and the numerical column
        correlation = df[[categorical_column + '_encoded', numerical_column]].corr().iloc[0, 1]
        correlations = correlation
    
    else:
        raise ValueError("Method must be 'one-hot' or 'label'")
    
    return correlations