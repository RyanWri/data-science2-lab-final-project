from scipy.stats import chi2_contingency, f_oneway
import pandas as pd


def show_statistics(df, feature_to_compare):
    # Create a list to store the results
    results = []

    # Automatically detect columns and apply the correct test
    for column in df.columns:
        if column not in [feature_to_compare]:  # Skip unitName2 itself
            if pd.api.types.is_numeric_dtype(df[column]):
                # Perform ANOVA for numerical features
                anova_result = f_oneway(*[df[df[feature_to_compare] == unit][column] for unit in df[feature_to_compare].unique()])
                results.append({
                    'Feature': column,
                    'Test': 'ANOVA',
                    'Statistic': anova_result.statistic,
                    'p-value': anova_result.pvalue
                })
            else:
                # Perform Chi-Square test for categorical features
                contingency_table = pd.crosstab(df[feature_to_compare], df[column])
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                results.append({
                    'Feature': column,
                    'Test': 'Chi-Square',
                    'Statistic': chi2,
                    'p-value': p
                })

    # Create a DataFrame from the results list
    results_df = pd.DataFrame(results)
    return results_df
