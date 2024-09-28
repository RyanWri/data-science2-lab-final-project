from scipy.stats import chi2_contingency


# Visualize relationships between categorical variables and re-hospitalization
def create_plot_relationship(df, x_col, title):
    fig = px.histogram(df, x=x_col, barmode='group'). \
        update_layout(yaxis_title="Number of Recordings",
                      title=title,
                      title_font_size=30)
    fig.update_xaxes(tickfont_size=20, title=" ")
    fig.update_yaxes(title_font={"size": 20})
    fig.update_traces(opacity=0.9)
    return fig

# Relationship between Education and Rehospitalization
create_plot_relationship(df_GeneralData, 'השכלה', 'Rehospitalization by Education Level')

# Relationship between Number of Children and Rehospitalization
create_plot_relationship(df_GeneralData, 'מספר ילדים', 'Rehospitalization by Number of Children')

# Relationship between Marital Status and Rehospitalization
create_plot_relationship(df_GeneralData, 'מצב משפחתי', 'Rehospitalization by Marital Status')

# Chi-square tests for relationships
def chi_square_test(df, col):
    contingency_table = pd.crosstab(df[col],data['days_between_admissions'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-square test for {col}: Chi2 = {chi2}, p-value = {p}")

# Apply Chi-square tests
chi_square_test(df_GeneralData, 'השכלה')
chi_square_test(df_GeneralData, 'מספר ילדים')
chi_square_test(df_GeneralData, 'מצב משפחתי')
