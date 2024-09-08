# A function that translates the contents of a column in a Dataframe according to a given dictionary
def translate_column(df, translations_dict, column_name):
    for i in range(len(df[column_name])):
        df.loc[i, column_name] = translations_dict[df[column_name][i]]