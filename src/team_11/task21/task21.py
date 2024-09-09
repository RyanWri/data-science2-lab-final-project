import pandas as pd

class Task21:
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

















