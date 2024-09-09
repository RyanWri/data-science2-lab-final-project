from datetime import datetime
from matplotlib import pyplot
from matplotlib.figure import Figure
from pandas import ExcelWriter, read_excel, Series, Timestamp, to_datetime
from pandas.core.frame import DataFrame

class SheetTypes:
  ER_BEFORE_HOSPITALIZATION_2 = "erBeforeHospitalization2"

class EDA:
  supported_sheet_names = [SheetTypes.ER_BEFORE_HOSPITALIZATION_2]

  @classmethod
  def fill_in_column_with_missing_values(cls, sheet_name: str, data_frame: DataFrame, conditions: Series) -> None:
    if sheet_name not in cls.supported_sheet_names:
      raise RuntimeError(f"Unsupported sheet type for EDA: {sheet_name}. Supported sheet types: {cls.supported_sheet_names}")
    if sheet_name == SheetTypes.ER_BEFORE_HOSPITALIZATION_2:
      data_frame.loc[conditions, 'Medical_Record'] = 1000000
      data_frame.loc[conditions, 'ev_Admission_Date'] = '1900-01-01'
      data_frame.loc[conditions, 'ev_Release_Time'] = '1900-01-01'
      data_frame.loc[conditions, 'Transport_Means_to_ER'] = 'No Emergency Visit'
      data_frame.loc[conditions, 'ER'] = 'No Emergency Visit'
      data_frame.loc[conditions, 'urgencyLevelTime'] = 0
      data_frame.loc[conditions, 'Diagnoses_in_ER'] = 0
      data_frame.loc[conditions, 'codeDoctor'] = 0
      
      data_frame['Transport_Means_to_ER'].fillna('Not provided', inplace=True)
      
      icu_condition = data_frame['ER'] == 'ICU'
      data_frame.loc[icu_condition & data_frame['Diagnoses_in_ER'].isnull(), 'Diagnoses_in_ER'] = 1
      data_frame.loc[icu_condition & data_frame['codeDoctor'].isnull(), 'codeDoctor'] = 1

      data_frame.fillna(0, inplace=True)

  @classmethod
  def filter_non_rehospitalized_patients_data(cls, hospitalization1_df: DataFrame, hospitalization2_df: DataFrame, patient_id_column: str) -> DataFrame:
    return hospitalization1_df[hospitalization1_df[patient_id_column].isin(hospitalization2_df[patient_id_column])]

  @classmethod
  def filter_rehospitalized_patients_data(cls, hospitalization1_df: DataFrame, hospitalization2_df: DataFrame, patient_id_column: str) -> DataFrame:
    return hospitalization1_df[~hospitalization1_df[patient_id_column].isin(hospitalization2_df[patient_id_column])]
  
  @classmethod
  def get_conditions_for_rows_with_missing_data(cls, sheet_name: str, data_frame: DataFrame) -> Series:
    if sheet_name not in cls.supported_sheet_names:
      raise RuntimeError(f"Unsupported sheet type for EDA: {sheet_name}. Supported sheet types: {cls.supported_sheet_names}")
    if sheet_name == SheetTypes.ER_BEFORE_HOSPITALIZATION_2:
      return (
        data_frame['Medical_Record'].isnull() & 
        data_frame['ev_Admission_Date'].isnull() & 
        data_frame['ev_Release_Time'].isnull() & 
        data_frame['Transport_Means_to_ER'].isnull() & 
        data_frame['ER'].isnull() & 
        data_frame['urgencyLevelTime'].isnull() & 
        data_frame['Diagnoses_in_ER'].isnull() &
        data_frame['codeDoctor'].isnull()
    )

  @classmethod
  def get_patients_with_release_day_of_week(cls, hospitalization1_df: DataFrame, patient_id_column: str, release_date_column: str, release_date_format: str, release_day_column: str) -> DataFrame:
    def day_of_week(date_obj: Timestamp):
      return date_obj.strftime("%A")
    return DataFrame({
      patient_id_column: hospitalization1_df[patient_id_column],
      release_day_column: hospitalization1_df[release_date_column].apply(day_of_week)
    })

  @classmethod
  def get_plot_time_series_by_month(cls, data_frame: DataFrame, column_name: str) -> Figure:
    monthly_data = cls.get_rehospitalization_count_by_month(data_frame, column_name)
    moving_avg = [monthly_data.rolling(window=3).mean()]
    fig, ax = pyplot.subplots(figsize=(15, 9))
    ax.plot(monthly_data.index, monthly_data, marker="o", linestyle="--", label="Monthly data")
    ax.plot(moving_avg[0].index, moving_avg[0], marker="o", linestyle="-", label="3-Month Moving Average")
    ax.set_title("Montly occurrences of rehospitalization")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of occurrences")
    ax.grid(True)
    ax.legend()
    pyplot.xticks(rotation=45, ha='right')
    return fig

  @classmethod
  def get_rehospitalization_count_by_month(cls, data_frame: DataFrame, column_name: str) -> DataFrame:
    df_copy = data_frame.copy()
    df_copy[column_name] = to_datetime(df_copy[column_name])
    df_copy.set_index(column_name, inplace=True)
    df_copy = df_copy.resample("ME").size()
    df_copy.index = df_copy.index.strftime("%b%y")
    return df_copy

  @classmethod
  def read_from_excel(cls, abs_file_path: str, sheet_name: str) -> DataFrame:
    return read_excel(abs_file_path, sheet_name=sheet_name)

  @classmethod
  def read_all_sheets_from_excel(cls, abs_file_path: str) -> DataFrame:
    return cls.read_from_excel(abs_file_path, None)

  @classmethod
  def store_plot(cls, figure: Figure, abs_file_path: str) -> None:
    figure.savefig(abs_file_path)

  @classmethod
  def write_to_excel(cls, data_frame: DataFrame, abs_file_path: str, sheet_name: str) -> DataFrame:
    with ExcelWriter(abs_file_path, mode='a', if_sheet_exists='replace') as writer:
      data_frame.to_excel(writer, sheet_name=sheet_name, index=False)

