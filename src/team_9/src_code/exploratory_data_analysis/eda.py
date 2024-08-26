from pandas import ExcelWriter, read_excel, Series
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
  def read_from_excel(cls, abs_file_path: str, sheet_name: str) -> DataFrame:
    return read_excel(abs_file_path, sheet_name=sheet_name)

  @classmethod
  def write_to_excel(cls, data_frame: DataFrame, abs_file_path: str, sheet_name: str) -> DataFrame:
    with ExcelWriter(abs_file_path, mode='a', if_sheet_exists='replace') as writer:
      data_frame.to_excel(writer, sheet_name=sheet_name, index=False)

