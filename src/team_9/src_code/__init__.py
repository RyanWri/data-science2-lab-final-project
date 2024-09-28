from os.path import join as path_join

from .config import ExecutionConfig
from .exploratory_data_analysis import EDA
from .excel import ExceltUtils

def ascii_encoded(input_filename: str, output_filename: str, sheet_name: str) -> None:
  if ExecutionConfig.VERBOSE_MODE:
    print("==> ascii_encoded() ====> START")
    print(f"==> input_filename: {input_filename} -- output_filename: {output_filename} -- sheet_name: {sheet_name}")
  input_document_path = path_join(ExecutionConfig.ASSETS_PATH, input_filename)
  if ExecutionConfig.VERBOSE_MODE:
    print(f"==> input_document_path: {input_document_path}")
  document = ExceltUtils.read_file(input_document_path)
  ExceltUtils.translate_cell_values_to_english(document, sheet_name)
  ExceltUtils.validate_ascii_encoded(document, sheet_name)
  output_document_path = path_join(ExecutionConfig.ASSETS_PATH, output_filename)
  if ExecutionConfig.VERBOSE_MODE:
    print(f"==> output_document_path: {output_document_path}")
  ExceltUtils.save_to_excel_document(document, output_document_path)
  if ExecutionConfig.VERBOSE_MODE:
    print("==> ascii_encoded() ====> END")

def original_encoded(input_filename: str, output_filename: str, sheet_name: str) -> None:
  if ExecutionConfig.VERBOSE_MODE:
    print("==> original_encoded() ====> START")
    print(f"==> input_filename: {input_filename} -- output_filename: {output_filename} -- sheet_name: {sheet_name}")
  input_document_path = path_join(ExecutionConfig.ASSETS_PATH, input_filename)
  if ExecutionConfig.VERBOSE_MODE:
    print(f"==> input_document_path: {input_document_path}")
  document = ExceltUtils.read_file(input_document_path)
  ExceltUtils.translate_cell_values_to_hebrew(document, sheet_name)
  output_document_path = path_join(ExecutionConfig.ASSETS_PATH, output_filename)
  if ExecutionConfig.VERBOSE_MODE:
    print(f"==> output_document_path: {output_document_path}")
  ExceltUtils.save_to_excel_document(document, output_document_path)
  if ExecutionConfig.VERBOSE_MODE:
    print("==> original_encoded() ====> END")

def relationship_test_release_date_rehospitalization(input_filename: str, sheet_hospitalization1_name: str = "hospitalization1", sheet_hospitalization2_name: str = "hospitalization2") -> None:
  if ExecutionConfig.VERBOSE_MODE:
    print("==> relationship_test_release_date_rehospitalization() ====> START")
    print(f"==> input_filename: {input_filename} -- sheet_hospitalization1_name: {sheet_hospitalization1_name} -- sheet_hospitalization2_name: {sheet_hospitalization2_name}")
  output_msg = "Type of target variable: discrete."
  output_msg += "\n\tPossible target labels: \"rehospitalized\", \"non-rehospitalized\""
  output_msg += "\nType of feature variable: discrete."
  output_msg += "\n\tPossible target labels: \"Sunday\", \"Monday\", \"Tuesday\", \"Wednesday\", \"Thursday\", \"Friday\", \"Saturday\""
  input_document_path = path_join(ExecutionConfig.ASSETS_PATH, input_filename)
  data_frames = EDA.read_all_sheets_from_excel(input_document_path)
  for sheet_name in [sheet_hospitalization1_name, sheet_hospitalization2_name]:
    if data_frames[sheet_name] is None:
      raise RuntimeError(f"Expected data_frame \"{sheet_name}\" to exist")
  if ExecutionConfig.VERBOSE_MODE:
    print("==> total number of patients (non-rehospitalized and hospitalized): %s" % str(len(data_frames[sheet_hospitalization1_name])))
  rehospitalized_df = EDA.filter_non_rehospitalized_patients_data(data_frames[sheet_hospitalization1_name], data_frames[sheet_hospitalization2_name], "Patient")
  if ExecutionConfig.VERBOSE_MODE:
    print("==> number of rehospitalized patients: %s" % str(len(rehospitalized_df)))
  rehospitalized_df = EDA.get_patients_with_release_day_of_week(rehospitalized_df, "Patient", "Release_Date", "%d/%m/%Y %H:%M:%S", "release_day")
  non_rehospitalized_df = EDA.filter_rehospitalized_patients_data(data_frames[sheet_hospitalization1_name], data_frames[sheet_hospitalization2_name], "Patient")
  if ExecutionConfig.VERBOSE_MODE:
    print("==> number of non-rehospitalized patients: %s" % str(len(non_rehospitalized_df)))
  if len(non_rehospitalized_df) == 0 or len(rehospitalized_df) == 0:
    output_msg += "\nConditions for statistical relationship test are not met, because of definitive bias:"
    output_msg += "\n\tNumber of rehospitilied patients: %s VS number of non-rehospitilized patients: %s" % (str(len(rehospitalized_df)), str(len(non_rehospitalized_df)))
    output_msg += "\n\tWe are unable to create \"contingency table\" that is a requirement for Chi-Squared or Fisher's Tests"
  if ExecutionConfig.VERBOSE_MODE:
    print("==> relationship_test_release_date_rehospitalization() ====> END")
  return output_msg

def store_sheet_as_file(input_filename: str, output_filename: str, sheet_name: str) -> None:
  if ExecutionConfig.VERBOSE_MODE:
    print("==> store_sheet_as_file() ====> START")
    print(f"==> input_filename: {input_filename} -- output_filename: {output_filename} -- sheet_name: {sheet_name}")
  output_file_format = output_filename.split(".")[1]
  if not output_file_format == "csv":
    raise RuntimeError(f"Unsupported file format for storage of a sheet \"\": {output_file_format}")
  input_document_path = path_join(ExecutionConfig.ASSETS_PATH, input_filename)
  if ExecutionConfig.VERBOSE_MODE:
    print(f"==> input_document_path: {input_document_path}")
  document = ExceltUtils.read_file(input_document_path)
  output_file_path = path_join(ExecutionConfig.ASSETS_PATH, output_filename)
  if ExecutionConfig.VERBOSE_MODE:
    print(f"==> output_file_path: {output_file_path}")
  if output_file_format == "csv":
    ExceltUtils.save_to_csv_file(ExceltUtils.extract_sheet(document, sheet_name)[sheet_name], output_file_path)
  if ExecutionConfig.VERBOSE_MODE:
    print("==> store_sheet_as_file() ====> END")

def split_sheet(input_filename: str, output_filename: str, sheet_name: str) -> None:
  if ExecutionConfig.VERBOSE_MODE:
    print("==> split_sheet() ====> START")
    print(f"==> input_filename: {input_filename} -- output_filename: {output_filename} -- sheet_name: {sheet_name}")
  input_document_path = path_join(ExecutionConfig.ASSETS_PATH, input_filename)
  if ExecutionConfig.VERBOSE_MODE:
    print(f"==> input_document_path: {input_document_path}")
  input_document = ExceltUtils.read_file(input_document_path)
  target_document = ExceltUtils.extract_sheet(input_document, sheet_name)
  output_document_path = path_join(ExecutionConfig.ASSETS_PATH, output_filename)
  if ExecutionConfig.VERBOSE_MODE:
    print(f"==> output_document_path: {output_document_path}")
  ExceltUtils.save_to_excel_document(target_document, output_document_path)
  if ExecutionConfig.VERBOSE_MODE:
    print("==> split_sheet() ====> END")

def time_series_analysis(input_filename: str, sheet_name: str, column_name: str) -> None:
  if ExecutionConfig.VERBOSE_MODE:
    print("==> time_series_analysis() ====> START")
    print(f"==> input_filename: {input_filename} -- sheet_name: {sheet_name} -- column_name: {column_name}")
  input_document_path = path_join(ExecutionConfig.ASSETS_PATH, input_filename)
  data_frame = EDA.read_from_excel(input_document_path, sheet_name)
  if ExecutionConfig.VERBOSE_MODE:
    print(f"==> DataFrame was read from file: {input_document_path}")
  figure = EDA.get_plot_time_series_by_month(data_frame, column_name)
  if ExecutionConfig.VERBOSE_MODE:
    print(f"==> Graphical representation of timeseries for \"{column_name}\" was completed")
  output_file_path = path_join(ExecutionConfig.ASSETS_PATH, "hospitalization_timeseries_analysis_by_month.png")
  if ExecutionConfig.VERBOSE_MODE:
    print(f"==> Graphical representation will be store at: {output_file_path}")
  EDA.store_plot(figure, output_file_path)
  if ExecutionConfig.VERBOSE_MODE:
    print("==> time_series_analysis() ====> END")

def treat_missing_values(input_filename: str, output_filename: str, sheet_name: str) -> None:
  if ExecutionConfig.VERBOSE_MODE:
    print("==> treat_missing_values() ====> START")
    print(f"==> input_filename: {input_filename} -- output_filename: {output_filename} -- sheet_name: {sheet_name}")
  input_document_path = path_join(ExecutionConfig.ASSETS_PATH, input_filename)
  data_frame = EDA.read_from_excel(input_document_path, sheet_name)
  if ExecutionConfig.VERBOSE_MODE:
    print(f"==> DataFrame was read from file: {input_document_path}")
    print("==> Number of missing data values:\n%s" % str(data_frame.isnull().sum()))
  conditions = EDA.get_conditions_for_rows_with_missing_data(sheet_name, data_frame)
  if ExecutionConfig.VERBOSE_MODE:
    print(f"==> Conditions for missing values in DataFrame were defined")
  EDA.fill_in_column_with_missing_values(sheet_name, data_frame, conditions)
  if ExecutionConfig.VERBOSE_MODE:
    print(f"==> Columns with missing values in DataFrame were filled")
  if ExecutionConfig.VERBOSE_MODE:
    print("==> Updated number of missing data values:\n%s" % str(data_frame.isnull().sum()))
  output_document_path = path_join(ExecutionConfig.ASSETS_PATH, output_filename)
  if ExecutionConfig.VERBOSE_MODE:
    print(f"==> output_document_path: {output_document_path}")
  EDA.write_to_excel(data_frame, output_document_path, sheet_name)
  original_document = ExceltUtils.read_file(input_document_path)
  processed_document = ExceltUtils.read_file(output_document_path)
  ExceltUtils.adjust_columns_width(original_document[sheet_name], processed_document[sheet_name])
  ExceltUtils.adjust_cells_font_and_font_size(processed_document[sheet_name])
  ExceltUtils.save_to_excel_document(processed_document, output_document_path)
  if ExecutionConfig.VERBOSE_MODE:
    print("==> treat_missing_values() ====> END")
