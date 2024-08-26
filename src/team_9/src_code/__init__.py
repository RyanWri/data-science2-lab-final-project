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
  ExceltUtils.save_to_file(document, output_document_path)
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
  ExceltUtils.save_to_file(document, output_document_path)
  if ExecutionConfig.VERBOSE_MODE:
    print("==> original_encoded() ====> END")

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
  ExceltUtils.save_to_file(processed_document, output_document_path)
  if ExecutionConfig.VERBOSE_MODE:
    print("==> treat_missing_values() ====> END")

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
  ExceltUtils.save_to_file(target_document, output_document_path)
  if ExecutionConfig.VERBOSE_MODE:
    print("==> split_sheet() ====> END")
