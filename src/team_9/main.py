#!python3
from argparse import ArgumentParser
from os.path import abspath, join as path_join, dirname

from src_code import ascii_encoded, ExecutionConfig, original_encoded, relationship_test_release_date_rehospitalization, split_sheet, store_sheet_as_file, time_series_analysis, treat_missing_values


def main():
  parser = ArgumentParser(description="This is Python CLI that performs EDA tasks. Its expected input originates from rehospitalization.xlsx. The program is expected to be executed from $PROJECT_ROOT/src/team_9")
  parser.add_argument("-i", "--input", type=str, required=True, help="Input document name. It is expected to be of type .xlsx and reside in team_9's assets directory.")
  parser.add_argument("-o", "--output", type=str, required=True, help="Output document name. Use NA (Not Applicable) to leave output in stdout. Its type depends on the nature of the task. Possible types: .xlsx")
  parser.add_argument("--ascii-encoded", type=str, help="Transforms a sheet from input document into ASCII-encoded.")
  parser.add_argument("--missing-values", type=str, help="Performs treatment for the missing values in input document. Works only for a single sheet in the document.")
  parser.add_argument("--original-encoded", type=str, help="Transforms a sheet from input document into original encoded.")
  parser.add_argument("--relationship-test-release-date-rehospitalization", action="store_true", help="Performs a test for statistical relationship between day of release and re-hospitalization.")
  parser.add_argument("--sheet-file", type=str, help="Stores a signle sheet in a new file. The ending controls the file format. For example, filename.csv will result in .csv file.")
  parser.add_argument("--split-sheet", type=str, help="Stores a sheet from input document as an independent .xlsx document.")
  parser.add_argument("--time-series-analysis", action="append", nargs=2, help="Performs time series analysis on a specified sheet&column. The pattern of interest is occurrence of rehospitalization.")
  parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")
  
  args = parser.parse_args()
  if args.verbose:
    ExecutionConfig.VERBOSE_MODE = True
    print(f"==> {ExecutionConfig.PROCESS_NAME} ====> START")
  
  ExecutionConfig.ASSETS_PATH = abspath(path_join(dirname(__file__), "assets"))
  if args.verbose:
    print(f"==> updated assets path to: {ExecutionConfig.ASSETS_PATH}")
  try:
    if args.ascii_encoded:
      ascii_encoded(args.input, args.output, args.ascii_encoded)
    if args.missing_values:
      treat_missing_values(args.input, args.output, args.missing_values)
    if args.relationship_test_release_date_rehospitalization:
      print(relationship_test_release_date_rehospitalization(args.input))
    if args.sheet_file:
      store_sheet_as_file(args.input, args.output, args.sheet_file)
    if args.split_sheet:
      split_sheet(args.input, args.output, args.split_sheet)
    if args.time_series_analysis:
      time_series_analysis(args.input, args.time_series_analysis[0][0], args.time_series_analysis[0][1])
    if args.original_encoded:
      original_encoded(args.input, args.output, args.original_encoded)

  except RuntimeError as e:
    print(f"ERROR: {e}")
  
  if ExecutionConfig.VERBOSE_MODE:
    print(f"==> {ExecutionConfig.PROCESS_NAME} ====> END")

if __name__ == "__main__":
  main()
