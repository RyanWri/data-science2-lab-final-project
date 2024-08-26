#!python3
from argparse import ArgumentParser
from os.path import abspath, join as path_join, dirname

from lib import ascii_encoded, ExecutionConfig, original_encoded, split_sheet, treat_missing_values

def main():
  parser = ArgumentParser(description="This is Python CLI that performs EDA tasks. Its expected input originates from rehospitalization.xlsx. The program is expected to be executed from $PROJECT_ROOT/src/team_9")
  parser.add_argument("-i", "--input", type=str, required=True, help="Input document name. It is expected to be of type .xlsx and reside in team_9's assets directory.")
  parser.add_argument("-o", "--output", type=str, required=True, help="Output document name. Its type depends on the nature of the task. Possible types: .xlsx")
  parser.add_argument("-sh", "--split-sheet", type=str, help="Stores a sheet from input document as an independent .xlsx document.")
  parser.add_argument("-ae", "--ascii-encoded", type=str, help="Transforms a sheet from input document into ASCII-encoded.")
  parser.add_argument("-oe", "--original-encoded", type=str, help="Transforms a sheet from input document into original encoded.")
  parser.add_argument("-mv", "--missing-values", type=str, help="Performs treatment for the missing values in input document. Works only for a single sheet in the document.")
  parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")
  
  args = parser.parse_args()
  if args.verbose:
    ExecutionConfig.VERBOSE_MODE = True
    print(f"==> {ExecutionConfig.PROCESS_NAME} ====> START")
  
  ExecutionConfig.ASSETS_PATH = abspath(path_join(dirname(__file__), "assets"))
  if args.verbose:
    print(f"==> updated assets path to: {ExecutionConfig.ASSETS_PATH}")
  try:
    if args.split_sheet:
      split_sheet(args.input, args.output, args.split_sheet)
    if args.ascii_encoded:
      ascii_encoded(args.input, args.output, args.ascii_encoded)
    if args.original_encoded:
      original_encoded(args.input, args.output, args.original_encoded)
    if args.missing_values:
      treat_missing_values(args.input, args.output, args.missing_values)

  except RuntimeError as e:
    print(f"ERROR: {e}")
  
  if ExecutionConfig.VERBOSE_MODE:
    print(f"==> {ExecutionConfig.PROCESS_NAME} ====> END")

if __name__ == "__main__":
  main()
