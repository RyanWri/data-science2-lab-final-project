@echo off
setlocal enabledelayedexpansion

:: Check if an integer parameter was provided
set "param=%1"
if "%param%"=="" goto no_param
echo %param%| findstr /r "^[0-9][0-9]*$" >nul || goto no_param

:: Directory containing the source code, including the team subdirectory with integer suffix
set "SRC_DIR=src/team_%param%"
set "TEST_DIR=tests/team_%param%"

:: Check if the source directory exists
if not exist "!SRC_DIR!\" (
    echo The directory !SRC_DIR! does not exist.
    goto end_script
)
if not exist "!TEST_DIR!\" (
    echo The directory !TEST_DIR! does not exist.
    goto end_script
)

echo Running pyclean on !SRC_DIR!...
pyclean !SRC_DIR!

echo Running pyclean on !TEST_DIR!...
pyclean !TEST_DIR!

:: Apply isort to sort imports
echo Running isort on !SRC_DIR!...
isort !SRC_DIR!

echo Running isort on !TEST_DIR!...
isort !TEST_DIR!

:: Apply black to format the code
echo Running black on !SRC_DIR!...
black !SRC_DIR!

echo Running black on !TEST_DIR!...
black !TEST_DIR!

echo Formatting complete.
goto end_script

:no_param
echo Please provide an integer parameter.
goto end_script

:end_script
endlocal
