# Team 09
We have implemented a CLI that enables to perform EDA on the file `rehospitalization.xlsx`.  
The file was originally uploaded to our course, in Moodle. The latest version we have worked with is in the directory `./assets`.

Our EDA consists of a couple of missions:
- **task_06**: Missing values treatment in the sheet `erBeforeHospitalization2`.  
The output is available in the file `team_09_task_06_erBeforeHospitalization.csv` (directory `assets`).
- **task_15**: Parameter exploration in the `hospitalization2.csv`.
The results & conclusions are available in the document `team_09_task_15_hospitalization2_EDA` (directory `assets`).
- **task_22**: Relationship exploration between the release day-of-week and rehospitalization.
The output is available in the file `team_09_task_22_relationship_day_of_week_to_rehospitalization.md` (directory `assets`).
- **task_31**: Timeseries analysis between 2nd admission date and rehospitalization occurrence.
The output is available in the file `team_09_task_31_admission_date_timeseries_analysis.md` (directory `assets`).

## How to use the CLI?
### Fill in missing values for erBeforeHospitalization2
The full path for `erBeforeHospitalization2` includes a number of steps, each represented by a call to `main.py` below:
- Transform sheet `erBeforeHospitalization2` to `ASCII` encoding only
- Fill in missing values
- Transform sheet `erBeforeHospitalization2` back to the original encoding
- Create `erBeforeHospitalization2`
```
./main.py -v -i rehospitalization.xlsx -o rehospitalization.xlsx --ascii-encoded erBeforeHospitalization2 && \
./main.py -v -i rehospitalization.xlsx -o rehospitalization.xlsx --missing-values erBeforeHospitalization2 && \
./main.py -v -i rehospitalization.xlsx -o rehospitalization.xlsx --original-encoded erBeforeHospitalization2 && \
./main.py -v -i rehospitalization.xlsx -o team09_task06_erBeforeHospitalization.csv --sheet-file erBeforeHospitalization2
```
### Relationship test between day of release and rehospitalization
```
./main.py -v -i rehospitalization.xlsx -o NA --relationship-test-release-date-rehospitalization
```

### Time series analysis between 2nd admission date and rehospitalization occurrence
```
./main.py -v -i rehospitalization.xlsx -o NA --time-series-analysis hospitalization2 Admission_Entry_Date
```

## Note-worthy implementation details
### task_06: Missing values treatment
`erBeforeHospitalization2` sheet has many patients who were admitted to the 2nd hospitalization without going through the `ER` (`מיון`).  
These patients lack details about `ER`, which led us to supplement values that indicate that they did not visit the ER.  
The parameters were chosen as following:
- `Medical_Record` = `1000000`
- `ev_Admission_Date` = `1900-01-01`
- `ev_Release_Time` = `1900-01-01`
- `Transport_Means_to_ER` (`דרך הגעה למיון`) = `'No Emergency Visit'`
- `ER` (`מיון`) = `'No Emergency Visit'`
- `urgencyLevelTime` = `0`
- `Diagnoses_in_ER` (`אבחנות במיון`) = `0`
- `codeDoctor` = `0`

Anyone who had a blank entry in the `Transport_Means_to_ER` (`דרך הגעה למיון`) column was updated with `'Not provided'`.

For those in the `ER` (`מיון`) column with the value `ICU` (`המחלקה לרפואה דחופה`) the missing values in the columns `Diagnoses_in_ER` (`אבחנות במיון`) and `codeDoctor` were updated with `1`.

### Non-ASCII chars
Hebrew characters belong to a broader encoding family, `UTF-8`.  
While it is widely used, "best-practice" recommends to avoid its usage as it is impossible to know which 3rd party module will be used in the system as a whole. On the contrary, `ASCII` encoding is supported by virtually any 3rd party module.
We have a dedicated mechanism, with module `HebEngTranslator` at its core, that transforms documents/tables to `ASCII` encoded only and back to the original format.

### task_15: Timeseries analysis between 2nd admission date and rehospitalization occurrence
_Load & Handle Missing Data_  
Start by loading the dataset from a CSV file.
Then count and print the number of missing values (NaNs) per column. Handle missing data by dropping rows with any missing values.

_Define Column Types_  
Dynamically identify numerical columns (`int64` and `float64`) and categorical columns (`object` and `category`).

_Descriptive Statistics_  
Print descriptive statistics for all columns, provide insights such as `mean`, `median`, `count`. Provide `standard deviation` insight for numerical data, count, unique values, and top values for categorical data.

_Visualization of Numerical Data_  
* Histograms: Generate histograms for all numerical columns to visualize their distributions. Use subplots to arrange these histograms in a grid that adapts to the number of numerical columns.  
* Box Plots: Create box plots for these columns to further analyze the distribution of data and identify outliers. Visualization of Categorical Data:
* Bar Charts: For categorical data, generate bar charts to visualize the frequency distribution of the top 10 most frequent categories in each categorical column. Adjust the figure size, rotate x-axis labels for readability, and ensure that the layout does not have overlapping elements. Correlation Analysis:
* Compute and visualize a correlation matrix for numerical columns using a heatmap, which helps identify any significant correlations between variables. Clustering of Individual Variables:
* Normalize the numerical data using StandardScaler to prepare for clustering. Perform KMeans clustering for each numerical column individually, visualizing the clustering result as a scatter plot of the scaled data against its index, colored by cluster label. This visualization is particularly useful for identifying patterns or groups within individual features. Key Details:
* Matplotlib's subplots are used extensively to create grids of plots. The grid size is dynamically adjusted based on the number of columns. Scikit-learn’s StandardScaler and KMeans are used for data scaling and clustering, respectively. Seaborn is used for enhanced visualization like box plots and heatmaps. DataFrame operations, such as selecting data types, handling missing values, and indexing, are effectively utilized to prepare and manipulate the data.

### task_22: Relationship between release day of week and re-hospitalization
None of the models should use this relationship.  
There is a definitive bias towards finding predictive ability between day of week and rehospitalization, as *only patients who are rehospitalized are mentioned in sheet `hospitalization1`*. We have no data regarding patients who are not rehospitilized.  
This prevents the *mandatory establishment of relationship existence* between day of week and rehospitalization, which makes any predictive ability describe above invalid.

Here's the output of the relevant piece of logic to prove our statement:
```
(venv) maximc@Maxims-MacBook-Pro team_9 % ./main.py -i rehospitalization.xlsx -o NA --relationship-test-release-date-rehospitalization
Type of target variable: discrete.
        Possible target labels: "rehospitalized", "non-rehospitalized"
Type of feature variable: discrete.
        Possible target labels: "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"
Conditions for statistical relationship test are not met, because of definitive bias:
        Number of rehospitilied patients: 7033 VS number of non-rehospitilized patients: 0
        We are unable to create "contingency table" that is a requirement for Chi-Squared or Fisher's Tests
(venv) maximc@Maxims-MacBook-Pro team_9 %
```
