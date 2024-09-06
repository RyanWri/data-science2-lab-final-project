# Team 09
We have coded a CLI that enables to perform EDA on sheet `erBeforeHospitalization2` from the file `rehospitalization.xlsx`.  
The file was originally uploaded to our course section in Moodle. In any case, you can find the latest version we have worked with in the dir `./assets`.

Our EDA consists of a couple of paths:
- Treatment of missing values in sheet `erBeforeHospitalization2`, document `rehospitalization.xlsx`.  
We store the output under the file `erBeforeHospitalization.csv` in the directory `assets`, with intended future use by other teams.
- Parameter exploration in the `hospitalization2.csv`.
**To be done**
- Relationship exploration between the release day of week and re-hospitalization.  
The output can be seen directly in the `stdout`. That being said, **the relationship usage is not applicable!**  
Primary reason is an existence of *definitive bias*.  
Sheet `hospitalization1` includes release dates *only for patients who are re-hospitalized*.  
More detailed explanation can be seen in the Note-worthy section "Relationship between release day of week and re-hospitalization".

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
./main.py -v -i rehospitalization.xlsx -o erBeforeHospitalization.csv --sheet-file erBeforeHospitalization2
```
### Relationship test between day of release and rehospitalization
```
./main.py -i rehospitalization.xlsx -o NA --relationship-test-release-date-rehospitalization
```

## Note-worthy implementation details
### Missing values treatment
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

### Relationship between release day of week and re-hospitalization
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
