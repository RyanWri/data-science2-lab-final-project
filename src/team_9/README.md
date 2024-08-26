# Team 09
We have coded a CLI that enables to perform EDA on sheet `erBeforeHospitalization2` from the file `rehospitalization.xlsx`.  
The file was originally uploaded to our course section in Moodle. In any case, you can find the latest version we have worked with in the dir `./assets`.

The purpose of our EDA is to remove non-ASCII values from sheet `erBeforeHospitalization2` and fill in missing values.

## Removal of non-ASCII chars
Hebrew characters belong to a broader encoding family, `UTF-8`. While it is widely used, "best-practice" recommends to avoid its usage as it is impossible to know which 3rd party module will be used in the system as a whole.  
`ASCII` encoding is supported by virtually any 3rd party module.

## Filling in missing values
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

## How to perform EDA?
Our _1st_ step is to validate that sheet `erBeforeHospitalization2` doesn't contain non-`ASCII` characters:
```
./main.py -v -i rehospitalization.xlsx -o rehospitalization.xlsx -ae erBeforeHospitalization2
```

Our _2nd_ step is to fill in missing values in sheet `erBeforeHospitalization2`:
```
./main.py -v -i rehospitalization.xlsx -o rehospitalization.xlsx -mv erBeforeHospitalization2
```

Our _optional_, 3rd step is to bring back the original non-`ASCII` characters in sheet `erBeforeHospitalization2`:
```
./main.py -v -i rehospitalization.xlsx -o rehospitalization.xlsx -oe erBeforeHospitalization2
```
