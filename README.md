# data-science2-lab-final-project
# Noam Tsfaty 314952912, Liav choen 209454693
_Final project for course Data Science 2 Afeka_
### Task 5: Cleaning Data and Completing the Table 'Hospitalization1'
Objective: Clean and preprocess the data from the hospitalization1 table, ensuring it is ready for analysis.
Methodology: We handled missing values, converted date columns to a proper datetime format, and added meaningful derived columns. We also categorized patients into rehospitalization categories (0: high, 1: medium, 2: low) based on the time between hospitalizations.
Outcome: The cleaned dataset was saved as general.csv, containing no missing values, fully structured, and ready for further analysis.


### Task 14: Exploratory Data Analysis (EDA) of Parameters in 'erBeforeHospitalization2'
Objective: Perform EDA on each of the parameters in the erBeforeHospitalization2 table.
Methodology: We visualized key distributions, including the number of drugs used per patient, days between hospitalizations, and the rehospitalization categories. Several summary statistics were computed, and the data's overall structure was explored to understand potential trends.
Outcome: Key insights into the dataset were visualized, helping identify patterns that will inform the next stages of analysis, such as the relationships between medications and rehospitalization risk.
#### Task 18: Analysis of Drugs' Influence on Rehospitalization
Objective: Investigate the relationship between the most commonly prescribed medications and rehospitalization.
Methodology: We selected the 10 most common medications from the dataset and created binary columns indicating whether each medication was prescribed to a patient. This information was used as input features in the neural network model predicting rehospitalization categories.
Outcome: Clear patterns emerged linking medication use with rehospitalization risk, providing valuable insights into patient management and care optimization.

#### Task 29: Finding the 10-20 Common Medications and Predicting Rehospitalization
Objective: Identify the 10-20 most common medications and predict their relationship with rehospitalization risk using deep learning.
Methodology: After selecting the top medications, a deep neural network model was built using these medications as features to predict rehospitalization categories. The model was trained and validated, achieving high accuracy in its predictions.
Outcome: The neural network successfully predicted rehospitalization risk categories (0: high, 1: medium, 2: low) with near-perfect accuracy, indicating strong connections between medication usage and patient outcomes.

#### Task 35: Rehospitalization Prediction for Submission
Objective: Develop a robust predictive model for rehospitalization as part of a submission project.
Methodology: A comprehensive end-to-end data pipeline was implemented, from data cleaning and EDA to model building and evaluation using neural networks. Various tasks, such as the relationship between medication and rehospitalization, were explored and successfully predicted using the model.
Outcome: The final results included high-performing models, detailed visualizations, and a clear understanding of the key factors driving rehospitalization, ready for submission.

## Guide on how to open an issue:

    1. Go to Final Project - Public in projects tab
    2. this is the link: https://github.com/users/RyanWri/projects/3
    3. add an issue in the backlog
    4. format of the issue - team_{id}-{issue title}
    5. make sure you link the issue to the repository data-science2-lab-final-project
    6. when you start to work, convert task to issue and move to in progress

## **If you’re unsure about something, please feel free to ask. We’ll gather information from various sources, and I kindly ask for your patience and understanding.<br/>Please Do not Edit anything outside src directory**

### **for any issue contact Ran or Son or Sharon**
