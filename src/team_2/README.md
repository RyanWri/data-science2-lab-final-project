## Team 2 - Son and Gal

### **Task 9** - EDA for Hospitalization1

The goal of this task was to explore and understand the structure of the hospitalization1 dataset. EDA is crucial for identifying patterns, trends, and any inconsistencies or missing data that may need to be addressed during preprocessing. We performed statistical summarization and visualization of the key variables using Python libraries

Translations to hebrew, multiple diagnoses - top 10 dignoses:

Admission:

1. 78609
2. 7865
3. 78060
4. 08889
5. 2859
6. 7895
7. 486
8. 4280
9. 42731
10. 7807

Release: all of the above and in addition 5990 & 514

            No strong correlation between the different features.

            Most hospitalizations were urgent and most were short, mainly a few days.

            No drastic amount of patients in any specific unit.

            Most hospitalization patients were excorted from home.

### Conclusions from Task 9:

We identified the distribution of critical features, including the length of hospital stays, the frequency of readmissions, and any apparent correlations between patient demographics and outcomes.

The EDA provided insights that guided our subsequent data preprocessing decisions, such as the need to handle missing data and outliers. This initial exploration also highlighted which features might be most relevant for our predictive models.

### **Task 20** - Classify Doctor rank (Senior/Not Senior) to rehospitalization

The objective of this task is to infer and understand if the doctors' experience and professional observance has any effect on getting rehospitalized and the duration of the hospitalizations.

In order to proceed with this task, we created an EDA for the 2 additional important datasets: hDoctor which tells us the doctors' id who released patients, and Doctors' rank if they are Seniors.That way we can understand the data like the previous task and prepare the dataset for model training by cleaning and transforming it.

#### **Preprocessing**

- Normalizing Date Columns: Many columns had date values that needed to be standardized to ensure consistency across the dataset. We used the normalize_date_column function to convert date fields into a usable format.

- Encoding Categorical Variables: Some features were categorical and needed to be encoded for machine learning models. We applied label encoding and one-hot encoding where appropriate using the encode_columns and encode_and_correlate functions.

- Merging Datasets: We integrated various datasets, including hospitalization records and rehospitalization status, using left joins to append relevant patient status information.

- Based on Doctor rank data(Senior/Not Senior), the ranks are split almost equally to 4 categories: Yes, No, ? and Depends from which date.
- Based on the PIE chart, They're split almost equally in amount of doctors.

Through Gradient Boosting Model, we trained the data to predict rehospitalization depentant on the doctor rank.

**Conclusions:**

The accuracy is 50% and ROC is 0.48 , while based on the report the precision for both classes is close to the 50% and the model recalls True outputs better, meaning we need Feature Extraction/Engineering in the preprocessing phase.

### **Task 26** - Age & Gender to hospitalization model

The objective of this task is to infer and understand if Age and/or Gender has any effect on the duration of the hospitalizations based on the categorical distributions of the duration between hospitalizations and the durations of the first and second hospitalization respectively.

In order to proceed with this task, our main assets to proceed with the preprocessing are:

1. Cleaned GeneralData,hospitalization1/2 datasets

2. The Categorized distributions for hospitalizations durations.

GeneralData was cleaned and we were able to merge the Age and Gender columns (and encode them like shown in the previous task) into the hospitalization1 dataset.

From The EDA of the GeneralData Dataset, we can see that the common ages are 70 and 80 years old, and a few were 110.

We cleaned the missing values assuming the data wasn't cleaned by filling the nan cells with averages/common.

#### Classification Model - MLP Batch normalization

We used an MLP NN model with dropout and batch normalization. It consists of 3 layers with ReLU. In future approach we'll try to add softmax to see its reaction.
Training was with Crossentropy as the loss critera and with Adam Optimizer. The values are 100 epochs, 0.01 learning rate and dropout 0.5 with 64 hidden layers and 80-20 data split.

**Conclusions**:

1. Model Performance:

   - The final loss after 100 epochs is 1.0736, indicating that the model has not fully converged to an optimal solution, as the loss remains relatively high.
   - The accuracy of 33.94% is low, suggesting the model is struggling to generalize well to the test set.
   - The classification report shows that precision, recall, and F1-scores for all three classes are below average:
     - Class "0" (Short) has a recall of 0.45, which is the highest among all the classes, but still low in terms of precision (0.34) and F1-score (0.39).
     - Class "1" (Medium) performs the worst in terms of both precision (0.26) and recall (0.28).
     - Class "2" (Long) has slightly better precision (0.44), but lower recall (0.29), reflecting an imbalance in how well the model detects this class.
   - Overall, the macro average F1-score is 0.33, which further highlights that the model does not capture the patterns well for any class.

2. Class Distribution:
   - The actual class distribution is relatively balanced, but there is a noticeable difference between the predicted class distribution and the actual one. Specifically:
     - The model tends to over-predict "Short" class instances while under-predicting "Medium" and "Long" classes. This is evident from the skew in the predicted distribution plot.
3. Confusion Matrix Insights:
   - The confusion matrix highlights that the model is often confusing "Medium" and "Long" classes with "Short."
   - For example, for the "Short" true label, 67 instances are correctly classified, but 51 are misclassified as "Medium" and 32 as "Long."
   - Similarly, a large number of "Medium" and "Long" instances are also being misclassified as "Short" (65 and 66 instances, respectively).
   - This suggests that the model might be overfitting to features that correlate more with the "Short" class while failing to adequately distinguish between the other two classes.

#### Key Takeaways:

- **Underfitting**: The relatively high loss and poor classification metrics point towards underfitting. The model might not be complex enough to capture the patterns in the data, or there might be a lack of sufficient feature extraction and representation.
- **Class Imbalance in Predictions**: The predicted class distribution is skewed towards the "Short" class, implying that the model may be biased towards certain features that over-represent this class.

### **Task 38** - Dimension Reducer for Hospitalization2

The objective of this task is to reduce the dimension of the data using different techniques in hospitalization2 and test its efficiency. we used the same aprroach from the previous task but with a PCA Classification model to seek the hospitalization duration.

#### Classification Model - PCA Classification Pipeline

We used a PCA NN model. It consists of 2 layers with ReLU.
Training was with Crossentropy as the loss critera and with Adam Optimizer. The values are 100 epochs, 0.01 learning rate and 4 dimensions with 10 hidden layers and 70-30 split.

#### Conclusions:

The neural network model showed promising results in predicting rehospitalization:

1. Model Performance:
   The model shows high performance on both the training and test sets, achieving a test accuracy of 96%, and macro average precision, recall, and F1-score of around 0.96.
   The model is highly precise, with perfect precision for class "0" (1.00) and strong performance for classes "1" (0.97) and "2" (0.91).

2. Class-Specific Performance:
   The recall for class '1' (0.86) is notably lower than the other classes. This suggests that the model might struggle to correctly identify all instances of this class, which could imply some generalization issues.
   The F1-score for class "1" is lower (0.91) compared to other classes, indicating that there is an imbalance between precision and recall for this class.

3. Potential Overfitting:
   Training loss is consistently decreasing over 100 epochs, reaching a very low value of 0.1742, which suggests that the model is fitting the training data very well.
   However, the near-perfect performance on the training set could indicate overfitting, especially if the model is not generalizing equally well on the test set.
   The model's recall for class '1' being lower hints that the model may have memorized the training set but is overfitting to certain classes or underperforming for certain test cases.

4. Loss and Accuracy Curves: The training and validation loss curves indicated convergence, but some minor overfitting may be present, as the validation loss started to diverge slightly from the training loss toward the end of training.

5. Class Imbalance:
   The dataset appears to be somewhat imbalanced, with different class distributions (458 instances of class '0', 352 of class '1', and 511 of class '2'). This imbalance can exacerbate overfitting, particularly for minority classes (e.g., class '1').

In summary, while the model achieves high accuracy and precision, especially for classes '0' and '2', the lower recall for class '1' and very low training loss may point to overfitting.
