# Conclusions
***
1. We decided to divide the data into 3 even time-deltas: Short, Medium , Long between hospitalizations. 
The thresholds were set according to the 33rd, 66th and 99th percentile of the data.
2. Deltas were presented as days elapsed from the first release to the 2nd admission of any patient.
3. We trained three models in order to try to find a relationship between Discharging Unit and the days elapsed between hospitalization:
   1. Linear model (MLP)
   2. Logistic (Softmax function at the end)
   3. Resnet

We saw that the MLP performed the best, but the results were bad (~33%) for all three models. It could be explained by underfitting - maybe the relationship is too complex for the model to learn based only on Discharging Unit.


### Conclusions from the Boxplot:

- **Rehospitalization Time vs Discharging Unit**:
  - The boxplot shows the distribution of discharging units for each rehospitalization time category (Short, Medium, Long).
  - Across all categories, the **median discharging unit** remains relatively similar, centered around unit 3.
  - No clear pattern suggests that a particular discharging unit is associated with a higher likelihood of a short, medium, or long rehospitalization time.

### Conclusions from the Countplot:

- **Rehospitalization Time Distribution by Discharging Unit**:
  - Each discharging unit has a relatively balanced distribution of rehospitalization times.
  - **Discharging Unit 1** shows a slightly higher proportion of medium rehospitalization times compared to other units.
  - **Discharging Unit 4** has a slightly higher proportion of short rehospitalization times compared to other units.
  - In general, the distribution of short, medium, and long rehospitalization times is fairly similar across all discharging units, indicating no strong association between any particular discharging unit and the length of rehospitalization time.

### Overall Insights:

- There is no strong indication that any specific discharging unit is consistently associated with shorter or longer rehospitalization times.
- The spread of discharging units across rehospitalization time categories is relatively consistent, with some minor differences in proportion for certain units. This suggests that the rehospitalization time is not heavily influenced by the discharging unit alone but could be influenced by other factors.
- The statistical significance indicates that discharging units do influence rehospitalization time, but the effect may be small and subtle, which is why it’s not immediately apparent in the boxplots or countplots.
Statistical tests like ANOVA detect even small differences in the means across groups, while visualizations like boxplots are more focused on the spread and distribution.