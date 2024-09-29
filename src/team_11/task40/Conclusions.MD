* We decided to remove the "unknown" category (represented as '?' or any equivalent) from the analysis because its presence was unbalanced and disproportionately affected the results. Including this category introduced bias, as it did not represent a clearly defined group of professionals. By excluding the "unknown" data, we aim to ensure that the analysis more accurately reflects the contributions and workload distribution among the clearly defined doctor categories: N (young doctors), M (advanced doctors), and Y (senior doctors).
* Young Doctors ('N') Underperform: Young doctors represent a large part of the workforce but contribute to fewer discharges. This may reflect less involvement in decision-making, possibly due to inexperience or reliance on senior doctors for complex cases.

* Advanced Doctors ('M') Handle More Discharges: Advanced doctors, though fewer in number, are responsible for a disproportionately high number of discharges. This suggests they are transitioning into greater responsibilities within the department.

* Senior Doctors ('Y') Play a Key Role in Complex Cases: Senior doctors, while fewer, contribute significantly to discharge decisions, likely handling more complex cases and finalizing critical discharge decisions.

#### 1. Distribution Plot (Histogram with KDE):
- **Purpose**: This plot shows the **distribution of total releases** across all days. It helps us understand how the number of total releases varies day-to-day.
- **Key Insights**:
  - The distribution appears approximately **normal** (bell-shaped), indicating that most days have a total release count concentrated around the mean.
  - The **KDE (Kernel Density Estimate)** curve smooths the histogram to give a continuous representation of the underlying distribution.
  - The **peaks** around the center suggest that most days have a total release count between **250 and 300**

#### 2. Boxplot:
- **Purpose**: The boxplot is used to visualize the **spread** and **outliers** in the total release data.
- **Key Insights**:
  - The **box** represents the **interquartile range (IQR)**, which contains the middle 50% of the data (between the 25th and 75th percentiles).
  - The **line inside the box** represents the **median** total releases, which is around **275**.
  - There are some **outliers** below the box (represented as individual points), indicating that on a few days, the number of total releases was significantly lower than usual.
  
#### 3. Time Series Plot:
- **Purpose**: This plot shows the variation in **total releases over time**, giving us a temporal view of the data.
- **Key Insights**:
  - There is a high level of variability in total releases on a day-to-day basis, but no clear upward or downward trend over time.
  - The total releases range from about **150 to 400** throughout the time period, with the majority of days clustering between **250 and 300**.
  - This plot highlights the day-to-day **fluctuations** in total releases but does not indicate any long-term patterns.

#### Key Observations:ס;
1. **Positive Correlation**:
   - There is a **positive correlation** between Doctor Count and Total Releases. As the number of doctors increases, the number of total releases also tends to increase.
   - However, the relationship is not perfect, as evidenced by the spread of the points around the regression line.

2. **Goodness of Fit**:
   - The **R-squared (R²) score** shown in the plot (0.54) indicates that about 54% of the variability in the total releases can be explained by the doctor count.
   - While this suggests a moderate fit, there is still a significant amount of unexplained variance, indicating that other factors may influence total releases.

3. **Model Error**:
   - The **Mean Squared Error (MSE)** value provides an estimate of the average squared difference between the actual and predicted values.
   - A lower MSE indicates better model performance, but in this case, the MSE of 1227 suggests that there are some large residuals, meaning the model's predictions are not always very close to the actual values.

### Conclusion:
- The model shows that there is a clear **positive relationship** between the number of doctors and total releases, indicating that as more doctors are available, the number of total releases tends to increase.
- The **R-squared (R²) score** of 0.54 suggests that the model captures about **54%** of the variability in the total releases, but **46% of the variability** remains unexplained, meaning there are likely **additional factors** that influence total releases that are not included in the model.
- The relatively high **Mean Squared Error (MSE)** value of 1227 suggests that while the model is capturing the general trend, there are **errors** in the predictions, possibly due to the limited feature set (only Doctor Count being used).
- To improve the model's performance, you could consider incorporating **additional features** (e.g., patient demographics, case complexity, time of year) that may have an impact on the total number of releases.


### Conclusions from the Plot:

#### 1. Random Distribution of Residuals:
- The residuals are randomly scattered around the red dashed line (zero residual line), indicating that there is **no clear pattern**. This suggests that the model does not have significant systematic errors and is likely **unbiased**.
- A random pattern means that the model’s errors are mostly due to **noise** and not due to a poor fit, which is a **good sign**.

#### 2. Homoscedasticity (Constant Variance of Errors):
- The spread of the residuals appears relatively **consistent across the range** of predicted values. There is no clear funnel-shaped pattern (which would indicate increasing or decreasing variance).
- This suggests that the model likely satisfies the assumption of **homoscedasticity** (constant variance of residuals), which is important for the validity of many regression models.

#### 3. No Clear Outliers:
- While there are some points with larger residuals, there doesn’t appear to be any extreme **outliers** that could significantly affect the model's fit.
- This is a **positive sign**, suggesting that the model is not being heavily influenced by a few outliers.

#### 4. Model Fit:
- Since the residuals are centered around zero and appear to be **symmetrically distributed**, we can conclude that the model is generally **fitting the data well**.
- However, there is still some spread, indicating that there is **room for improvement** in the accuracy of the model.
