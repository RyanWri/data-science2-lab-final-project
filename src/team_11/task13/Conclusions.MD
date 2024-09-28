# Conclusions 
***
1. The top 3 doctors who have released the most patients in average are:
    * Doctor 7916
    * Doctor 7917
    * Doctor 9399
2. Most doctors haven't released more than 2 patients in average.
3. The data contains 7.78% outliers, higher than the upper bound for Amount Of Releases (which is 3.5)
4. Since the dataset is large enough (above 100,000 samples) and the outliers are only 7.78% of it, we tried to think if they are meaningful to the dataset and have the ability to skew a model which is trying to train on this data. We came to a conclusion that these outliers hold a meaningful data (Days when the doctors released a significant amount of patients), and therefore should not be dropped or manipulated.
5. From the distribution of the Amount Of Releases we can see that:
   * Right-Skewed Distribution: The distribution is heavily skewed to the right, with most of the values concentrated around the lower end (closer to 1). This suggests that the majority of the data points have a relatively low average amount of releases.

   * Heavy Tail: The distribution has a long tail extending toward higher values. This indicates that while most of the data points are small (closer to 1), there are some instances with significantly higher values (greater than 2 or 3). These could be outliers or rare cases where a much higher average amount of releases is observed.

   *  High Density Around Low Values: The density peaks sharply around the lower values (close to 1), implying that the most common average amount of releases is relatively low. This might suggest that, in general, the typical amount of releases is consistently low, with only a few exceptions.