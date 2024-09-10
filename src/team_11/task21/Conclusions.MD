# Conclusions
***
1. We decided to divide the data into 3 even time-deltas: Short, Medium , Long between hospitalizations. 
The thresholds were set according to the 33rd, 66th and 99th percentile of the data.
2. Deltas were presented as days elapsed from the first release to the 2nd admission of any patient.
3. We trained three models in order to try to find a relationship between Discharging Unit and the days elapsed between hospitalization:
   1. Linear model (MLP)
   2. Logistic (Softmax function at the end)
   3. Resnet

We saw that the MLP performed the best, but the resukts were bad (~33%) for all three models. It could be explained by underfitting - maybe the relationship is too complex for the model to learn based only on Discharging Unit.