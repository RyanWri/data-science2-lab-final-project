## Team 2 - Son and Gal

### Tasks:

| Task ID | Description                                                        | Status                                   | Progress                        |
|---------|--------------------------------------------------------------------|------------------------------------------|---------------------------------|
| 2       | Create a full Word file and Jupyter notebook   | Written when everyone finishes their tasks.                                 | Not Started                     |
| 9       | EDA of hospitalization1                                            | **Finished**                             | 100%                            |
| 20      | Connection between Doctor Type and rehospitalization  | Created Classification model, considered adjustments to the data.                             | 80% |
| 26      | Connection between age and gender to rehospitalization from 16-19   | Pending for tasks 16-18                                 | Not Started          |
| 38      | Dimension Reduction for hospitalization2     | Pending for task 20's pull request                               | 60% |
| 46      | Editing conclusions chapter                                        | Pending                              | Not Started                   |
| Unique  | PowerPoint Presentation                                            | Pending                                  | Not Started                     |


### Conclusions:
    **Task 9: **
    Translations to hebrew, multiple diagnoses - top 10 dignoses:
            Admission: 1. 78609
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
    **Task 20: **
    - Based on Doctor rank data(Senior/Not Senior), the ranks are split almost equally to 4  categories: Yes, No, ? and Depends from which date.
    - Based on the PIE chart, They're split almost equally in amount of doctors.
    - Based on Gradient Boosting Model that predicts rehospitalization based on Doctor rank, the accuracy is 50% meaniing we need Feature Extraction/Engineering in the preprocessing.
    


        
