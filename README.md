# data-drift-simulation

This is a simulation of a Data Drift.<br>
We have a dataset of 90,000 rows and from 2010 to 2021. So, to simulate a data drift, we took the rows from
2010 -> 2017, let's call it df_A.

### trainig Model 1 on df_A

We build a model on the df_A data. And, logged and versioned this model on ML Flow<br><br>
**Model 1 metrics on df_A:**<br>
![Model 1 metrics](images/model_1_metrics.png)

### Testing Model 1 on df_B (2018-2019):

![Model 1 on df_B](images/data_b_drift.png) <br>
As we can see, we got an error, because the data drifted. and now our model is not wotking on the new years.<br>
now we need to re-train our model (Model 2) on the new unseen data.

### Trainig Model 2 on df_A_B (2010-2019):

![Model 2 smetrics](images/model_2_metrics.png) <br>
We see after re-training the model is working again.

### Testing Model 2 on df_C (2020-2021):

_r2: 0.7276983406662799_ <br>
_rmse: 67429.97941184546_ <br>
we don't notice a huge data drift here. anyway we will re-train the model (Model 3) on the whole data.

### Trainig Model 3 on df_A_B_C (2010-2021):

![Model 3 smetrics](images/model_3_metrics.png) <br>
We did some improvments.
