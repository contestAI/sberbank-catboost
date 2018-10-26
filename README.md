# Sberbank Data Science Journey 2018: CatBoost Baseline
CatBoost Baseline [SDSJ 2018 AutoML](http://sdsj.sberbank.ai/).

Benefits: 
* CatBoost is optimized for work with categorical features: strings and ids are processed automatically; many new combinations of features are tested automatically during the training. 
* Model has evaluation on a hold-out dataset during the training. 
* Directly control the memory resources available for the model.  
* Model optimizes directly for RMSE.
* Model automatically adjusts the number of iterations to train depending on the size of the data. 
* Model has early stopping and overfitting detectors. 
* Tackle the class imbalance. 
* Solution passes all 8 public tests. 
* Evaluate locally RMSE and AUC on your datasets. 
* Hyperparameter tuning using hyperopt. 
* CatBoost is well documented and has fast support from Yandex team. 
