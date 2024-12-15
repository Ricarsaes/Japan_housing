# Real Estate Price Analysis and Prediction Project

## **1. Data Collection and Preparation**
- Real estate price data was collected from various prefectures in Japan.
- Data from different files was combined into a single Pandas DataFrame.
- Initial data exploration (EDA) was conducted to understand the dataset's characteristics.
- Data cleaning was performed by handling missing values and limiting outliers.
- Data was prepared for modeling by encoding categorical variables and imputing missing values using the mean or KNN imputation methods (this latter approach gave me better results during the modeling stage).

## **2. Modeling**
- Three different machine learning models were trained: Random Forest, XGBoost, and Stacking Regressor.

- The best-performing models for each prefecture were selected based on the R² score.
- For the prefectures with negative score, the approach is to train a model with the data of its region (all prefectures are grouped into 8 regions in Japan).
- Also in the Appendix, a time series cross-validation strategy was attemped to evaluate model performance.
- Trained models were saved using `joblib` for future use.

## **3. Visualization**
- Model performance across prefectures was visualized using bar charts.
- Heatmaps were created to display average property prices by region.

## **4. Prediction**
- Trained models and imputers were loaded.
- Test data was prepared by imputing missing values.
- Predictions were made using the loaded models.

## **Project Files**
The files in this repository include:
- Jupyter notebooks detailing the steps for data preprocessing, modeling, and evaluation.
- Scripts for loading and using trained models for prediction.

## **Usage**
To use these models for prediction, follow the steps described in the `tradeprice_prefecture_prediction.py` file. This file provides:
- A function to load models and imputers.
- A function to make predictions on a Pandas DataFrame.
- The binary files of the model are provided through an external link due to their size (and good practices to keep repositories lightweight and avoid data).

## **Potential Next Steps**
 - Feature Engineering: New features could be created from the existing ones to improve the model's accuracy. For example, a variable indicating the distance to the nearest train station or a variable representing population density in the area could be added.

 - Hyperparameter Tuning: The hyperparameters of the machine learning models could be fine-tuned to further enhance their performance. Techniques such as grid search (with Time Series Cross Validation) or Bayesian optimization could be used to find the best hyperparameters.

 - Testing Different Models: Additional machine learning models, such as neural networks (GRU or sequence models suited for time series), could be tested to determine if they outperform the models used in this project.

 - Incorporating External Data: External data, such as demographic or economic information, could be incorporated to further improve the model's accuracy.
