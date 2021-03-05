# UCI_Energy_Prediction_Model
## Objective:
The objective of the study is to understand the time series dataset for UCI energy use of Appliances along with
various factors of impact like Temperature and Relative Humidity of internal environment, Pressure, Windspeed,
Visibility of external environment and develop models for prediction for energy use. The problem comes under
regression analysis and hence regression models will be constructed.

Based on DateTime indexing that was performed on the dataset for time series analysis, more features like Hours,
Minutes, Day of the Year, Day of the Week, Year, and Month were constructed from data and added to it, in order to
understand the behaviour of the energy use as per the time and date. The new dataset contained 33 features.
The data were seen to be varied with the newly constructed features like Hours and Weekday, which were plotted
later on. While understanding the features it was found that some features tended to be distributed normally in
consideration to Central Limit Theorem, while some do show bimodal distribution and skewness hence to attend a
uniformness, normalization through MinMaxScaler was carried out, so that all features do fall under the bell-shaped
curve. Not much of outliers were noted. However, we did notice some outliers in the target variable considering an
hourly and daily basis consumption distribution . The energy consumption does show a variation with an hour and
Weekday as plotted in the report.

## Correlational Analysis:
![Correlation Analysis](https://github.com/Eva86271/UCI_Energy_Prediction_Model/blob/main/images_UCI/output_11_0.png)
## Data Exploration:
Various data visualition regarding the variables in data are analysed
-Showcase behaviour between T1 and Target Energy
![Showcase behaviour between T1 and Target Energy](https://github.com/Eva86271/UCI_Energy_Prediction_Model/blob/main/images_UCI/output_16_0.png)
-Showcase behaviour between T6 and Target Energy
![Showcase behaviour between T6 and Target Energy](https://github.com/Eva86271/UCI_Energy_Prediction_Model/blob/main/images_UCI/output_16_10.png)
-Showcase behaviour between Hours of Consumption and Target Energy
![Showcase behaviour between Hours of Consumption and Target Energy](https://github.com/Eva86271/UCI_Energy_Prediction_Model/blob/main/images_UCI/output_27_1.png)
-Showcase Variation between Weekdays and Target Energy
![Showcase Variation between Weekdays and Target Energy](https://github.com/Eva86271/UCI_Energy_Prediction_Model/blob/main/images_UCI/output_8_0.png)

## Feature Selection:
Feature Selection was done by Select K Best methods:
![Feature Selection](https://github.com/Eva86271/UCI_Energy_Prediction_Model/blob/main/images_UCI/output_25_1.JPG)
## Model Development and Evaluation :
As the target variable does not bear a linear relation with the features as plotted in the scatter plots in the Jupyter
Note and the Lasso Linear Regression Model does show a large variation estimating a large error in Linear Model
Prediction hence, we will be choosing a non-linear model prediction approach. To initiate with, the nonlinear model like Gradient Boost, and DecisionTree algorithms were chosen to build the Regressor Model.
The dataset has been cross-validated with 80 % of training data and 20 % of validating data, the data has been prepared and passed to each model in the GridSearchCV with cv=5. The GridSearchCV iterations ensure that no data has been left out for the training set/validating set.





