# UCI_Energy_Prediction_Model
## Objective:
The objective of the study is to understand the time series dataset for UCI energy use of Appliances along with
various factors of impact like Temperature and Relative Humidity of internal environment, Pressure, Windspeed,
Visibility of external environment and develop models for prediction for energy use. The problem comes under
regression analysis and hence regression models will be constructed.

![Correlation Analysis](https://github.com/Eva86271/UCI_Energy_Prediction_Model/blob/main/images_UCI/output_11_0.png)

Based on DateTime indexing that was performed on the dataset for time series analysis, more features like Hours,
Minutes, Day of the Year, Day of the Week, Year, and Month were constructed from data and added to it, in order to
understand the behaviour of the energy use as per the time and date. The new dataset contained 33 features.
The data were seen to be varied with the newly constructed features like Hours and Weekday, which were plotted
later on. While understanding the features it was found that some features tended to be distributed normally in
consideration to Central Limit Theorem, while some do show bimodal distribution and skewness hence to attend a
uniformness, normalization through MinMaxScaler was carried out, so that all features do fall under the bell-shaped
curve. Not much of outliers were noted. However, we did notice some outliers in the target variable considering an
hourly and daily basis consumption distribution . The energy consumption does show a variation with an hour and
Weekday as plotted in the report. (Refer to the attached Jupyter notebooks in the zip file for boxplot and histogram
plotting of features for outlier detection and normalization study).

