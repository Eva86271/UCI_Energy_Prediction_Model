```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn import preprocessing, feature_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression,LassoCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split,GridSearchCV
import math
sns.set()

UCI_data=pd.read_csv("C:/Users/subha/OneDrive/Desktop/UCI-electricity/UCI_data.csv",index_col=0,parse_dates=True)
UCI_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>T1</th>
      <th>RH_1</th>
      <th>T2</th>
      <th>RH_2</th>
      <th>T3</th>
      <th>RH_3</th>
      <th>T4</th>
      <th>RH_4</th>
      <th>T5</th>
      <th>RH_5</th>
      <th>...</th>
      <th>RH_9</th>
      <th>T_out</th>
      <th>Press_mm_hg</th>
      <th>RH_out</th>
      <th>Windspeed</th>
      <th>Visibility</th>
      <th>Tdewpoint</th>
      <th>rv1</th>
      <th>rv2</th>
      <th>TARGET_energy</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-04-19 20:30:00</th>
      <td>22.200000</td>
      <td>39.500000</td>
      <td>20.566667</td>
      <td>37.656667</td>
      <td>22.230000</td>
      <td>37.030000</td>
      <td>22.318571</td>
      <td>36.610000</td>
      <td>20.633333</td>
      <td>62.166667</td>
      <td>...</td>
      <td>33.90</td>
      <td>9.70</td>
      <td>766.100000</td>
      <td>65.5</td>
      <td>3.500000</td>
      <td>40.000000</td>
      <td>3.350000</td>
      <td>24.061869</td>
      <td>24.061869</td>
      <td>60</td>
    </tr>
    <tr>
      <th>2016-03-05 04:40:00</th>
      <td>20.356667</td>
      <td>37.126667</td>
      <td>17.566667</td>
      <td>40.230000</td>
      <td>20.890000</td>
      <td>37.663333</td>
      <td>18.700000</td>
      <td>36.260000</td>
      <td>18.463333</td>
      <td>43.560000</td>
      <td>...</td>
      <td>41.09</td>
      <td>0.30</td>
      <td>740.333333</td>
      <td>99.0</td>
      <td>1.000000</td>
      <td>41.333333</td>
      <td>0.100000</td>
      <td>4.622052</td>
      <td>4.622052</td>
      <td>50</td>
    </tr>
    <tr>
      <th>2016-03-14 12:40:00</th>
      <td>20.926667</td>
      <td>38.790000</td>
      <td>21.100000</td>
      <td>35.526667</td>
      <td>21.600000</td>
      <td>36.290000</td>
      <td>21.000000</td>
      <td>34.826667</td>
      <td>18.100000</td>
      <td>46.126667</td>
      <td>...</td>
      <td>38.76</td>
      <td>4.40</td>
      <td>768.466667</td>
      <td>72.0</td>
      <td>6.000000</td>
      <td>22.666667</td>
      <td>-0.266667</td>
      <td>5.635898</td>
      <td>5.635898</td>
      <td>80</td>
    </tr>
    <tr>
      <th>2016-01-22 15:30:00</th>
      <td>18.290000</td>
      <td>38.900000</td>
      <td>17.290000</td>
      <td>39.260000</td>
      <td>18.390000</td>
      <td>39.326667</td>
      <td>16.100000</td>
      <td>38.790000</td>
      <td>16.100000</td>
      <td>47.700000</td>
      <td>...</td>
      <td>39.20</td>
      <td>3.35</td>
      <td>760.600000</td>
      <td>82.0</td>
      <td>5.500000</td>
      <td>41.000000</td>
      <td>0.500000</td>
      <td>49.216445</td>
      <td>49.216445</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2016-02-10 00:40:00</th>
      <td>22.290000</td>
      <td>42.333333</td>
      <td>21.600000</td>
      <td>40.433333</td>
      <td>22.666667</td>
      <td>43.363333</td>
      <td>19.100000</td>
      <td>40.900000</td>
      <td>19.290000</td>
      <td>50.745000</td>
      <td>...</td>
      <td>43.73</td>
      <td>3.20</td>
      <td>738.900000</td>
      <td>88.0</td>
      <td>7.333333</td>
      <td>56.000000</td>
      <td>1.400000</td>
      <td>47.617579</td>
      <td>47.617579</td>
      <td>60</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
np.shape(UCI_data)
```




    (19735, 27)




```python
UCI_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 19735 entries, 2016-04-19 20:30:00 to 2016-02-16 10:20:00
    Data columns (total 27 columns):
    T1               19735 non-null float64
    RH_1             19735 non-null float64
    T2               19735 non-null float64
    RH_2             19735 non-null float64
    T3               19735 non-null float64
    RH_3             19735 non-null float64
    T4               19735 non-null float64
    RH_4             19735 non-null float64
    T5               19735 non-null float64
    RH_5             19735 non-null float64
    T6               19735 non-null float64
    RH_6             19735 non-null float64
    T7               19735 non-null float64
    RH_7             19735 non-null float64
    T8               19735 non-null float64
    RH_8             19735 non-null float64
    T9               19735 non-null float64
    RH_9             19735 non-null float64
    T_out            19735 non-null float64
    Press_mm_hg      19735 non-null float64
    RH_out           19735 non-null float64
    Windspeed        19735 non-null float64
    Visibility       19735 non-null float64
    Tdewpoint        19735 non-null float64
    rv1              19735 non-null float64
    rv2              19735 non-null float64
    TARGET_energy    19735 non-null int64
    dtypes: float64(26), int64(1)
    memory usage: 4.2 MB
    


```python
UCI_data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>T1</th>
      <th>RH_1</th>
      <th>T2</th>
      <th>RH_2</th>
      <th>T3</th>
      <th>RH_3</th>
      <th>T4</th>
      <th>RH_4</th>
      <th>T5</th>
      <th>RH_5</th>
      <th>...</th>
      <th>RH_9</th>
      <th>T_out</th>
      <th>Press_mm_hg</th>
      <th>RH_out</th>
      <th>Windspeed</th>
      <th>Visibility</th>
      <th>Tdewpoint</th>
      <th>rv1</th>
      <th>rv2</th>
      <th>TARGET_energy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>19735.000000</td>
      <td>19735.000000</td>
      <td>19735.000000</td>
      <td>19735.000000</td>
      <td>19735.000000</td>
      <td>19735.000000</td>
      <td>19735.000000</td>
      <td>19735.000000</td>
      <td>19735.000000</td>
      <td>19735.000000</td>
      <td>...</td>
      <td>19735.000000</td>
      <td>19735.000000</td>
      <td>19735.000000</td>
      <td>19735.000000</td>
      <td>19735.000000</td>
      <td>19735.000000</td>
      <td>19735.000000</td>
      <td>19735.000000</td>
      <td>19735.000000</td>
      <td>19735.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>21.686571</td>
      <td>40.259739</td>
      <td>20.341219</td>
      <td>40.420420</td>
      <td>22.267611</td>
      <td>39.242500</td>
      <td>20.855335</td>
      <td>39.026904</td>
      <td>19.592106</td>
      <td>50.949283</td>
      <td>...</td>
      <td>41.552401</td>
      <td>7.411665</td>
      <td>755.522602</td>
      <td>79.750418</td>
      <td>4.039752</td>
      <td>38.330834</td>
      <td>3.760707</td>
      <td>24.988033</td>
      <td>24.988033</td>
      <td>101.496833</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.606066</td>
      <td>3.979299</td>
      <td>2.192974</td>
      <td>4.069813</td>
      <td>2.006111</td>
      <td>3.254576</td>
      <td>2.042884</td>
      <td>4.341321</td>
      <td>1.844623</td>
      <td>9.022034</td>
      <td>...</td>
      <td>4.151497</td>
      <td>5.317409</td>
      <td>7.399441</td>
      <td>14.901088</td>
      <td>2.451221</td>
      <td>11.794719</td>
      <td>4.194648</td>
      <td>14.496634</td>
      <td>14.496634</td>
      <td>104.380829</td>
    </tr>
    <tr>
      <th>min</th>
      <td>16.790000</td>
      <td>27.023333</td>
      <td>16.100000</td>
      <td>20.463333</td>
      <td>17.200000</td>
      <td>28.766667</td>
      <td>15.100000</td>
      <td>27.660000</td>
      <td>15.330000</td>
      <td>29.815000</td>
      <td>...</td>
      <td>29.166667</td>
      <td>-5.000000</td>
      <td>729.300000</td>
      <td>24.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>-6.600000</td>
      <td>0.005322</td>
      <td>0.005322</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>20.760000</td>
      <td>37.333333</td>
      <td>18.790000</td>
      <td>37.900000</td>
      <td>20.790000</td>
      <td>36.900000</td>
      <td>19.530000</td>
      <td>35.530000</td>
      <td>18.277500</td>
      <td>45.400000</td>
      <td>...</td>
      <td>38.500000</td>
      <td>3.666667</td>
      <td>750.933333</td>
      <td>70.333333</td>
      <td>2.000000</td>
      <td>29.000000</td>
      <td>0.900000</td>
      <td>12.497889</td>
      <td>12.497889</td>
      <td>50.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>21.600000</td>
      <td>39.656667</td>
      <td>20.000000</td>
      <td>40.500000</td>
      <td>22.100000</td>
      <td>38.530000</td>
      <td>20.666667</td>
      <td>38.400000</td>
      <td>19.390000</td>
      <td>49.090000</td>
      <td>...</td>
      <td>40.900000</td>
      <td>6.916667</td>
      <td>756.100000</td>
      <td>83.666667</td>
      <td>3.666667</td>
      <td>40.000000</td>
      <td>3.433333</td>
      <td>24.897653</td>
      <td>24.897653</td>
      <td>60.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>22.600000</td>
      <td>43.066667</td>
      <td>21.500000</td>
      <td>43.260000</td>
      <td>23.290000</td>
      <td>41.760000</td>
      <td>22.100000</td>
      <td>42.156667</td>
      <td>20.619643</td>
      <td>53.663333</td>
      <td>...</td>
      <td>44.338095</td>
      <td>10.408333</td>
      <td>760.933333</td>
      <td>91.666667</td>
      <td>5.500000</td>
      <td>40.000000</td>
      <td>6.566667</td>
      <td>37.583769</td>
      <td>37.583769</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>26.260000</td>
      <td>63.360000</td>
      <td>29.856667</td>
      <td>56.026667</td>
      <td>29.236000</td>
      <td>50.163333</td>
      <td>26.200000</td>
      <td>51.090000</td>
      <td>25.795000</td>
      <td>96.321667</td>
      <td>...</td>
      <td>53.326667</td>
      <td>26.100000</td>
      <td>772.300000</td>
      <td>100.000000</td>
      <td>14.000000</td>
      <td>66.000000</td>
      <td>15.500000</td>
      <td>49.996530</td>
      <td>49.996530</td>
      <td>1110.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 27 columns</p>
</div>




```python
#pd.to_datetime(UCI_data.index)
```


```python
UCI_dates=UCI_data.sort_index()
```


```python
plt.figure()
UCI_data.hist(figsize=(20,20))
plt.show()
```


    <Figure size 432x288 with 0 Axes>



![png](output_6_1.png)



```python
UCI_data['Hours']=UCI_data.index.hour
UCI_data['Weekday']=UCI_data.index.weekday
UCI_data['Month']=UCI_data.index.month
UCI_data['Day']=UCI_data.index.day
UCI_data['Minutes']=UCI_data.index.minute
UCI_data['Years']=UCI_data.index.year
```


```python
ax = sns.boxplot(x="Weekday", y="TARGET_energy", data=UCI_data)
```


![png](output_8_0.png)



```python
ax = sns.boxplot(x="Hours", y="TARGET_energy", data=UCI_data)
```


![png](output_9_0.png)



```python
np.shape(UCI_data)
```




    (19735, 33)




```python
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(UCI_data.corr(),vmin=-1, vmax=1, center=0,cmap="YlOrRd",annot=True, fmt='.2f')
plt.show()
```


![png](output_11_0.png)



```python

figure,ax=plt.subplots(figsize=(5,5))
UCI_data['TARGET_energy'].resample('M').sum().plot(kind='bar',color='orange')
x=np.arange(5)
ax.set_xticks(x)
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May'])
plt.show()

```


![png](output_12_0.png)



```python

```


```python
UCI_data=UCI_data.resample('H').mean()
```


```python
#Extracting the target and predictors 
target=UCI_data['TARGET_energy']
predictors=UCI_data.drop('TARGET_energy',axis=1)
```


```python
#Visualizing between every column relation
for i in predictors.columns:
    sns.scatterplot(predictors[i],target,color="red")
    plt.title("Plot for "+i+"with Target Energy")
    plt.ylabel("TARGET_energy")
    plt.xlabel(i)
    plt.show()
    
```


![png](output_16_0.png)



![png](output_16_1.png)



![png](output_16_2.png)



![png](output_16_3.png)



![png](output_16_4.png)



![png](output_16_5.png)



![png](output_16_6.png)



![png](output_16_7.png)



![png](output_16_8.png)



![png](output_16_9.png)



![png](output_16_10.png)



![png](output_16_11.png)



![png](output_16_12.png)



![png](output_16_13.png)



![png](output_16_14.png)



![png](output_16_15.png)



![png](output_16_16.png)



![png](output_16_17.png)



![png](output_16_18.png)



![png](output_16_19.png)



![png](output_16_20.png)



![png](output_16_21.png)



![png](output_16_22.png)



![png](output_16_23.png)



![png](output_16_24.png)



![png](output_16_25.png)



![png](output_16_26.png)



![png](output_16_27.png)



![png](output_16_28.png)



![png](output_16_29.png)



![png](output_16_30.png)



![png](output_16_31.png)



```python
UCI_data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>T1</th>
      <th>RH_1</th>
      <th>T2</th>
      <th>RH_2</th>
      <th>T3</th>
      <th>RH_3</th>
      <th>T4</th>
      <th>RH_4</th>
      <th>T5</th>
      <th>RH_5</th>
      <th>...</th>
      <th>Tdewpoint</th>
      <th>rv1</th>
      <th>rv2</th>
      <th>TARGET_energy</th>
      <th>Hours</th>
      <th>Weekday</th>
      <th>Month</th>
      <th>Day</th>
      <th>Minutes</th>
      <th>Years</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-01-11 17:00:00</th>
      <td>19.890000</td>
      <td>46.502778</td>
      <td>19.200000</td>
      <td>44.626528</td>
      <td>19.790000</td>
      <td>44.897778</td>
      <td>18.932778</td>
      <td>45.738750</td>
      <td>17.166667</td>
      <td>55.116667</td>
      <td>...</td>
      <td>5.050000</td>
      <td>26.823044</td>
      <td>26.823044</td>
      <td>90.000000</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>2016-01-11 18:00:00</th>
      <td>19.897778</td>
      <td>45.879028</td>
      <td>19.268889</td>
      <td>44.438889</td>
      <td>19.770000</td>
      <td>44.863333</td>
      <td>18.908333</td>
      <td>46.066667</td>
      <td>17.111111</td>
      <td>54.977778</td>
      <td>...</td>
      <td>4.658333</td>
      <td>22.324206</td>
      <td>22.324206</td>
      <td>228.333333</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>2016-01-11 19:00:00</th>
      <td>20.495556</td>
      <td>52.805556</td>
      <td>19.925556</td>
      <td>46.061667</td>
      <td>20.052222</td>
      <td>47.227361</td>
      <td>18.969444</td>
      <td>47.815556</td>
      <td>17.136111</td>
      <td>55.869861</td>
      <td>...</td>
      <td>4.391667</td>
      <td>33.734932</td>
      <td>33.734932</td>
      <td>198.333333</td>
      <td>19.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>2016-01-11 20:00:00</th>
      <td>20.961111</td>
      <td>48.453333</td>
      <td>20.251111</td>
      <td>45.632639</td>
      <td>20.213889</td>
      <td>47.268889</td>
      <td>19.190833</td>
      <td>49.227917</td>
      <td>17.615556</td>
      <td>74.027778</td>
      <td>...</td>
      <td>4.016667</td>
      <td>25.679642</td>
      <td>25.679642</td>
      <td>160.000000</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>2016-01-11 21:00:00</th>
      <td>21.311667</td>
      <td>45.768333</td>
      <td>20.587778</td>
      <td>44.961111</td>
      <td>20.373333</td>
      <td>46.164444</td>
      <td>19.425556</td>
      <td>47.918889</td>
      <td>18.427222</td>
      <td>69.037778</td>
      <td>...</td>
      <td>3.816667</td>
      <td>18.826274</td>
      <td>18.826274</td>
      <td>126.666667</td>
      <td>21.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>2016-01-11 22:00:00</th>
      <td>21.572222</td>
      <td>44.663333</td>
      <td>20.905556</td>
      <td>44.118889</td>
      <td>20.469444</td>
      <td>45.829444</td>
      <td>20.108889</td>
      <td>47.506667</td>
      <td>19.112917</td>
      <td>53.129306</td>
      <td>...</td>
      <td>3.741667</td>
      <td>27.143708</td>
      <td>27.143708</td>
      <td>288.333333</td>
      <td>22.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>2016-01-11 23:00:00</th>
      <td>21.531667</td>
      <td>44.271111</td>
      <td>20.934444</td>
      <td>43.712500</td>
      <td>20.317917</td>
      <td>45.695833</td>
      <td>20.909722</td>
      <td>46.551250</td>
      <td>19.275000</td>
      <td>50.923333</td>
      <td>...</td>
      <td>3.925000</td>
      <td>29.209795</td>
      <td>29.209795</td>
      <td>75.000000</td>
      <td>23.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>2016-01-12 00:00:00</th>
      <td>21.266111</td>
      <td>44.960556</td>
      <td>20.637222</td>
      <td>44.018333</td>
      <td>20.144444</td>
      <td>45.542222</td>
      <td>20.514444</td>
      <td>47.112778</td>
      <td>19.155556</td>
      <td>50.396111</td>
      <td>...</td>
      <td>4.016667</td>
      <td>26.296718</td>
      <td>26.296718</td>
      <td>158.333333</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>12.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>2016-01-12 01:00:00</th>
      <td>20.963611</td>
      <td>45.613194</td>
      <td>20.333333</td>
      <td>44.273333</td>
      <td>20.133333</td>
      <td>45.540000</td>
      <td>21.505556</td>
      <td>46.946667</td>
      <td>18.969444</td>
      <td>50.076111</td>
      <td>...</td>
      <td>3.941667</td>
      <td>28.723073</td>
      <td>28.723073</td>
      <td>176.666667</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>12.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>2016-01-12 02:00:00</th>
      <td>20.696667</td>
      <td>46.197778</td>
      <td>20.066667</td>
      <td>44.503333</td>
      <td>20.170833</td>
      <td>45.454167</td>
      <td>21.143056</td>
      <td>45.905556</td>
      <td>18.813333</td>
      <td>50.122083</td>
      <td>...</td>
      <td>3.833333</td>
      <td>17.282387</td>
      <td>17.282387</td>
      <td>45.000000</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>12.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 33 columns</p>
</div>




```python
UCI_data.reset_index(inplace=True)
UCI_data.drop('date',axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>T1</th>
      <th>RH_1</th>
      <th>T2</th>
      <th>RH_2</th>
      <th>T3</th>
      <th>RH_3</th>
      <th>T4</th>
      <th>RH_4</th>
      <th>T5</th>
      <th>RH_5</th>
      <th>...</th>
      <th>Tdewpoint</th>
      <th>rv1</th>
      <th>rv2</th>
      <th>TARGET_energy</th>
      <th>Hours</th>
      <th>Weekday</th>
      <th>Month</th>
      <th>Day</th>
      <th>Minutes</th>
      <th>Years</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19.890000</td>
      <td>46.502778</td>
      <td>19.200000</td>
      <td>44.626528</td>
      <td>19.790000</td>
      <td>44.897778</td>
      <td>18.932778</td>
      <td>45.738750</td>
      <td>17.166667</td>
      <td>55.116667</td>
      <td>...</td>
      <td>5.050000</td>
      <td>26.823044</td>
      <td>26.823044</td>
      <td>90.000000</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>19.897778</td>
      <td>45.879028</td>
      <td>19.268889</td>
      <td>44.438889</td>
      <td>19.770000</td>
      <td>44.863333</td>
      <td>18.908333</td>
      <td>46.066667</td>
      <td>17.111111</td>
      <td>54.977778</td>
      <td>...</td>
      <td>4.658333</td>
      <td>22.324206</td>
      <td>22.324206</td>
      <td>228.333333</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20.495556</td>
      <td>52.805556</td>
      <td>19.925556</td>
      <td>46.061667</td>
      <td>20.052222</td>
      <td>47.227361</td>
      <td>18.969444</td>
      <td>47.815556</td>
      <td>17.136111</td>
      <td>55.869861</td>
      <td>...</td>
      <td>4.391667</td>
      <td>33.734932</td>
      <td>33.734932</td>
      <td>198.333333</td>
      <td>19.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20.961111</td>
      <td>48.453333</td>
      <td>20.251111</td>
      <td>45.632639</td>
      <td>20.213889</td>
      <td>47.268889</td>
      <td>19.190833</td>
      <td>49.227917</td>
      <td>17.615556</td>
      <td>74.027778</td>
      <td>...</td>
      <td>4.016667</td>
      <td>25.679642</td>
      <td>25.679642</td>
      <td>160.000000</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21.311667</td>
      <td>45.768333</td>
      <td>20.587778</td>
      <td>44.961111</td>
      <td>20.373333</td>
      <td>46.164444</td>
      <td>19.425556</td>
      <td>47.918889</td>
      <td>18.427222</td>
      <td>69.037778</td>
      <td>...</td>
      <td>3.816667</td>
      <td>18.826274</td>
      <td>18.826274</td>
      <td>126.666667</td>
      <td>21.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3285</th>
      <td>25.544444</td>
      <td>46.638889</td>
      <td>26.421369</td>
      <td>41.205054</td>
      <td>28.397778</td>
      <td>41.160000</td>
      <td>24.666667</td>
      <td>45.883889</td>
      <td>22.890000</td>
      <td>53.052222</td>
      <td>...</td>
      <td>13.475000</td>
      <td>27.553946</td>
      <td>27.553946</td>
      <td>103.333333</td>
      <td>14.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>27.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>3286</th>
      <td>25.500000</td>
      <td>45.677639</td>
      <td>26.154762</td>
      <td>41.041238</td>
      <td>28.240000</td>
      <td>40.306667</td>
      <td>24.694444</td>
      <td>45.270000</td>
      <td>23.007500</td>
      <td>52.368611</td>
      <td>...</td>
      <td>13.258333</td>
      <td>25.429025</td>
      <td>25.429025</td>
      <td>76.666667</td>
      <td>15.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>27.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>3287</th>
      <td>25.487778</td>
      <td>46.250000</td>
      <td>26.000000</td>
      <td>41.736190</td>
      <td>27.953333</td>
      <td>40.607778</td>
      <td>24.700000</td>
      <td>45.476667</td>
      <td>23.150000</td>
      <td>52.094444</td>
      <td>...</td>
      <td>13.283333</td>
      <td>23.229344</td>
      <td>23.229344</td>
      <td>135.000000</td>
      <td>16.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>27.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>3288</th>
      <td>25.533333</td>
      <td>46.783333</td>
      <td>25.772190</td>
      <td>42.495476</td>
      <td>27.164444</td>
      <td>41.247778</td>
      <td>24.700000</td>
      <td>45.658889</td>
      <td>23.210000</td>
      <td>52.296667</td>
      <td>...</td>
      <td>13.316667</td>
      <td>27.186003</td>
      <td>27.186003</td>
      <td>183.333333</td>
      <td>17.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>27.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>3289</th>
      <td>25.500000</td>
      <td>46.600000</td>
      <td>25.264286</td>
      <td>42.971429</td>
      <td>26.823333</td>
      <td>41.156667</td>
      <td>24.700000</td>
      <td>45.963333</td>
      <td>23.200000</td>
      <td>52.200000</td>
      <td>...</td>
      <td>13.200000</td>
      <td>34.118851</td>
      <td>34.118851</td>
      <td>440.000000</td>
      <td>18.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>27.0</td>
      <td>0.0</td>
      <td>2016.0</td>
    </tr>
  </tbody>
</table>
<p>3290 rows × 33 columns</p>
</div>




```python
#Divinding the data into Train,Test and Validating dataset for Model Building
```


```python

UCI_predictors_train_valid,UCI_predictors_test,UCI_target_energy_train_valid,UCI_target_energy_test=train_test_split(predictors,target,test_size=0.1)
```


```python
UCI_predictors_train,UCI_predictors_valid,UCI_target_energy_train,UCI_target_energy_valid=train_test_split(UCI_predictors_train_valid,UCI_target_energy_train_valid,test_size=0.2)
```


```python
UCI_predictors_train.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>T1</th>
      <th>RH_1</th>
      <th>T2</th>
      <th>RH_2</th>
      <th>T3</th>
      <th>RH_3</th>
      <th>T4</th>
      <th>RH_4</th>
      <th>T5</th>
      <th>RH_5</th>
      <th>...</th>
      <th>Visibility</th>
      <th>Tdewpoint</th>
      <th>rv1</th>
      <th>rv2</th>
      <th>Hours</th>
      <th>Weekday</th>
      <th>Month</th>
      <th>Day</th>
      <th>Minutes</th>
      <th>Years</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-05-24 09:00:00</th>
      <td>24.100000</td>
      <td>43.466667</td>
      <td>21.986667</td>
      <td>45.938889</td>
      <td>24.986667</td>
      <td>40.317778</td>
      <td>23.500000</td>
      <td>42.307857</td>
      <td>23.100000</td>
      <td>49.618889</td>
      <td>...</td>
      <td>40.000000</td>
      <td>7.541667</td>
      <td>25.888395</td>
      <td>25.888395</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>24.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>2016-01-21 16:00:00</th>
      <td>19.862222</td>
      <td>36.831667</td>
      <td>19.218889</td>
      <td>36.175556</td>
      <td>18.532222</td>
      <td>38.073333</td>
      <td>18.422222</td>
      <td>36.816389</td>
      <td>18.017407</td>
      <td>39.738333</td>
      <td>...</td>
      <td>39.916667</td>
      <td>-2.700000</td>
      <td>33.644714</td>
      <td>33.644714</td>
      <td>16.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>21.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>2016-04-16 20:00:00</th>
      <td>22.770000</td>
      <td>44.173333</td>
      <td>20.405000</td>
      <td>44.919444</td>
      <td>24.451111</td>
      <td>39.766667</td>
      <td>21.723143</td>
      <td>39.223238</td>
      <td>20.890000</td>
      <td>47.301111</td>
      <td>...</td>
      <td>34.583333</td>
      <td>5.300000</td>
      <td>14.498622</td>
      <td>14.498622</td>
      <td>20.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>16.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>2016-04-02 08:00:00</th>
      <td>21.000000</td>
      <td>39.240000</td>
      <td>18.566667</td>
      <td>43.092222</td>
      <td>21.790000</td>
      <td>37.400000</td>
      <td>19.842778</td>
      <td>38.294722</td>
      <td>19.290000</td>
      <td>45.200000</td>
      <td>...</td>
      <td>64.583333</td>
      <td>4.916667</td>
      <td>23.777620</td>
      <td>23.777620</td>
      <td>8.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>2016-05-10 07:00:00</th>
      <td>24.856667</td>
      <td>47.118889</td>
      <td>23.138889</td>
      <td>48.457778</td>
      <td>26.479571</td>
      <td>41.675952</td>
      <td>25.066667</td>
      <td>46.138333</td>
      <td>23.828889</td>
      <td>57.174444</td>
      <td>...</td>
      <td>40.000000</td>
      <td>13.983333</td>
      <td>25.808036</td>
      <td>25.808036</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>10.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>2016-05-01 19:00:00</th>
      <td>23.213611</td>
      <td>35.012222</td>
      <td>21.388889</td>
      <td>34.842222</td>
      <td>22.790000</td>
      <td>35.020000</td>
      <td>21.510000</td>
      <td>35.306667</td>
      <td>20.886667</td>
      <td>58.883333</td>
      <td>...</td>
      <td>40.000000</td>
      <td>0.375000</td>
      <td>22.314088</td>
      <td>22.314088</td>
      <td>19.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>2016-05-13 13:00:00</th>
      <td>25.830000</td>
      <td>46.933889</td>
      <td>27.848889</td>
      <td>38.906111</td>
      <td>28.404238</td>
      <td>42.269619</td>
      <td>25.783571</td>
      <td>43.815397</td>
      <td>24.155556</td>
      <td>49.221111</td>
      <td>...</td>
      <td>28.416667</td>
      <td>12.975000</td>
      <td>19.730881</td>
      <td>19.730881</td>
      <td>13.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>13.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>2016-03-02 13:00:00</th>
      <td>21.053333</td>
      <td>38.540000</td>
      <td>20.364444</td>
      <td>36.384444</td>
      <td>20.677778</td>
      <td>38.741111</td>
      <td>21.053333</td>
      <td>36.064444</td>
      <td>18.177778</td>
      <td>46.640000</td>
      <td>...</td>
      <td>40.000000</td>
      <td>0.966667</td>
      <td>30.204284</td>
      <td>30.204284</td>
      <td>13.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>2016-04-15 04:00:00</th>
      <td>21.951111</td>
      <td>42.398889</td>
      <td>19.730000</td>
      <td>45.694444</td>
      <td>23.740000</td>
      <td>40.151111</td>
      <td>22.177778</td>
      <td>40.638889</td>
      <td>20.538889</td>
      <td>51.621667</td>
      <td>...</td>
      <td>28.333333</td>
      <td>8.808333</td>
      <td>25.471032</td>
      <td>25.471032</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>15.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>2016-03-25 06:00:00</th>
      <td>21.116667</td>
      <td>39.596111</td>
      <td>18.505556</td>
      <td>43.341111</td>
      <td>22.334444</td>
      <td>39.020000</td>
      <td>19.890000</td>
      <td>38.590000</td>
      <td>19.422083</td>
      <td>50.001944</td>
      <td>...</td>
      <td>39.250000</td>
      <td>5.483333</td>
      <td>23.611448</td>
      <td>23.611448</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>25.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>2016-03-22 12:00:00</th>
      <td>21.818889</td>
      <td>37.708889</td>
      <td>22.527778</td>
      <td>35.620000</td>
      <td>21.932778</td>
      <td>36.587222</td>
      <td>22.027778</td>
      <td>35.515000</td>
      <td>19.000000</td>
      <td>46.127778</td>
      <td>...</td>
      <td>32.416667</td>
      <td>2.691667</td>
      <td>38.748535</td>
      <td>38.748535</td>
      <td>12.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>22.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>2016-03-19 04:00:00</th>
      <td>21.426667</td>
      <td>37.072222</td>
      <td>18.559722</td>
      <td>40.131250</td>
      <td>21.541667</td>
      <td>36.505000</td>
      <td>19.938889</td>
      <td>35.000000</td>
      <td>18.378889</td>
      <td>46.740000</td>
      <td>...</td>
      <td>64.416667</td>
      <td>2.916667</td>
      <td>30.475813</td>
      <td>30.475813</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>19.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>2016-04-13 01:00:00</th>
      <td>22.312778</td>
      <td>41.733333</td>
      <td>19.841111</td>
      <td>44.157222</td>
      <td>24.532222</td>
      <td>39.963333</td>
      <td>22.802778</td>
      <td>38.275556</td>
      <td>21.030000</td>
      <td>54.528667</td>
      <td>...</td>
      <td>40.000000</td>
      <td>5.358333</td>
      <td>31.611628</td>
      <td>31.611628</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>13.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>2016-03-02 09:00:00</th>
      <td>20.050000</td>
      <td>39.878889</td>
      <td>18.974167</td>
      <td>40.254167</td>
      <td>20.505556</td>
      <td>38.505000</td>
      <td>21.301667</td>
      <td>37.414444</td>
      <td>18.570556</td>
      <td>47.035556</td>
      <td>...</td>
      <td>40.000000</td>
      <td>1.916667</td>
      <td>26.913230</td>
      <td>26.913230</td>
      <td>9.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>2016-04-06 22:00:00</th>
      <td>21.687778</td>
      <td>42.131111</td>
      <td>19.213333</td>
      <td>43.944444</td>
      <td>22.770000</td>
      <td>39.560000</td>
      <td>21.328889</td>
      <td>39.590000</td>
      <td>20.750000</td>
      <td>58.085556</td>
      <td>...</td>
      <td>35.416667</td>
      <td>3.583333</td>
      <td>24.844511</td>
      <td>24.844511</td>
      <td>22.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>25.0</td>
      <td>2016.0</td>
    </tr>
  </tbody>
</table>
<p>15 rows × 32 columns</p>
</div>




```python
UCI_predictors_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 2368 entries, 2016-05-24 09:00:00 to 2016-02-09 22:00:00
    Data columns (total 32 columns):
    T1             2368 non-null float64
    RH_1           2368 non-null float64
    T2             2368 non-null float64
    RH_2           2368 non-null float64
    T3             2368 non-null float64
    RH_3           2368 non-null float64
    T4             2368 non-null float64
    RH_4           2368 non-null float64
    T5             2368 non-null float64
    RH_5           2368 non-null float64
    T6             2368 non-null float64
    RH_6           2368 non-null float64
    T7             2368 non-null float64
    RH_7           2368 non-null float64
    T8             2368 non-null float64
    RH_8           2368 non-null float64
    T9             2368 non-null float64
    RH_9           2368 non-null float64
    T_out          2368 non-null float64
    Press_mm_hg    2368 non-null float64
    RH_out         2368 non-null float64
    Windspeed      2368 non-null float64
    Visibility     2368 non-null float64
    Tdewpoint      2368 non-null float64
    rv1            2368 non-null float64
    rv2            2368 non-null float64
    Hours          2368 non-null float64
    Weekday        2368 non-null float64
    Month          2368 non-null float64
    Day            2368 non-null float64
    Minutes        2368 non-null float64
    Years          2368 non-null float64
    dtypes: float64(32)
    memory usage: 610.5 KB
    


```python
print(np.shape(UCI_target_energy_train))
```

    (2368,)
    


```python
print(UCI_target_energy_train.head(10))
```

    date
    2016-05-24 09:00:00     38.333333
    2016-01-21 16:00:00     41.666667
    2016-04-16 20:00:00    101.666667
    2016-04-02 08:00:00     51.666667
    2016-05-10 07:00:00     61.666667
    2016-05-01 19:00:00    118.333333
    2016-05-13 13:00:00    268.333333
    2016-03-02 13:00:00     88.333333
    2016-04-15 04:00:00     55.000000
    2016-03-25 06:00:00     43.333333
    Name: TARGET_energy, dtype: float64
    


```python
print(UCI_predictors_valid.info())

```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 593 entries, 2016-01-27 04:00:00 to 2016-04-28 01:00:00
    Data columns (total 32 columns):
    T1             593 non-null float64
    RH_1           593 non-null float64
    T2             593 non-null float64
    RH_2           593 non-null float64
    T3             593 non-null float64
    RH_3           593 non-null float64
    T4             593 non-null float64
    RH_4           593 non-null float64
    T5             593 non-null float64
    RH_5           593 non-null float64
    T6             593 non-null float64
    RH_6           593 non-null float64
    T7             593 non-null float64
    RH_7           593 non-null float64
    T8             593 non-null float64
    RH_8           593 non-null float64
    T9             593 non-null float64
    RH_9           593 non-null float64
    T_out          593 non-null float64
    Press_mm_hg    593 non-null float64
    RH_out         593 non-null float64
    Windspeed      593 non-null float64
    Visibility     593 non-null float64
    Tdewpoint      593 non-null float64
    rv1            593 non-null float64
    rv2            593 non-null float64
    Hours          593 non-null float64
    Weekday        593 non-null float64
    Month          593 non-null float64
    Day            593 non-null float64
    Minutes        593 non-null float64
    Years          593 non-null float64
    dtypes: float64(32)
    memory usage: 152.9 KB
    None
    


```python
sns.set()
sns.lineplot(UCI_predictors_valid['Hours'],UCI_target_energy_valid,color="red")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1d426b23608>




![png](output_27_1.png)



```python

```


```python
plot_a=sns.lineplot(UCI_predictors_train['Weekday'],UCI_target_energy_train,color="green")
x=np.arange(7)
plot_a.set_xticks(x)
plot_a.set_xticklabels(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],rotation=30)
```




    [Text(0, 0, 'Monday'),
     Text(0, 0, 'Tuesday'),
     Text(0, 0, 'Wednesday'),
     Text(0, 0, 'Thursday'),
     Text(0, 0, 'Friday'),
     Text(0, 0, 'Saturday'),
     Text(0, 0, 'Sunday')]




![png](output_29_1.png)



```python
plot_a=sns.lineplot(UCI_predictors_valid['Weekday'],UCI_target_energy_valid,color="green")
x=np.arange(7)
plot_a.set_xticks(x)
plot_a.set_xticklabels(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],rotation=30)
```




    [Text(0, 0, 'Monday'),
     Text(0, 0, 'Tuesday'),
     Text(0, 0, 'Wednesday'),
     Text(0, 0, 'Thursday'),
     Text(0, 0, 'Friday'),
     Text(0, 0, 'Saturday'),
     Text(0, 0, 'Sunday')]




![png](output_30_1.png)





```python
# Standardization of the values 
mx=MinMaxScaler()
X_train_transformed=mx.fit_transform(UCI_predictors_train)
X_valid_transformed=mx.fit_transform(UCI_predictors_valid)


```


```python
X_test_transformed=mx.fit_transform(UCI_predictors_test)
```


```python
#applying random forest for feature importance demonstration
selector=feature_selection.SelectKBest(feature_selection.f_regression,k=20)
X_new=selector.fit_transform(X_train_transformed,UCI_target_energy_train)
X_test_new=selector.transform(X_valid_transformed)
print(X_new.shape)
skb_mask=selector.get_support()
print(skb_mask)
out_list=[]
skb_features = [] 
for bool,feature in zip(skb_mask, UCI_predictors_train.columns):
 if bool:
  skb_features.append(feature)
print('Optimal number of features :',len(skb_features))
print('Best features :',skb_features)
for col in UCI_predictors_train.columns:
    skb_pvalues=stats.pearsonr(UCI_predictors_train[col],UCI_target_energy_train)
    out_list.append([col,skb_pvalues[0],skb_pvalues[1]])
p_value_df=pd.DataFrame(out_list,columns=["Features","Correlation","P-values"])
print(p_value_df.head())
    
 

```

    (2368, 20)
    [ True  True  True  True  True  True  True  True False False  True  True
     False  True False  True False  True  True  True  True  True False False
     False False  True False  True False  True False]
    Optimal number of features : 20
    Best features : ['T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T6', 'RH_6', 'RH_7', 'RH_8', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Hours', 'Month', 'Minutes']
      Features  Correlation      P-values
    0       T1     0.056703  5.779079e-03
    1     RH_1     0.146270  8.516979e-13
    2       T2     0.145777  1.016989e-12
    3     RH_2    -0.055474  6.930872e-03
    4       T3     0.091313  8.569650e-06
    

    C:\Users\subha\anaconda3\lib\site-packages\sklearn\feature_selection\_univariate_selection.py:299: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    C:\Users\subha\anaconda3\lib\site-packages\scipy\stats\_distn_infrastructure.py:903: RuntimeWarning: invalid value encountered in greater
      return (a < x) & (x < b)
    C:\Users\subha\anaconda3\lib\site-packages\scipy\stats\_distn_infrastructure.py:903: RuntimeWarning: invalid value encountered in less
      return (a < x) & (x < b)
    C:\Users\subha\anaconda3\lib\site-packages\scipy\stats\_distn_infrastructure.py:1912: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= _a)
    C:\Users\subha\anaconda3\lib\site-packages\scipy\stats\stats.py:3508: PearsonRConstantInputWarning: An input array is constant; the correlation coefficent is not defined.
      warnings.warn(PearsonRConstantInputWarning())
    


```python
X_test_transformed=selector.transform(X_test_transformed)
from sklearn.ensemble import GradientBoostingRegressor
parameter_for_gradient_boost={'n_estimators':[100,200,250,300],'max_depth':[3,4,5],'learning_rate':[0.2,0.3,0.5],'max_features':['auto','sqrt','log2'],'criterion':['mse','friedman']}
gradient_regressor=GradientBoostingRegressor()
gradient_regressor_model=GridSearchCV(gradient_regressor,parameter_for_gradient_boost,cv=3,scoring='r2')
gradient_regressor_model.fit(X_new,UCI_target_energy_train)
result=gradient_regressor_model.predict(X_test_new)

print("The best parameter for the model is")
print(gradient_regressor_model.best_params_)
print("The best score for model is ")
print(gradient_regressor_model.best_score_)


```

    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    

    The best parameter for the model is
    {'criterion': 'mse', 'learning_rate': 0.2, 'max_depth': 4, 'max_features': 'sqrt', 'n_estimators': 300}
    The best score for model is 
    0.37920161232769606
    


```python
from sklearn.tree import DecisionTreeRegressor
parameter_for_gradient_boost={'max_depth':[3,4,5,7],'max_features':['auto','sqrt','log2'],'criterion':['mse','friedman']}
gradient_regressor=DecisionTreeRegressor()
print(gradient_regressor.get_params().keys())
gradient_regressor_model=GridSearchCV(gradient_regressor,parameter_for_gradient_boost,cv=3,scoring='r2')
gradient_regressor_model.fit(X_new,UCI_target_energy_train)
result=gradient_regressor_model.predict(X_test_transformed)
print('Mean test score: {}'.format(gradient_regressor.cv_results_['mean_test_score']))
print('Mean train score: {}'.format(gradient_regressor.cv_results_['mean_train_score']))

print("The best parameter for the model is")
print(gradient_regressor_model.best_params_)
print("The best score for model is ")
print(gradient_regressor_model.best_score_)


```

    dict_keys(['ccp_alpha', 'criterion', 'max_depth', 'max_features', 'max_leaf_nodes', 'min_impurity_decrease', 'min_impurity_split', 'min_samples_leaf', 'min_samples_split', 'min_weight_fraction_leaf', 'presort', 'random_state', 'splitter'])
    

    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    C:\Users\subha\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:536: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    KeyError: 'friedman'
    
      FitFailedWarning)
    


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-44-25787b242226> in <module>
          6 gradient_regressor_model.fit(X_new,UCI_target_energy_train)
          7 result=gradient_regressor_model.predict(X_test_transformed)
    ----> 8 print('Mean test score: {}'.format(gradient_regressor.cv_results_['mean_test_score']))
          9 print('Mean train score: {}'.format(gradient_regressor.cv_results_['mean_train_score']))
         10 
    

    AttributeError: 'DecisionTreeRegressor' object has no attribute 'cv_results_'



```python

```


```python

```


```python

```
