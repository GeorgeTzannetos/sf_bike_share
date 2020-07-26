# The source code for developing and training a model for the SF bike share dataset to predict net change of bikes every hour in each station

import pandas as pd 
import numpy as np
import xgboost as xgb 
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from matplotlib import pyplot as plt


net_table = pd.read_csv("net_change_weather_avail.csv",parse_dates=[1],dayfirst=True,dtype={'Weekday':int,'Business_day':int,'Holiday':int,
'Station_ID':int, 'Dock_count':int,'Zip':int,'Month':int, 'Hour':int,'Max TemperatureF':float,'Mean TemperatureF':float, 'Min TemperatureF':float,
'Max Dew PointF':float, 'MeanDew PointF':float, 'Min DewpointF':float, 'Max Humidity':float,'Mean Humidity':float, 'Min Humidity':float, 'Max Sea Level PressureIn':float,
'Mean Sea Level PressureIn':float, 'Min Sea Level PressureIn':float,'Max Wind SpeedMPH':float, 'Mean Wind SpeedMPH':float, 'Max Gust SpeedMPH':float,
'PrecipitationIn':float, 'CloudCover':int, 'Events':str, 'WindDirDegrees':float,'bikes_available':int,'docks_available':int})

#Keep only month from date
net_table["Year"] = pd.to_datetime(net_table["Time"]).dt.strftime('%Y')
net_table["Month"] = pd.to_datetime(net_table["Time"]).dt.strftime('%m')#strftime('%d/%m/%Y')
net_table["Hour"] = pd.to_datetime(net_table["Time"]).dt.strftime('%H')

# The specific day is not relevant since it happens only once
net_table.drop(columns=['Time'],inplace=True)
##Remove the start stop count 
net_table.drop(columns=['Start_CountTrips','Stop_CountTrips'],inplace=True)


# net_table = net_table[['Station_ID', 'Dock_count', 'Start_CountTrips',
#        'Stop_CountTrips', 'Date', 'Hour', 'Net_Rate']]


net_table = net_table[['Weekday','Business_day','Holiday','Station_ID', 'Dock_count','bikes_available','Zip','Month', 'Hour','Max TemperatureF', 
       'Mean TemperatureF', 'Min TemperatureF',
       'Max Dew PointF', 'MeanDew PointF', 'Min DewpointF', 'Max Humidity',
       'Mean Humidity', 'Min Humidity', 'Max Sea Level PressureIn',
       'Mean Sea Level PressureIn', 'Min Sea Level PressureIn',
       'Max Wind SpeedMPH', 'Mean Wind SpeedMPH', 'Max Gust SpeedMPH',
       'PrecipitationIn', 'CloudCover', 'Events', 'WindDirDegrees', 'Net_Rate']]

print(net_table.keys())

lbl = preprocessing.LabelEncoder()
#net_table['Year'] = lbl.fit_transform(net_table['Year'])
net_table['Month'] = lbl.fit_transform(net_table['Month'])
net_table['Hour'] = lbl.fit_transform(net_table['Hour'])
#net_table['Zip'] = lbl.fit_transform(net_table['Zip'])

#Change type of values depending on their nature 
net_table['Station_ID']=net_table['Station_ID'].apply(int)
net_table['Dock_count']=net_table['Dock_count'].apply(int)
net_table['Net_Rate']=net_table['Net_Rate'].apply(int)

net_table['Events']=net_table['Events'].apply(str)
net_table['Events'] = lbl.fit_transform(net_table['Events'])

net_table.to_csv("train_data.csv")
#net_table = net_table.fillna(0)

## Separate target variable with rest of variables

X, y = net_table.iloc[:,:-1], net_table.iloc[:,-1]

data_dmatrix = xgb.DMatrix(data=X, label=y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Instead of random split I seperate train as past data and test as future data 
## This part is done if you want to split the data sequentially

# idx = int(len(X)*0.80)
# X_train = X.iloc[:idx,:]
# X_test = X.iloc[idx:,:]
# y_train = y.iloc[:idx]
# y_test = y.iloc[idx:]


# A function for doing cross validation 
def cv_scores(clf):
    scores = cross_val_score(clf, X_train, y_train, cv=15, n_jobs=1, scoring = 'neg_mean_squared_error')
    print(scores)


eval_set = [(X_train, y_train), (X_test, y_test)]

xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.9, learning_rate = 0.6, subsample=0.9,
                max_depth = 8, alpha=3.0, n_estimators = 500) #15


# This part is uncommented if we need to run GridSearch
# xg_reg = xgb.XGBRegressor(objective ='reg:squarederror')

# parameters= {
#        "colsample_bytree": [0.8,0.3],
#        "learning_rate": [0.2],
#        "max_depth" :[10],
#        "alpha" :[0.5],
#        "n_estimators" :[100]
# }

# xg_reg = GridSearchCV(xg_reg, parameters, n_jobs=1, cv=2, refit=True)

xg_reg.fit(X_train,y_train, eval_metric='rmse',eval_set=eval_set,verbose=True)

scoring(xg_reg)

# Results of the GridSearch

# results = xg_reg.cv_results_
# score = xg_reg.best_score_
# print('score:', score)
# print(xg_reg.best_params_)

preds = xg_reg.predict(X_test)
train_preds = xg_reg.predict(X_train)

preds_df = pd.DataFrame(preds, y_test)
preds_df.to_csv("predictionsCorrect.csv")

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE_test: %f" % (rmse))

rmse_train = np.sqrt(mean_squared_error(y_train, train_preds))
print("RMSE_test: %f" % (rmse_train))

plot_importance(xg_reg._Booster)

gain = xg_reg._Booster.get_score(importance_type='gain')
print(gain)

# cover = xg_reg._Booster.get_score(importance_type='cover')
# print(cover)

plt.savefig('features.jpg')

####
# k-fold cross validation

params = {"objective":"reg:squarederror",'colsample_bytree': 0.8,'learning_rate': 0.5,
                'max_depth': 5, 'alpha': 0.5}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=500, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)

print(cv_results)#.head()

print((cv_results["test-rmse-mean"]).tail(1))

## Visualize important features

xgb.plot_tree(xg_reg,num_trees=0)
plt.rcParams['figure.figsize'] = [50, 10]
plt.show()

xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()


