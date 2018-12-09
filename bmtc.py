import MakeSeperateFile
import PrepareData
import Test

import sklearn
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle
from pygame import mixer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from random import randint
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import math
import xgboost as xgb

class Train:

    X = 0
    y = 0
    def __init__(self,bmtc):
        self.bmtc = bmtc

    def Encoding(self):
        obj_df = self.bmtc.select_dtypes(include=['object']).copy()
        lb = LabelEncoder()

        for i in obj_df.columns:
            obj_df[i] = lb.fit_transform(obj_df[i])
            pickle.dump(lb,open('./pickle/LabelEncoder.pickle','wb'))
        flt_df = self.bmtc.select_dtypes(include=['int','float']).copy()
        flt_df.drop(['timeTaken'],axis=1,inplace=True)

        on = OneHotEncoder()
        obj_df = on.fit_transform(obj_df).toarray()
        pickle.dump(on,open('./pickle/OneHotEncoder.pickle','wb'))
        flt_arr = np.array(flt_df)

        Train.X = np.concatenate((flt_arr,obj_df),axis=1)
        Train.y = np.array([self.bmtc['timeTaken']]).T

    #Linear Regression
    def Linear_Regression(self):
        self.Encoding()
        X_train,X_test,y_train,y_test = train_test_split(Train.X,Train.y,test_size=0.25,random_state=4)
        lr = LinearRegression().fit(X_train,y_train)
        print("Traning score: ",lr.score(X_train,y_train))
        print("Test score: ",lr.score(X_test,y_test))
        print("RMSE train: ",np.sqrt(mean_squared_error(y_train,lr.predict(X_train))))
        print("RMSE test: ",np.sqrt(mean_squared_error(y_test,lr.predict(X_test))))
        pickle.dump(lr,open('./model/LinearRegression.pickle','wb'))

    #Random Forest
    def RandomForest(self):
        self.Encoding()
        X_train,X_test,y_train,y_test = train_test_split(Train.X,Train.y,test_size=0.25,random_state=4)
        rm = RandomForestRegressor(n_estimators=10).fit(X_train,y_train)
        print("Traning score: ",rm.score(X_train,y_train))
        print("Test score: ",rm.score(X_test,y_test))
        print("RMSE train: ",np.sqrt(mean_squared_error(y_train,rm.predict(X_train))))
        print("RMSE test: ",np.sqrt(mean_squared_error(y_test,rm.predict(X_test))))
        pickle.dump(lr,open('./model/RandomForest.pickle','wb'))

    def XGBoost(self):
        self.Encoding()
        X_train,X_test,y_train,y_test = train_test_split(Train.X,Train.y,test_size=0.25,random_state=4)
        clf = xgb.XGBModel(max_depth=8,n_estimators=100,objective="reg:linear", random_state=17,n_jobs=-1)
        clf.fit(X_train, y_train, eval_metric='rmse', verbose = True, eval_set = [(X_train,y_train),(X_test, y_test)])
        clf.save_model('./model/XGBoost.model')
        pickle.dump(clf, open("XGBosst.pickle.dat", "wb"))





if __name__ == "__main__":
    aa = MakeSeperateFile.MakeSeperateFile(sys.argv[1])
    aa.makeFile()
    with open('BusIdList.csv','rb') as f:
        list = pickle.load(f)
    data = PrepareData.PrepareData(list,2000)
    data.prepareData()
    col = ["Lat1","Lat2","Long1","Long2","hour", "minute","second",
       "timeTaken","week",'sin1','sin2','sin3','sin4','sin5','sin6',
       'sin7','sin8','sin9','sin10','sin11','sin12','sin13','sin14',
       'cos1','cos2','cos3','cos4','cos5','cos6','cos7','cos8','cos9',
       'cos10','cos11','cos12','cos13','cos14']
    final_data = pd.read_csv('./final_data/finalData.csv',names=col)
    train = Train(final_data)
    train.Linear_Regression()
    train.RandomForest()
    train.XGBoost()


    test_df = pd.read_csv(sys.argv[2])
    model = pickle.load(open('./model/LinearRegression.pickle','rb'))
    test = Test.Test(test_df)
    test.make_submission_file(model)
