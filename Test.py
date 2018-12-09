import sklearn
import csv
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pygame import mixer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from random import randint
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import math
import xgboost as xgb
import random


class Test:

    X=0
    bmtc = 0

    def __init__(self,testFile):
        self.testFile = testFile
    def seperateDateTime(self,y):
#     dic = {0:31,}
        date_object = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in Test.bmtc['Date and Time']]
        date_object = np.array(date_object)
        date_object.reshape(-1,1)
        Year = []
        Month = []
        Day = []
        Hour = []
        Minute = []
        Second = []
        hr = int(y/3600)
        y = y%3600
        mi = int(y/60)
        y = y%60
        sec = y
        for i in date_object:
            Year.append(i.year)
            Month.append(i.month)
            Day.append(i.day)
            if i.second +sec >59:
                Second.append(i.second+sec-60)
                mi+=1
            else:
                Second.append(i.second+sec)
            if i.minute + mi>59:
                Minute.append(i.minute+mi-60)
                hr+=1
            else:
                Minute.append(i.minute+mi)
            if i.hour+hr>23:
                Hour.append(i.hour+hr-24)
                Day[0]+=1
            else:
                Hour.append(i.hour+hr)
        if Day[0]>31:
            Day[0]=1
            Month[0]+=1
        Test.bmtc['year']=Year
        Test.bmtc['month']=Month
        Test.bmtc['day']=Day
        Test.bmtc['hour']=Hour
        Test.bmtc['minute']=Minute
        Test.bmtc['second']=Second
        Test.bmtc.drop('Date and Time',inplace=True,axis=1)

        Test.bmtc['time_in_minute']=Test.bmtc['hour']*60 + Test.bmtc['minute'] + Test.bmtc['second']/60# Convert starting time to minute


    # convert Date into week
    def DateToWeek(self):
        week = []
        for i in range(len(Test.bmtc['year'])):
            date_object = datetime(Test.bmtc['year'][i],Test.bmtc['month'][i],Test.bmtc['day'][i])
            date_object=date_object.strftime('%A')
            week.append(date_object)

        # out Partial data (train) dosen't contain Thrisday so I converted week of thrusday as wednesday
        if week[0]=='Thursday':
            week[0]='Wednesday'
        Test.bmtc['week'] = week

    # make sin(hour) and cos(hour) columns
    def sincosConversion(self,k=15):
        for i in range(1,k):
            Test.bmtc['hoursin'+str(i)]=np.sin(np.array(2*math.pi*i*Test.bmtc['hour']/14))
            Test.bmtc['hourcos'+str(i)]=np.cos(np.array(2*math.pi*i*Test.bmtc['hour']/14))


    def Encoding(self):
        from sklearn.preprocessing import LabelEncoder,OneHotEncoder

        obj_df = Test.bmtc.select_dtypes(include=['object']).copy()

        lb = pickle.load(open('./pickle/LabelEncoder.pickle','rb'))

        for i in obj_df.columns:
            obj_df[i] = lb.transform(obj_df[i])
        flt_df = Test.bmtc.select_dtypes(include=['int','float']).copy()

        on = pickle.load(open('./pickle/OneHotEncoder.pickle','rb'))
        obj_df = on.transform(obj_df).toarray()
        flt_arr = np.array(flt_df)
        Test.X = np.concatenate((flt_arr,obj_df),axis=1)


    def prepareData(self,y):
        self.seperateDateTime(y)
        self.DateToWeek()
        self.sincosConversion()


    def make_submission_file(self,model):

        xlf = model
        Test.bmtc = self.testFile

        n=0
        tr=pd.DataFrame()
        for i in range(len(Test.bmtc)):
            for j in range(len(Test.bmtc.iloc[i])):
                if(n==0):
        #             print(Test.bmtc.iloc[i][j])
                    bus_id = Test.bmtc.iloc[i][j]
                    n=n+1
                elif(n==1):
        #             print(Test.bmtc.iloc[i][j])
                    n+=1
                elif(n<101):
                    zeroth = float(Test.bmtc.iloc[i][j].split(":",1)[0])
                    first = float(Test.bmtc.iloc[i][j].split(":",1)[1])
                    second = float(Test.bmtc.iloc[i][j+1].split(":",1)[0])
                    third = float(Test.bmtc.iloc[i][j+1].split(":",1)[1])
                    tr = tr.append([[bus_id,zeroth,first,second,third]])
                    n=n+1
                if(n==101):
                    n+=1
                elif(n==102):
                    n=0
        time_list=list(Test.bmtc['TimeStamp'])

        ans_list = []
        ans=0
        y=0
        ll =  np.array([time_list[0]],dtype=object).reshape(-1,1)
        for i in range(len(tr)+1):

            if i%99==0 and i!=0:
                ans_list.append(ans)
                ans=0
                if i==len(tr):
                    break
                ll =  np.array([time_list[int(i/99)]],dtype=object).reshape(-1,1)
                print(i/99," ",time_list[int(i/99)])

            l =[]
            l.append(tr.iloc[i][1])
            l.append(tr.iloc[i][3])
            l.append(tr.iloc[i][2])
            l.append(tr.iloc[i][4])
            l = np.array([l])
            df1 = pd.DataFrame(l,columns=['Lat1','Lat2','Long1','Long2'])
            df2 = pd.DataFrame(ll,columns=['Date and Time'])
        #     print(df2.head())
            Test.bmtc = pd.concat([df1,df2],axis=1)

            self.prepareData(int(y))

            Test.bmtc.drop(['year','month','day','time_in_minute'],axis=1,inplace=True)

            self.Encoding()
            X = Test.X
            y = xlf.predict(X)
            ans+=y

        # make_submission_file
        submission = pd.read_csv('sample_submission.csv')
        ans_list2 = np.array(ans_list)
        ans_list2=ans_list2.reshape(-1,1)
        submission['Duration']=(ans_list2)
        submission.to_csv('submission1.csv',index=False)
