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

class PrepareData:
    bmtc = 0
    def __init__(self,BusIdList,number_of_busId=10):
        # number_of_busId = number of busID's you want to take to prepapre data from BusIdList
        # we can't take all busId's due to hardware limitations
        self.BusIdList = BusIdList
        self.number_of_busId = number_of_busId

    # Sperate date and time
    def seperateDateTime(self):
        date_object = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in PrepareData.bmtc['Date and Time']]
        date_object = np.array(date_object)
        date_object.reshape(-1,1)
        Year = []
        Month = []
        Day = []
        Hour = []
        Minute = []
        Second = []
        for i in date_object:
            Year.append(i.year)
            Month.append(i.month)
            Day.append(i.day)
            Hour.append(i.hour)
            Minute.append(i.minute)
            Second.append(i.second)
        PrepareData.bmtc['year']=Year
        PrepareData.bmtc['month']=Month
        PrepareData.bmtc['day']=Day
        PrepareData.bmtc['hour']=Hour
        PrepareData.bmtc['minute']=Minute
        PrepareData.bmtc['second']=Second
        PrepareData.bmtc.drop('Date and Time',inplace=True,axis=1)

        PrepareData.bmtc['time_in_minute']=PrepareData.bmtc['hour']*60 + PrepareData.bmtc['minute'] + PrepareData.bmtc['second']/60# Convert starting time to minute


    # Week column
    def DateToWeek(self):
        week = []
        for i in range(len(PrepareData.bmtc['year'])):
            date_object = datetime(PrepareData.bmtc['year'][i],PrepareData.bmtc['month'][i],PrepareData.bmtc['day'][i])
            date_object=date_object.strftime('%A')
            week.append(date_object)
        PrepareData.bmtc['week'] = week


    #Removing Errors in Latitude and Longitude
    def RemoveError(self):
        PrepareData.bmtc.drop(PrepareData.bmtc[PrepareData.bmtc['Longitude']>=78].index,inplace=True)
        PrepareData.bmtc.drop(PrepareData.bmtc[PrepareData.bmtc['Longitude']<=77].index,inplace=True)
        PrepareData.bmtc.drop(PrepareData.bmtc[PrepareData.bmtc['Latitude']<=12.5].index,inplace=True)
        PrepareData.bmtc.drop(PrepareData.bmtc[PrepareData.bmtc['Latitude']>=14].index,inplace=True)
        PrepareData.bmtc.reset_index(inplace=True,drop=True) #Reset index


    #Sort vlaues by BusId and then time
    def sortData(self):
        PrepareData.bmtc.sort_values(by=['BusId','year','month','day','hour','minute','second'],inplace=True)
        PrepareData.bmtc.reset_index(inplace=True,drop=True) #Reset Index


    #Drop consecutive rows with speed as 0

    def drop_rows_with_speed_zero(self):
        i=0
        list_index = []
        while i < (len(PrepareData.bmtc['BusId'])-1):
            if PrepareData.bmtc['Speed'][i]==PrepareData.bmtc['Speed'][i+1] and  PrepareData.bmtc['BusId'][i]==PrepareData.bmtc['BusId'][i+1]:
                list_index.append(i+1)
            i+=1
        PrepareData.bmtc.drop(list_index,inplace=True)
        PrepareData.bmtc.reset_index(inplace=True,drop=True)

    # preparing data and making initial and final co-ordiante
    def convert(self):
        Lat1 = []
        Lat2 = []
        Long1 = []
        Long2 = []
        timeTaken = []
        startHour = []
        startMinute = []
        startSecond = []
        week=[]
        hourCos=[]
        hourSin=[]

        for i in range(1,len(PrepareData.bmtc)-11):
            Lat1.append(PrepareData.bmtc['Latitude'][i])
            Long1.append(PrepareData.bmtc['Longitude'][i])
            startHour.append(PrepareData.bmtc['hour'][i])
            startMinute.append(PrepareData.bmtc['minute'][i])
            startSecond.append(PrepareData.bmtc['second'][i])
            week.append(PrepareData.bmtc['week'][i])
    #         hourCos.append(math.cos(PrepareData.bmtc['hour'][i]*math.pi/180))
    #         hourSin.append(math.sin(PrepareData.bmtc['hour'][i]*math.pi/180))
            if PrepareData.bmtc['week'][i]!=PrepareData.bmtc['week'][i+1]:
                Lat2.append(PrepareData.bmtc['Latitude'][i])
                Long2.append(PrepareData.bmtc['Longitude'][i])
                timeTaken.append(0)
                i+=2
            else:
                a = randint(1, 9)
                while PrepareData.bmtc['week'][i]!=PrepareData.bmtc['week'][i+a]:
                    a = randint(1, a)
                Lat2.append(PrepareData.bmtc['Latitude'][i+a])
                Long2.append(PrepareData.bmtc['Longitude'][i+a])
                timeTaken.append((PrepareData.bmtc['time_in_minute'][i+a]-PrepareData.bmtc['time_in_minute'][i])*60)

        col = {"Lat1":Lat1,"Long1":Long1,"Lat2":Lat2,"Long2":Long2,"hour":startHour,
               "minute":startMinute,"second":startSecond,"week":week,
               "timeTaken":timeTaken}
        PrepareData.bmtc = pd.DataFrame(col)

    # making sin(hour) and cos(hour) rows
    def sincosConversion(bmtc,k=15):
        for i in range(1,k):
            PrepareData.bmtc['hoursin'+str(i)]=np.sin(np.array(2*math.pi*i*PrepareData.bmtc['hour']/14))
            PrepareData.bmtc['hourcos'+str(i)]=np.cos(np.array(2*math.pi*i*PrepareData.bmtc['hour']/14))

    # prepare Data
    def prepareData(self):
        new_list = random.sample(self.BusIdList,self.number_of_busId)
        columns = ['BusId','Latitude','Longitude','Angle','Speed','Date and Time']
        for i in range(len(new_list)):
            PrepareData.bmtc = pd.read_csv('./data/'+new_list[i]+'.csv',names=columns)
            #print('./data/'+i+'.csv')
            self.seperateDateTime()
            self.DateToWeek()
            self.RemoveError()
            self.sortData()
            self.drop_rows_with_speed_zero()
            self.convert()
            self.sincosConversion()

            PrepareData.bmtc.to_csv('./final_data/finalData.csv',index=False,mode='a',header=False)
