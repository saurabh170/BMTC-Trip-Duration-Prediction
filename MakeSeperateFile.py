import csv
import pickle

class MakeSeperateFile:

    def __init__(self,File):
        self.File = File

    def makeFile(self):
        busId_list = []
        with open(self.File) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                with open('./data/'+str(row[0])+'.csv','a') as fd:
                    wr = csv.writer(fd, quoting=csv.QUOTE_ALL)
                    wr.writerow(row)
                if row[0] not in busId_list:
                    busId_list.append(row[0])
        busId_list.sort()
        pickle.dump(busId_list,open('BusIdList.pickle','wb'))
