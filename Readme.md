This folder contains following files/folder:

1. data : Empty directory. At the time of run files seperated with busID will get stored in the folder.

2. final_data : After completing preprocessing final data will get stored in this directory.

3. model : https://drive.google.com/open?id=1BdIP70J-Me8mjyjyD-XtkNXTdmKIsys2 directory contains all models

4. pickle : contains pickle files.

5. MakeSeperateFile.py : code to seperate full bmtc data of 14GB into different files for different busID

6. PrepareData.py : Code for data preprocessing.

7. bmtc.py : contain code to train data after preprocessing.

8. Test.py : will make test.csv into appropriate format to run and will make submission file.

Put w1.csv (bmtc data) into the folder and run the program as: python bmtc.py w1.csv test.csv
Note : Since w1.csv is very large so we are unable to upload it on google drive
