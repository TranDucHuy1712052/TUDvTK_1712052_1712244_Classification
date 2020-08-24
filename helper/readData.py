import pandas as pd
import numpy as np
from helper import DataPack
from sklearn.preprocessing import OneHotEncoder

## Điều kiện sử dụng: Cả 2 tập dữ liệu train/test đều phải cùng 1 kiểu (cùng 1 bộ hoặc là cùng cấu trúc, quy tắc...)
class DataReader:

    def __init__(self):
        self.train = None
        self.test = None
        self.encoders = []                      ## các encoder cho tập dữ liệu này
        self.atrNames = []                     ## tên các thuộc tính
        self.atrVals = []                        ## tập giá trị của từng thuộc tính (-1 nếu là biến liên tục)

    ## đọc file ban đầu, nạp vào các biến
    def ReadDataDescription(self, url):
        f = open(url)
        lines = f.readlines()
        names = []
        vals = []
        for line in lines:
            line = line.strip(' .\t\n\r')
            _line = line.split(':')
            _line[0] = _line[0].strip(' \t\n\r')        # xóa hết dấu cách và enter
            names.append(_line[0])
            val = _line[1].split(',')
            for i in range(0, len(val)):
                val[i] = [val[i].strip(' \t\n\r')]
            vals.append(val)
        print("Data attributes found: ", names)
        self.__Initialize(names, vals)

    def __Initialize(self, atrNames, atrVals):
        self.atrNames = atrNames
        self.atrVals = atrVals
        ## self.__EncodeData()

    def __IsContinuous(self, idx):
        return (self.atrVals[idx] == [['continuous']])

    ## trả về dataPack 
    def __readData_helper(self, url):
        df = pd.read_csv(url, header=0)
        print("Data shape = ", df.shape)
        self.__TrimStrings(df)
        self.__FindNull(df)
        print(df.columns)

        encodedColumns = pd.DataFrame()
        ## encode các cột theo dạng category => one hot encoding
        for i in range(0, len(self.atrNames) ):
            #if (self.encoders[i] != None):          # không phải None => có encode rồi
            if not (self.__IsContinuous(i)):
                #enc = self.encoders[i]              # lấy encoder tương ứng với cột này
                col = df.columns[i]
                
                # duyệt từng phần tử trong cột và encode
                # newDF = pd.DataFrame()
                # for entry in df[col]:
                #     tmpDF = pd.DataFrame( enc.transform([ [entry] ]).toarray() )
                #     pd.concat([newDF, tmpDF], axis=0)
                newDF = pd.get_dummies(df[col], prefix=df.columns[i])
                ## print(newDF, '\n\n')

                encodedColumns = pd.concat([encodedColumns, newDF], axis=1)
                print("Column", df.columns[i], "encoded.")
        
        dropIdx = []
        for i in range(0, len(self.atrNames) ):
            if not (self.__IsContinuous(i)):
                col = df.columns[i]
                print("Deleting column : ", col)
                dropIdx.append(col)

        df.drop(columns=dropIdx, inplace=True)
        
        ##print(encodedColumns)
        result = pd.concat( [df, encodedColumns], axis=1 )   

        labels = result["label"]
        result.drop(columns="label", inplace=True)            #gỡ ra khỏi dataframe để khỏi bị trùng lắp
       ## labels_encoded = pd.get_dummies(labels)

        features = result
        dataPack = DataPack.DataPack(features, labels)
        print("Read file succesfully.")
        ##print( pd.concat([result, labels], axis=1) )
        return dataPack

    def __readTrainData(self, url):
        print("[!] Train data reading...\n\n")
        pack = self.__readData_helper(url)
        self.train = pack
        print("[!] Train data read successfully!\n\n")

    def __readTestData(self, url):
        print("[!] Test data reading...\n\n")
        pack = self.__readData_helper(url)
        self.test = pack
        print("[!] Test data read successfully!\n\n")

    def readData(self, train_url, test_url):
        self.__readTrainData(train_url)
        self.__readTestData(test_url)
        ## chuẩn hóa lại để cả 2 tập đều cùng số cột
        col_list = (self.train.features.append([self.test.features])).columns.tolist()
        self.train.features = self.train.features.reindex(columns=col_list, fill_value=0)
        self.test.features = self.test.features.reindex(columns=col_list, fill_value=0)
        
        ## xuất ra ngoài
        self.train.PrintToScreen()
        self.test.PrintToScreen()
        print("[i] Data is ready.")

    ## dữ liệu có dấu '?' tức là bị thiếu/mất
    def __FindNull(self, df):
        rows = df.shape[0]
        cols = df.shape[1]
        print("Finding missing values... ")
        for i in range(0, rows):
            for j in range(0, cols):
                if (df.iat[i,j] == '?'):
                    df.iat[i,j] = np.nan


    def __TrimStrings(self, df: pd.DataFrame()):
        rows = df.shape[0]
        cols = df.shape[1]
        print("Trimming strings...")
        for i in range(0, rows):
            for j in range(0, cols):
                if (type( df.iat[i,j] ) == str):
                    df.iat[i,j] = df.iat[i,j].strip(' \t\n\r.')         # label bộ test có dấu chấm, cẩn thận

    ## encode nó sang dạng one-hot để có thể chạy được trên các mô hình
    def __EncodeData(self):
        print("Encoding categories...")
        for i in range(0, len(self.atrNames) ):
            if (self.atrVals[i] == [['continuous']]):
                self.encoders.append(None)
            else:
                vals = self.atrVals[i]            ## 2D array
                encoder = OneHotEncoder(handle_unknown = 'ignore')
                encoder.fit(vals)
                self.encoders.append(encoder)
                print("Attribute ", self.atrNames[i], " encoded.")

    