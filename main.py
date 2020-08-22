## B1: Phân tích, mô tả dữ liệu (cho báo cáo)
from helper import readData
from models import svm, logreg
from sklearn.preprocessing import OneHotEncoder

# trainDR = readData.DataReader()
# trainDR.readFile('data/adult.train.csv')
# testDR = readData.DataReader()
# testDR.readFile('data/adult.test.csv')

dataReader = readData.DataReader()
dataReader.ReadDataDescription('data/adult.names.txt')
dataReader.readTrainData('data/adult.train.csv')
dataReader.readTestData('data/adult.train.csv')

# ## B2: Làm SVM (do được phép xài thư viện)
SVM_model = svm.SVMClassifier()
SVM_model.train(dataReader.train.features, dataReader.train.labels, save_url='models/trained/svm.pkl')
SVM_model.predict(dataReader.test.features, dataReader.test.labels)

## B3: Làm logistic reg (không được xài thư viện, tự code)

## B4: So sánh

## B5: Xuất ra file kết quả (.csv)