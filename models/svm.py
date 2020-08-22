from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import pickle                   ## serialize model

class SVMClassifier:
    def __init__(self):
        self.data = None
        self.weights = None
        self.score = None
        self.model = svm.SVC()

    def train(self, X_train, y_train, save_url = None):
        print("Training...")
        self.model.fit(X_train, y_train)
        if (save_url != None):
            with open(save_url, 'wb') as f:
                pickle.dump(self.model, f)
                print("Model saved at ", save_url)
        print("[!] Train completed.")


    def predict(self, X_test, Y_test):
        print("[!] Testing...")
        Y_res = self.model.predict(X_test)
        print("[!] Test completed.")
        print("[i] SVM Classifier accuracy: ", accuracy_score(Y_res, Y_test))

    