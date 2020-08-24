import pandas as pd

class DataPack:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def PrintToScreen(self):
        print(pd.concat([self.features, self.labels], axis = 1))