import pandas as pd
from sklearn import model_selection

class DataSet:
    def __init__(self):
        self.df = pd.DataFrame()

        self.X = pd.DataFrame()
        self.y = pd.DataFrame()

        self.X_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.y_test = pd.DataFrame()
        
    def fetch_data(self):
        self.df = pd.read_csv("auto-mpg.csv")

    def split_X_y(self):
        self.X = self.df.drop(["mpg"], axis=1)
        self.y = self.df[["mpg"]]

    def split_X_y_train_test(self):
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(self.X,self.y, test_size = 0.2, random_state = 42)
