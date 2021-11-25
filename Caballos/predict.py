from train import (X_test,clf_xgb)
from librerias import *


def main():
    print(X_test)
    def predict(model, df):
    return model.predict(df.loc[:, ~df.columns.isin(['ID'])])


    predictions = (X_test.groupby('ID').apply(lambda x: predict(clf_xgb, x)))
     
main()
