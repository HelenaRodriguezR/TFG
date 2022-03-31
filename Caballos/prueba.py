import pandas as pd
import numpy as np
from pylab import * 
import matplotlib.pyplot as plt
 
if __name__ == "__main__":
 
    y_test = pd.read_csv('resultados.csv', delimiter=',',encoding="utf-8", header=None, names =['ID','POSICION'])
 
 
 
    y_test = y_test.groupby(['ID'])['POSICION'].apply(lambda x: ','.join(str(e) for e in x))
 
    y_test.to_csv('resultados.csv', header=0, index=False)
 
 
    y_test = pd.read_csv('resultados.csv', delimiter=',',encoding="utf-8", header=None, names =['ID',0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
    
    print(y_test)
