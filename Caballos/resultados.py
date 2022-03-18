import pandas as pd
import numpy as np


if __name__ == "__main__":

    #data_index = predictions.keys().to_list()
    #predictions_df = pd.read_csv('prediccion.csv', delimiter=',',encoding="utf-8", header=None, names =['ID',0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], index_col = 0)
    
    predictions_df = pd.read_csv('prediccion.csv', delimiter=',',encoding="utf-8", header=None, index_col = 0)
    
    print (predictions_df)
	
    #Convertir los datos en la posicion de la prediccion
    data_index = predictions_df.keys().to_list()
    #print(data_index)
    
