import pandas as pd
import numpy as np
from pylab import * 
import matplotlib.pyplot as plt

if __name__ == "__main__":

    #Importar los datos de la predccion y los resultados reales
    
    predictions_df = pd.read_csv('prediccion.csv', delimiter=',',encoding="utf-8", header=None, names =['ID',0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
    
    data_index = predictions_df['ID'].unique() #Guardar en una variable los ID
    
    predictions_df.set_index('ID',inplace = True)
    predictions_df.index.name = None
    #print (predictions_df)
    
    y_test = pd.read_csv('resultados.csv', delimiter=',',encoding="utf-8", header=None, names =['ID','POSICION'])
    

	
    #Convertir los datos en la posicion de la prediccion
    data=[]

    for elem in data_index:
    	data_poss = predictions_df.sort_values(by=elem, axis=1, ascending = False).loc[elem].dropna()
    	data_poss_pred = data_poss.keys().to_list()
    	data.append(data_poss_pred)
    
    predictions_df = pd.DataFrame(data, index = data_index)
    #print (predictions_df)
    
    
    #Aciertos en la primera posicion 
    lista=[]

    for elem in predictions_df.index:

        p = predictions_df.columns[predictions_df.loc[elem] == 0].tolist()
        #print (p)
        poss_pred = predictions_df.loc[elem,p].iloc[0]
        #print (poss_pred)
        poss_real = y_test.loc[y_test['ID']==elem]
        cur_results = y_test.loc[y_test['ID']==elem]
        poss_real = cur_results['POSICION'].iloc[p].tolist()
        #poss_real = poss_real.tolist()
        
        #print (poss_real)
        resul = int(poss_real[0]-1) - poss_pred
        #print(resul)
        lista.append(resul)
 
    #print(lista)
    """lista=[]
    for elem in predictions_df.index:
    	caballo_poss = predictions_df.loc[elem,[0]].iloc[0]
    	poss_real = y_test.iloc[[caballo_poss],[1]].iloc[0]-1
    	resul = poss_real - 0
    	lista.append(resul)"""
    #lista = dict(zip(lista,map(lambda x: lista.count(x),lista)))
    print(lista)

    #Visualizamos los resultados en una grafica
    #plot_resul = pd.DataFrame(lista, index = data_index)
    #plot_resul = plot_resul['POSICION'].value_counts(sort=False)
    
    plot_resul = pd.DataFrame(lista, index = data_index)
    print(plot_resul)
    plot_resul = plot_resul[0].value_counts(normalize = True, sort=False) * 100
    plot_resul = plot_resul.sort_index(axis=0)
    
    print(plot_resul)
    #print(predictions_df.columns)
    plot_resul.plot.bar(plot_resul)
    #lista.bar(lista)
 
 
    draw()
    savefig("prueba", dpi=300)
