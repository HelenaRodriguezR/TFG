import pandas as pd
import numpy as np
from pylab import * 
import matplotlib.pyplot as plt

def aciertos_poss(predictions_df, y_test, i):
 
    lista=[]

    for elem in predictions_df.index:
    
        caballo = predictions_df.columns[predictions_df.loc[elem] == i].tolist() #Que caballo quedó en la posicion i (0,1,2)
        #print (caballo)
          
        poss_pred = predictions_df.loc[elem,caballo].iloc[0] #Posicion en la que quedo el caballo (0,1,2)
        #print (poss_pred)
        
        #poss_real = y_test.loc[y_test['ID']==elem] #Mostrar la clasificacion real de la carrera
        #print(poss_real)
        
        cur_results = y_test.loc[y_test['ID']==elem]
        poss_real = cur_results['POSICION'].iloc[caballo].tolist() #Posicion real en la que quedo ese caballo
        #print (poss_real)
          
        resul = int(poss_real[0]-1) - poss_pred #Diferencia entre el puesto real y la posicion predicha
        #print(resul)
        
        lista.append(resul) #Añadir el resultado a una lista para luego mostrarla
        
    #print(lista)    
    return lista
    
  

def visualizar_plot(lista, i):
    plot_resul = pd.DataFrame(lista, index = data_index) #Se crea un dataframe con los resultados
    #print(plot_resul)
    plot_resul = plot_resul[0].value_counts(normalize = True, sort=False) * 100 #% de posiciones acertadas
    #plot_resul = plot_resul[0].value_counts(sort=False) #Nº de posiciones acertadas
    plot_resul = plot_resul.sort_index(axis=0)
    
    print ('Posicion ' + str(i))
    
    print(plot_resul)
    plot_resul.plot.bar(plot_resul)
    
    draw()
    savefig("012"[i], dpi=300) #Guardar las figuras
    
    
if __name__ == "__main__":

    ###Importar los datos de la predccion y los resultados reales###
    
    print ('\n[1]##############################')
    
    predictions_df = pd.read_csv('prediccion.csv', delimiter=',',encoding="utf-8", header=None, names =['ID',0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
    
    data_index = predictions_df['ID'].unique() #Guardar en una variable los ID
    
    predictions_df.set_index('ID',inplace = True)
    predictions_df.index.name = None
    #print (predictions_df)
    
    y_test = pd.read_csv('resultados.csv', delimiter=',',encoding="utf-8", header=None, names =['ID','POSICION'])
    
    print ('\nLos datos de la prediccion y los reales han sido importados')

#########################################################################################################################
    
    print ('\n[2]##############################')
    
    ###Convertir los datos de la prediccion en datos claros###
    
    data=[]

    for elem in data_index:
    	data_poss = predictions_df.sort_values(by=elem, axis=1, ascending = False).loc[elem].dropna()
    	data_poss_pred = data_poss.keys().to_list()
    	data.append(data_poss_pred)
    
    predictions_df = pd.DataFrame(data, index = data_index)
    predictions_df.to_csv('prediccion_mod.csv', header=0)
    #print (predictions_df)
    
    print ('\nSe han tratado los resultados para que estos sean claros')
    
#########################################################################################################################

    print ('\n[3]##############################')
    
    ###Comparar la prediccion con la posicion real y visualizar###
    
    for i in [0,1,2]:
        ###Aciertos en las tres primeras posiciones###
        lista = aciertos_poss(predictions_df, y_test, i)
        
        ###Visualizar el resultado###
        visualizar_plot(lista, i)
     
    

    

    
    
    

