import glob
import pandas as pd

from limpiar_datos import *


if __name__ == "__main__":

    print ('\n[1]##############################')
    
    ###Importar datos y crear un df con todas las carreras para trabajar con el###
    
    #Importar carreras
    archivos_csv = glob.glob('../Caballos/ID-Carreras/*.csv')
    print("Se han importado todas las carreras")
    
    #Creamos un dataframe y lo eliminamos del conjunto
    filename = archivos_csv[0]
    archivos_csv.remove(filename)
    df = pd.read_csv(filename, delimiter=',',encoding="utf-8",header=0)

    #Escribimos un loop que ira añadiendo cada uno de los nombres del archivo al dataframe final
    for filename in archivos_csv:
        data = pd.read_csv(filename,delimiter=',',encoding="utf-8",header=0)
        df = df.append(data)

    print(df)
    
    print ('Todos las las carreras se encuentran en un Dataframe')

#########################################################################################################################

    print ('\n[2]##############################')
    
    ###Ver el tipo de datos de cada columna##
    print ('\nColumnas:')
    print(df.dtypes)    
    
#########################################################################################################################

    print ('\n[3]##############################')
    
    ###Comprobar los datos que tiene cada columna###
    data_index = df.keys().to_list()
    for elem in data_index:
        dato_colum(df,elem)
        
    ###Eliminar todos los caballos que han sido eliminados antes de comenzar las carrera###
    df = df.loc[df['POSICION']!=99.0]
    
    print ('Se han comprobado los datos que hay en cada columna')
    
#########################################################################################################################

    print ('\n[4]##############################')
    
    ###Sobreescribir valores con formato/datos correctos###
    print ('\nModificando datos erroneos...')
    
    df['EDAD']=definir_formato_edad(df)
    intercambiar_datos(df)
    definir_formato_hipodromo(df)
    
    print ('\nSe han modificado todos los datos') 
    
#########################################################################################################################

    print ('\n[5]##############################')
    
    ###Volver a mostrar todas las columnas para comprobar que todos los datos esten correctos###
    
    print (df.dtypes)
    for elem in data_index:
        dato_colum(df,elem)
    
#########################################################################################################################
    
    print ('\n[6]##############################')
    
    ###Guardar los datos correctos en csv###

    data_index = df.keys().to_list()
    df.to_csv('datos.csv', header=0, index=False)
    
    print ('\nSe han guardado todos los datos correcatmente')

    
    
    
    
    
    
    
    
    
    
    
    
   
