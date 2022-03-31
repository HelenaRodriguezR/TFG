from limpiar_datos import *


if __name__ == "__main__":

    print ('\n[0]##############################')
    #Importar todas las carreras y crear el dataframe
    df = crear_df()
    
    print ('\n[1]##############################')
    #Ver el tipo de datos de cada columna
    print (df.dtypes)

        
    #Comprobar los datos que tiene cada columna
    print ('\n[2]##############################')
    data_index = df.keys().to_list()
    for elem in data_index:
        dato_colum(df,elem)
    #Eliminar todos los caballos que han sido eliminados antes de comenzar las carrera
    df = df.loc[df['POSICION']!=99.0]

    #Sobreescribir valores con formato/datos correctos
    print ('\n[3]##############################')
    print ('\nModificando datos erroneos...')
    df['EDAD']=definir_formato_edad(df)
    intercambiar_datos(df)
    definir_formato_hipodromo(df)    

    #Volver a mostrar todas las columnas para comprobar que todos los datos esten correctos
    print ('\n[4]##############################')
    print (df.dtypes)
    for elem in data_index:
        dato_colum(df,elem)


    #Guardar los datos en csv
    print ('\n[5]##############################')
    data_index = df.keys().to_list()
    df.to_csv('datos.csv', header=0, index=False)
    print ('\nSe han guardado todos los datos correcatmente')


