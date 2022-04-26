import glob
import pandas as pd


    
#Comprobar cuentos y que datos hay en cada columna
def dato_colum(df,colum):
    print(colum)
    print(df[colum].value_counts().head(50))


#Limpiar los datos
    #Poner todos los datos de un mismo formato
def definir_formato_edad(df):
    df = df.replace({"2:00 AM": 2,"3:00 AM": 3,"4:00 AM": 4,"5:00 AM": 5,"6:00 AM": 6,"7:00 AM": 7,"8:00 AM": 8})
    
    return df['EDAD'].replace(to_replace='[^0-9,]', regex=True, value='')
    

#Convertimos a numericos los datos necesarios
def convertir_num(df,col):
    #Los datos que hemos recogidos tienen una ',' como indicador de decimal y necesitamos un '.' para convertirlos a float
    df[col].replace(',','.', regex=True, inplace=True)

#Convertimos los datos en el tipo que necesitamos
    df[col]=pd.to_numeric(df[col])

#Colocamos los datos que estan intercambiados en Edad y Kg
def intercambiar_datos(df):
    convertir_num(df,'EDAD')
    convertir_num(df,'POSICION')
    convertir_num(df,'KG')

    #Algunos datos de EDAD y KG estan intercambiados por lo que vamos a colocarlos adecuadamente
    #Guardamos los datos que no estan bien en aux
    auxEdad = df.loc[df['EDAD']>=20]
    auxEdad = auxEdad['EDAD']

    auxKG = df.loc[df['KG']<=20]
    auxKG = auxKG['KG']

    #Introducimos los datos en las columnas correspondientes
    df.loc[df.EDAD >= 20, ['EDAD']] = auxKG
    df.loc[df.KG <= 20, ['KG']] = auxEdad


def definir_formato_hipodromo(df):
    #Vamos a poner el hipodromo en mayusculas para que no haya problema entra mayusculas y minusculas
    df['HIPODROMO'] = df['HIPODROMO'].str.upper()
    df['HIPODROMO'] = df['HIPODROMO'].replace(regex={'.*ZAR.*': 'HIPODROMO LA ZARZUELA', '.*SEBAS.*': 'HIPODROMO DE SAN SEBASTIAN', '.*ANDAL.*': 'GRAN HIPODROMO DE ANDALUCIA JAVIER PIÑAR HAFNER', '.*COSTA.*': 'HIPODROMO COSTA DEL SOL', '.*PINED.*': 'HIPODROMO DE PINEDA',})
