#Importar las librerias que vamos a utilizar
import glob
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

def dividir_datos(df)
    
    #Dividir el conjunto de datos en train y test. 
    #Todos los caballos con mismo ID deben de estar en el mismo grupo
    gss = GroupShuffleSplit(train_size=.60, n_splits=1, random_state = 7).split(df, groups=df['ID']) 

    train_inds, test_inds = next(gss)
    
    #Convertir los datos en un df
    train= df.iloc[train_inds]

    X_train = train.loc[:, train.columns!='POSICION']
    y_train = train['POSICION'].copy()

    test= df.iloc[test_inds]

    X_test = test.loc[:, train.columns!='POSICION']
    y_test = test[['ID','POSICION']].copy()

def entrenar(X_train,y_train)
    
    #Eliminar el ID de X_train para agruparlas en el entrenamiento por ID
    X_train = X_train.drop('ID',axis=1).copy()

    #Para utilizar XGBoostRanker, debemos indicar el grupo al que debe clasificar
    groups = train.groupby('ID').size().to_frame('size')['size'].to_numpy()

    #Elegimos los hipermarametros del entrenamiento
    clf_xgb=xgb.XGBRanker( 
        booster='gbtree',
        objective='rank:ndcg',
        random_state=7, 
        learning_rate=0.05, 
        max_depth=6, 
        n_estimators=100, 
        subsample=0.5 
        )

    clf_xgb.fit(X_train, y_train,group =groups)


