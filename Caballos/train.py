import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

if __name__ == "__main__":

    print ('\n[1]##############################')
    
    ###Importar los datos para trabajar con ellos###
    
    #La primera linea importa los datos no revisados manualmente
    #df = pd.read_csv('datos.csv', delimiter=',',encoding="utf-8", header=0, names=['ID','CABALLO','POSICION','EDAD','KG','JINETE','CUADRA','PREPARADOR','CAJÓN','HIPODROMO','DISTANCIA','PISTA'])
    df = pd.read_csv('datos-rev.csv', delimiter=',',encoding="utf-8", header=0, names=['ID','CABALLO','POSICION','EDAD','KG','JINETE','CUADRA','PREPARADOR','CAJÓN','HIPODROMO','DISTANCIA','PISTA'])
    #print (df)
    
    print('Se han importado correctamente los datos')

#########################################################################################################################

    print ('\n[2]##############################')
    
    ###Con XGBoots no podemos tener datos categoricos, por lo que vamos a realizar una codificacion en caliente###
    df = pd.get_dummies(df,columns=['CABALLO','JINETE','CUADRA','PREPARADOR','HIPODROMO','PISTA'])
    
    
    #####MODIFICACION DE PARAMETROS DE ENTRADA#####
    #df = df.drop(['PREPARADOR','HIPODROMO','DISTANCIA','PISTA'],axis=1).copy()
    #print(df)
    #df = pd.get_dummies(df,columns=['CABALLO', 'JINETE', 'CUADRA'])
    
    print('Codificacion en caliente realizada')

#########################################################################################################################

    print ('\n[3]##############################')
    
    ###Se divididen los datos en dos grupos sin que se separen caballos de una carrera###
    gss = GroupShuffleSplit(train_size=.60, n_splits=1, random_state = 7).split(df, groups=df['ID'])

    train_inds, test_inds = next(gss)
    
    ###Se guardan los datos necesarios en las variables de entrenamiento y test###
    train= df.iloc[train_inds]
    X_train = train.loc[:, train.columns!='POSICION']
    y_train = train['POSICION'].copy()

    test= df.iloc[test_inds]

    X_test = test.loc[:, train.columns!='POSICION']
    y_test = test[['ID','POSICION']].copy()
   
    #Comprobaciones de division de datos
    #print(y_test)
    #print(X_train['ID'].value_counts())
    #print(X_test['ID'].value_counts())

    ###Una vez comprobado que se han dividido correctamente, eliminamos la columna ID###
    X_train = X_train.drop('ID',axis=1).copy()

    print("Se han dividido correctamente los datos")
    
#########################################################################################################################

    print ('\n[4]##############################')
    
    ###Se va a entrenar el modelo a traves de una API de Scikit-Learn --> XGBRanker###

    #Para utilizar XGBoostRanker, debemos indicar el grupo al que debe clasificar
    groups = train.groupby('ID').size().to_frame('size')['size'].to_numpy()

    ###Elegimos los hipermarametros del entrenamiento###
    
    ####MODELO 1####
    #clf_xgb=xgb.XGBRanker( booster='gbtree', objective='rank:ndcg', random_state=7, learning_rate=0.05, max_depth=6, n_estimators=100, subsample=0.5)
    
    ####MODELO 2####
    #clf_xgb=xgb.XGBRanker( tree_method='hist', booster='gbtree', objective='rank:pairwise', random_state=7, learning_rate=0.05, max_depth=6, n_estimators=100, subsample=0.75, colsample_bytree=0.9) 
    
    ####MODELO 3####
    clf_xgb=xgb.XGBRanker( tree_method='hist', booster='gbtree', objective='rank:pairwise', random_state=27, learning_rate=0.2, max_depth=6, n_estimators=120, subsample=0.65, colsample_bytree=0.9)

    ###Se entrena el modelo###
    clf_xgb.fit(X_train, y_train,group =groups)
    
    print ('Se ha entrenado el modelo')
    
#########################################################################################################################
    
    print ('\n[5]##############################')
    
    ###Se va a predecir el resultado de las carreras###
    def predict(model, df):
    	return model.predict(df.loc[:, ~df.columns.isin(['ID'])])
    
    predictions = (X_test.groupby('ID').apply(lambda x: predict(clf_xgb, x)))
    
    print ('Se ha obtenido el resultado de las carreras')
    
#########################################################################################################################

    print ('\n[6]##############################')
    
    ###Convertimos los resultados en un dataframe para guardarlos y comprarlos posteriormente###
    data_index = predictions.keys().to_list()
    data=[]
    
    for elem in predictions:
    	data.append(dict(zip(range(len(elem)),elem)))
    
    predictions_df = pd.DataFrame(data, index = data_index)
    
    ###Guardar el resultado de la prediccion y el real###
    predictions_df.to_csv('prediccion.csv', header=0)
    y_test.to_csv('resultados.csv', header=0, index=False)
    
    print ('Se ha guardado en un archivo los resultados de la prediccion')
    




