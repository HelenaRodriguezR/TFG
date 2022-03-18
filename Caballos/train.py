import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

if __name__ == "__main__":

#Importar los datos para trabajar con ellos
    df = pd.read_csv('datos.csv', delimiter=',',encoding="utf-8", header=0, names=['ID','CABALLO','POSICION','EDAD','KG','JINETE','CUADRA','PREPARADOR','CAJÓN','HIPODROMO','DISTANCIA','PISTA'])
#    print (df)
    print ('\n[1]##############################')
    print('Se han importado correctamente los datos')


    print ('\n[2]##############################')
    #Con XGBoots no podemos tener datos categoricos, por lo que vamos a realizar una codificacion en caliente
    df = pd.get_dummies(df,columns=['CABALLO','JINETE','CUADRA','PREPARADOR','HIPODROMO','PISTA'])


#    print(df)
    print('Codificacion en caliente realizada')


    print ('\n[3]##############################')
    #Queremos dividir los datos y que cada id se encuentre completo por lo que utilizamos GroupShuffleSplit

    gss = GroupShuffleSplit(train_size=.60, n_splits=1, random_state = 7).split(df, groups=df['ID'])

    train_inds, test_inds = next(gss)

    train= df.iloc[train_inds]
    X_train = train.loc[:, train.columns!='POSICION']
    y_train = train['POSICION'].copy()

    test= df.iloc[test_inds]

    X_test = test.loc[:, train.columns!='POSICION']
    y_test = test[['ID','POSICION']].copy()

#    print(X_train['ID'].value_counts())
#    print(X_test['ID'].value_counts())

    #Una vez comprobado que se han dividido correctamente, eliminamos la columna ID
    X_train = X_train.drop('ID',axis=1).copy()

    print("Se han dividido correctamente los datos")

    print ('\n[4]##############################')
     #Vamos a entrenar el modelo

    #Para utilizar XGBoostRanker, debemos indicar el grupo al que debe clasificar
    groups = train.groupby('ID').size().to_frame('size')['size'].to_numpy()

    #Elegimos los hipermarametros del entrenamiento
    clf_xgb=xgb.XGBRanker( booster='gbtree', objective='rank:ndcg', random_state=7, learning_rate=0.05, max_depth=6, n_estimators=100, subsample=0.5)

    clf_xgb.fit(X_train, y_train,group =groups)
    
    print ('Se ha entrenado el modelo')
    
    print ('\n[5]##############################')
    #Se va a predecir el resultado de las carreras
    def predict(model, df):
    	return model.predict(df.loc[:, ~df.columns.isin(['ID'])])
    
    predictions = (X_test.groupby('ID').apply(lambda x: predict(clf_xgb, x)))
    
    print ('Se ha obtenido el resultado de las carreras')
    
        
    print ('\n[6]##############################')
    	#Convertimos la serie en un dataframe
    data_index = predictions.keys().to_list()
    data=[]
    
    for elem in predictions:
    	data.append(dict(zip(range(len(elem)),elem)))
    
    predictions_df = pd.DataFrame(data, index = data_index)
    predictions_df.to_csv('prediccion.csv', header=0, index= data_index)
    print ('Se ha guardado en un archivo los resultados de la prediccion')
    print (predictions_df)
    



