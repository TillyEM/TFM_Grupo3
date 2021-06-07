import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import train_test_split
from app import cos


def make_dataset(path, timestamp, target, model_type='RandomForest'):

    """
        Función que permite crear el dataset usado para el entrenamiento
        del modelo.

        Args:
           path (str):  Ruta hacia los datos.
           timestamp (float):  Representación temporal en segundos.
           target (str):  Variable dependiente a usar.

        Kwargs:
           model_type (str): tipo de modelo usado.

        Returns:
           DataFrame, DataFrame. Datasets de train y test para el modelo.
    """

    print('---> Getting data')
    df = get_raw_data_from_local(path)
    print('---> Train / test split')
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(df, test_size=0.2, stratify=y, random_state=50)

    print('---> Transforming data')
    X_train, X_test = transform_data(X_train, X_test, timestamp)

    return X_train.copy(), y_train.copy(), y_train.copy(), y_test.copy()


def get_raw_data_from_local(path):

    """
        Función para obtener los datos originales desde local

        Args:
           path (str):  Ruta hacia los datos.

        Returns:
           DataFrame. Dataset con los datos de entrada.
    """

    df = pd.read_csv(path)
    return df.copy()


    
def transform_data(X_train, X_test, timestamp, cols_to_remove,columns_to_drop,indDes,gen,exprel,uni,educ,exp,tam,tipo,ultNT,horas):

    """
        Función que permite realizar las primeras tareas de transformación
        de los datos de entrada.

        Args:
           train_df (DataFrame):  Dataset de train.
           test_df (DataFrame):  Dataset de test.
           timestamp (float):  Representación temporal en segundos.
           target (str):  Variable dependiente a usar.
           cols_to_remove (list): Columnas a retirar.

        Returns:
           DataFrame, DataFrame. Datasets de train y test para el modelo.
    """

    # Quitando columnas no usables # 'empleado_id', 'ciudad'
    print('------> Removing unnecessary columns')
    X_train = remove_unwanted_columns(X_train, cols_to_remove)
    X_test = remove_unwanted_columns(X_test, cols_to_remove)


    # generación de configuracione de X_train
    print('------> Encoding data')
    trans_IndDes = ('tIndDes', tIndDes(tipo=indDes)) 
    trans_tGen = ('tGen', tGen(tipo=gen))  
    trans_tExpRel = ('tExpRel', tExpRel(tipo=exprel)) 
    trans_tUni = ('tUniversidad', tUniversidad(tipo=uni)) 
    trans_tNEdu = ('tNEducacion', tNEducacion(tipo=educ))   
    trans_tEdu = ('tEdu', tEdu(tipo='One-Hot'))   
    trans_tExp = ('tExperiencia', tExperiencia(tipo=exp)) 
    trans_tTamComp = ('tTamComp', tTamComp(tipo=tam)) 
    trans_tTipComp = ('tTipComp', tTipComp(tipo=tipo))   
    trans_tUltNT = ('tUltNT', tUltNT(tipo=ultNT)) 
    trans_tHoras = ('tHoras', tHoras(tipo=horas))

    # En train_model: model_config
    # columns_to_drop = ['genero','experiencia_relevante', 'universidad_matriculado', 'nivel_educacion','educacion', 'experiencia', 'tamano_compania', 'tipo_compania','ultimo_nuevo_trabajo']

    trans_Drop = ('DropColumns', DropColumns(columns_to_drop))

   ### Lista de operaciones
    pipe_steps = [trans_IndDes, trans_tGen, trans_tExpRel, trans_tUni, trans_tNEdu, trans_tEdu, 
             trans_tExp, trans_tTamComp, trans_tTipComp, trans_tUltNT,trans_tHoras, trans_Drop]
   ### Creo el objeto Pipeline
    data_pipe = Pipeline(pipe_steps)

    X_train_trans = data_pipe.fit_transform(X_train)
    X_test_trans = data_pipe.fit_transform(X_test)
    #df_dc_test_FT = data_pipe.transform(X_test)

    # guardando las columnas resultantes en IBM COS
    print('---------> Saving encoded columns')
    cos.save_object_in_cos(X_train_trans.columns, 'encoded_columns', timestamp)


    return X_train_trans.copy(), X_test_trans.copy()




def remove_unwanted_columns(df, cols_to_remove):
    """
        Función para quitar variables innecesarias

        Args:
           df (DataFrame):  Dataset.

        Returns:
           DataFrame. Dataset.
    """
    return df.drop(columns=cols_to_remove)




def trans_clave_valor(df,col,clave,valor):
    df_temp = df[col].copy()
    for i in range(len(clave)):
        df_temp.loc[(df[col]==clave[i])] = valor[i]
        
    return df_temp

## Funciones para Pipeline

### aux functions

class SelectColumns(TransformerMixin):
    def __init__(self, columns: list) -> pd.DataFrame:
        if not isinstance(columns, list):
            raise ValueError('Specify the columns into a list')
        self.columns = columns
    def fit(self, X, y=None): # we do not need to specify the target in the transformer. We leave it as optional arg for consistency
        return self
    def transform(self, X):
        return X[self.columns]
    
class DropColumns(TransformerMixin):
    def __init__(self, columns: list) -> pd.DataFrame:
        if not isinstance(columns, list):
            raise ValueError('Specify the columns into a list')
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(self.columns, axis=1)
    


# Pipelines


### Indice de desarrollo de la ciudad
class tIndDes(TransformerMixin): 
    
    # 'eq' para dejarlo igual
    # 'log' para el log de la variable   
    
    def __init__(self, tipo='eq') -> pd.DataFrame:
        self.tipo = tipo
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if self.tipo == 'eq': 
            X['tIndDes'] = X['indice_desarrollo_ciudad'] 
            return X
        elif self.tipo == 'log':
            X['tIndDes'] = np.log(X[['indice_desarrollo_ciudad']])  
            return X
        else :
            print('ERROR definition tIndDes.tipo')
            
    def set_params(self, **parameters):    # Necesario por error en GridSearch
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self      

### Genero
class tGen(TransformerMixin): 
    
    # 'des' funcion desaparecido de Paula
    # '0.5' 0.5 para Other y para NaN 1 Fem, 0 Male
    
    def __init__(self, tipo='des') -> pd.DataFrame:
        self.tipo = tipo
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if self.tipo == 'des':    # Funcion desaparecido --> hot encoding para esto, no?
            X['tGen'] = X['genero'].replace(np.nan, "desconocido", regex=True)
            return X
        elif self.tipo == '0.5':    # Funcion desaparecido
            clave = ['Male', 'Female', 'Other']
            valor = [0, 1, 0.5]
            X['tGen'] = trans_clave_valor(X,'genero', clave, valor).replace(np.nan, 0.5, regex=True)
            return X
        elif self.tipo == '0.3':    # Funcion desaparecido
            clave = ['Male', 'Female', 'Other']
            valor = [0, 1, 0.3]
            X['tGen'] = trans_clave_valor(X,'genero', clave, valor).replace(np.nan, 0.3, regex=True)         
            return X

        else :
            print('ERROR definition tGen.tipo')
    def set_params(self, **parameters):    # Necesario por error en GridSearch
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self   



### Experiencia relevante

class tExpRel(TransformerMixin): 
    def __init__(self, tipo='des') -> pd.DataFrame:
        self.tipo = tipo
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if self.tipo == 'des':    # Funcion desaparecido
            clave = ['Has relevent experience', 'No relevent experience']
            valor = [1, 0]
            X['tExpRel'] = trans_clave_valor(X,'experiencia_relevante', clave, valor) 
            return X
        else :
            print('ERROR definition tExpRel.tipo')
    def set_params(self, **parameters):    # Necesario por error en GridSearch
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self   

### Universidad matriculado

class tUniversidad(TransformerMixin): 
    def __init__(self, tipo='c_v') -> pd.DataFrame:
        self.tipo = tipo
        self.imp = SimpleImputer(missing_values=np.nan, strategy='median' )
        
    def fit(self, X, y=None):
        if self.tipo == 'c_v_NaN_median':  
            clave = ['no_enrollment', 'Full time course', 'Part time course']
            valor = [0, 2, 1]    
            self.imp.fit(trans_clave_valor(X,'universidad_matriculado', clave, valor).values.reshape(-1,1))
        if self.tipo == 'c_vN_NaN_median':  
            clave = ['no_enrollment', 'Full time course', 'Part time course']
            valor = [0, 1, 0.5]    
            self.imp.fit(trans_clave_valor(X,'universidad_matriculado', clave, valor).values.reshape(-1,1))

        return self
    def transform(self, X):
        if self.tipo == 'c_v':    # Transformación graduada
            clave = ['no_enrollment', 'Full time course', 'Part time course']
            valor = [0, 2, 1] 
            X['tUniv'] = trans_clave_valor(X,'universidad_matriculado', clave, valor) 
            return X
        elif self.tipo == 'c_v_NaN_median':    # Transformación graduada
            clave = ['no_enrollment', 'Full time course', 'Part time course']
            valor = [0, 2, 1]             
            X['tUniv'] = self.imp.transform(trans_clave_valor(X,'universidad_matriculado', clave, valor).values.reshape(-1,1))            
            return X
        elif self.tipo == 'c_vN_NaN_median':    # Transformación graduada
            clave = ['no_enrollment', 'Full time course', 'Part time course']
            valor = [0, 1, 0.5]             
            X['tUniv'] = self.imp.transform(trans_clave_valor(X,'universidad_matriculado', clave, valor).values.reshape(-1,1))            
            return X
        

        
        else :
            print('ERROR definition tUniversidad.tipo')
    def set_params(self, **parameters):    # Necesario por error en GridSearch
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self       


### Nivel educacion

class tNEducacion(TransformerMixin): 
    def __init__(self, tipo='c_v') -> pd.DataFrame:
        self.tipo = tipo
        self.imp = SimpleImputer(missing_values=np.nan, strategy='median' )
        
    def fit(self, X, y=None):
        if self.tipo == 'c_v_NaN_median': 
            clave = ['Graduate', 'Masters', 'High School', 'Phd', 'Primary School']
            valor = [ 3, 4,2 ,5 ,1 ]            
            self.imp.fit(trans_clave_valor(X,'nivel_educacion', clave, valor).values.reshape(-1,1))
        elif self.tipo == 'c_vN_NaN_median': 
            clave = ['Graduate', 'Masters', 'High School', 'Phd', 'Primary School']
            valor = [ 0.5, 0.75, 0.25, 1, 0]            
            self.imp.fit(trans_clave_valor(X,'nivel_educacion', clave, valor).values.reshape(-1,1))

            
            
        return self
    
    def transform(self, X):
        if self.tipo == 'c_v':    # Transformación graduada
            clave = ['Graduate', 'Masters', 'High School', 'Phd', 'Primary School']
            valor = [ 3, 4,2 ,5 ,1 ]
            X['tNEdu'] = trans_clave_valor(X,'nivel_educacion', clave, valor) 
            return X
        elif self.tipo == 'c_v_NaN_median':    # Transformación graduada
            clave = ['Graduate', 'Masters', 'High School', 'Phd', 'Primary School']
            valor = [ 3, 4,2 ,5 ,1 ]    
            X['tNEdu'] = self.imp.transform(trans_clave_valor(X,'nivel_educacion', clave, valor).values.reshape(-1,1))            
            return X
        elif self.tipo == 'c_vN_NaN_median':    # Transformación graduada
            clave = ['Graduate', 'Masters', 'High School', 'Phd', 'Primary School']
            valor = [ 0.5, 0.75, 0.25, 1, 0]   
            X['tNEdu'] = self.imp.transform(trans_clave_valor(X,'nivel_educacion', clave, valor).values.reshape(-1,1))            
            return X
        
        else :
            print('ERROR definition tNEducacion.tipo')
    def set_params(self, **parameters):    # Necesario por error en GridSearch
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self  

### Educacion

class tEdu(TransformerMixin): 
    def __init__(self, tipo='des') -> pd.DataFrame:
        self.tipo = tipo
        self.enc = OneHotEncoder(sparse=False, handle_unknown='error')
        
    def fit(self, X, y=None):
        if self.tipo == 'One-Hot':                
            arr = X['educacion'].replace(np.nan, "NaN", regex=True)
            self.enc.fit(arr.values.reshape(-1,1))   
        return self
    
    def transform(self, X):
        if self.tipo == 'des':    # Funcion desaparecido
            X['tEdu'] = X['educacion'].replace(np.nan, "desconocido", regex=True)
            return X
        elif self.tipo == 'One-Hot':                
            arr = X['educacion'].replace(np.nan, "NaN", regex=True)
            matr = self.enc.transform(arr.values.reshape(-1,1))
            cat = self.enc.categories_[0]
            j = 0
            for i in cat: 
                X[['tEdu_' + i]] = matr[:,j]
                j = j+1
            
            return X
        else :
            print('ERROR definition tEdu.tipo')
    def set_params(self, **parameters):    # Necesario por error en GridSearch
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self   

# Experiencia

class tExperiencia(TransformerMixin): 
    def __init__(self, tipo='to_num') -> pd.DataFrame:
        self.tipo = tipo
        self.imp = SimpleImputer(missing_values=np.nan, strategy='median')
        self.ss = StandardScaler()
        self.qt = QuantileTransformer(output_distribution='normal', random_state=0)
        
    def fit(self, X, y=None):
        if self.tipo == 'to_num_NaN_median':    # Transformación graduada        
            temp_df = X['experiencia'].copy()
            temp_df.iloc[temp_df=='>20']=21     # Mas de 20 en 21
            temp_df.iloc[temp_df=='<1']=0     # Menos de 1 en 0
            self.imp.fit(temp_df.values.astype(float).reshape(-1,1))  
            
        elif self.tipo == 'to_num_NaN_medianSS':              
            temp_df = X['experiencia'].copy()
            temp_df.iloc[temp_df=='>20']=21     # Mas de 20 en 21
            temp_df.iloc[temp_df=='<1']=0     # Menos de 1 en 0
            temp_df = self.imp.fit_transform(temp_df.values.astype(float).reshape(-1,1)) 
            self.ss.fit(temp_df)  
            
        elif self.tipo == 'to_num_NaN_medianQT':              
            temp_df = X['experiencia'].copy()
            temp_df.iloc[temp_df=='>20']=21     # Mas de 20 en 21
            temp_df.iloc[temp_df=='<1']=0     # Menos de 1 en 0
            temp_df = self.imp.fit_transform(temp_df.values.astype(float).reshape(-1,1)) 
            self.qt.fit(temp_df)  
            
        return self
    def transform(self, X):
        if self.tipo == 'to_num':    # Transformación graduada
            temp_df = X['experiencia'].copy()
            temp_df.iloc[temp_df=='>20']=21     # Mas de 20 en 21
            temp_df.iloc[temp_df=='<1']=0     # Menos de 1 en 0
            X['tExp'] = temp_df.astype(float) 
            return X
        
        elif self.tipo == 'to_num_NaN_median':         
            temp_df = X['experiencia'].copy()
            temp_df.iloc[temp_df=='>20']=21     # Mas de 20 en 21
            temp_df.iloc[temp_df=='<1']=0     # Menos de 1 en 0
            X['tExp'] = self.imp.transform(temp_df.values.astype(float).reshape(-1,1)) 

            return X
        
        elif self.tipo == 'to_num_NaN_medianSS':  
            temp_df = X['experiencia'].copy()
            temp_df.iloc[temp_df=='>20']=21     # Mas de 20 en 21
            temp_df.iloc[temp_df=='<1']=0     # Menos de 1 en 0                          
            temp_df = self.imp.transform(temp_df.values.astype(float).reshape(-1,1))
            X['tExp'] = self.ss.transform(temp_df)      
            
            return X

        elif self.tipo == 'to_num_NaN_medianQT':  
            temp_df = X['experiencia'].copy()
            temp_df.iloc[temp_df=='>20']=21     # Mas de 20 en 21
            temp_df.iloc[temp_df=='<1']=0     # Menos de 1 en 0                          
            temp_df = self.imp.transform(temp_df.values.astype(float).reshape(-1,1))
            X['tExp'] = self.qt.transform(temp_df)      
            
            return X

        else :
            print('ERROR definition tExperiencia.tipo')
    def set_params(self, **parameters):    # Necesario por error en GridSearch
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self 

### Tamaño de la compañia

class tTamComp(TransformerMixin): 
    def __init__(self, tipo='c_v') -> pd.DataFrame:
        self.tipo = tipo
        self.imp = SimpleImputer(missing_values=np.nan, strategy='median' )
        self.ss = StandardScaler()
        
    def fit(self, X, y=None):
        if self.tipo == 'c_v_NaN_median':    # Transformación graduada
            clave = ['<10', '10/49', '50-99', '100-500', '500-999', '1000-4999', '5000-9999', '10000+'  ]
            valor = [1, 2, 3, 4, 5, 6, 7, 8]
            temp_df = trans_clave_valor(X,'tamano_compania', clave, valor)    
            self.imp.fit(temp_df.values.reshape(-1,1))
            
        elif self.tipo == 'c_vN_NaN_median':    # Transformación graduada
            clave = ['<10', '10/49', '50-99', '100-500', '500-999', '1000-4999', '5000-9999', '10000+'  ]
            valor = [0/7, 1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 7/7]
            temp_df = trans_clave_valor(X,'tamano_compania', clave, valor)    
            self.imp.fit(temp_df.values.reshape(-1,1))
            
            
        elif self.tipo == 'c_v_NaN_medianSS':    # Transformación graduada
            clave = ['<10', '10/49', '50-99', '100-500', '500-999', '1000-4999', '5000-9999', '10000+'  ]
            valor = [1, 2, 3, 4, 5, 6, 7, 8]
            temp_df = trans_clave_valor(X,'tamano_compania', clave, valor)    
            temp_df = self.imp.fit_transform(temp_df.values.reshape(-1,1))
            self.ss.fit(temp_df)
            
        return self
    
    def transform(self, X):
        if self.tipo == 'des':    # Funcion desaparecido
            X['tTamComp'] = X['tamano_compania'].replace(np.nan, "desconocido", regex=True)
            return X
        elif self.tipo == 'c_v':    # Transformación graduada
            clave = ['<10', '10/49', '50-99', '100-500', '500-999', '1000-4999', '5000-9999', '10000+'  ]
            valor = [1, 2, 3, 4, 5, 6, 7, 8]
            X['tTamComp'] = trans_clave_valor(X,'tamano_compania', clave, valor)
            return X
        elif self.tipo == 'des_c_v':    # Transformación graduada
            clave = ['<10', '10/49', '50-99', '100-500', '500-999', '1000-4999', '5000-9999', '10000+'  ]
            valor = [1, 2, 3, 4, 5, 6, 7, 8]
            X['tTamComp'] = trans_clave_valor(X,'tamano_compania', clave, valor).replace(np.nan, "desconocido", regex=True)
            return X
        elif self.tipo == 'c_v_NaN_median':    # Transformación graduada
            clave = ['<10', '10/49', '50-99', '100-500', '500-999', '1000-4999', '5000-9999', '10000+'  ]
            valor = [1, 2, 3, 4, 5, 6, 7, 8]
            X['tTamComp'] = self.imp.transform(trans_clave_valor(X,'tamano_compania', clave, valor).values.reshape(-1,1))
            return X
        elif self.tipo == 'c_vN_NaN_median':    # Transformación graduada
            clave = ['<10', '10/49', '50-99', '100-500', '500-999', '1000-4999', '5000-9999', '10000+'  ]
            valor = [0/7, 1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 7/7]
            X['tTamComp'] = self.imp.transform(trans_clave_valor(X,'tamano_compania', clave, valor).values.reshape(-1,1))
            return X
        
        elif self.tipo == 'c_v_NaN_medianSS':    # Transformación graduada
            clave = ['<10', '10/49', '50-99', '100-500', '500-999', '1000-4999', '5000-9999', '10000+'  ]
            valor = [1, 2, 3, 4, 5, 6, 7, 8]
            temp_df = self.imp.transform(trans_clave_valor(X,'tamano_compania', clave, valor).values.reshape(-1,1))
            X['tTamComp'] = self.ss.transform(temp_df) 
            
            return X

        
        
        
        else :
            print('ERROR definition tTamComp.tipo')
    def set_params(self, **parameters):    # Necesario por error en GridSearch
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self  


### Tipo de compañia

class tTipComp(TransformerMixin): 
    def __init__(self, tipo='des') -> pd.DataFrame:
        self.tipo = tipo
        self.enc = OneHotEncoder(sparse=False, handle_unknown='error')
        
    def fit(self, X, y=None):
        if self.tipo == 'One-Hot':                
            arr = X['tipo_compania'].replace(np.nan, "NaN", regex=True)
            self.enc.fit(arr.values.reshape(-1,1))   

        return self
    
    def transform(self, X):
        if self.tipo == 'des':    # Funcion desaparecido
            X['tTipComp'] = X['tipo_compania'].replace(np.nan, "desconocido", regex=True)
            return X
        elif self.tipo == 'One-Hot':                
            arr = X['tipo_compania'].replace(np.nan, "NaN", regex=True)
            matr = self.enc.transform(arr.values.reshape(-1,1))
            cat = self.enc.categories_[0]
            j = 0
            for i in cat: 
                X[['tTipComp_' + i]] = matr[:,j]
                j = j+1
                
            return X
   
        else :
            print('ERROR definition tTipComp.tipo')
    def set_params(self, **parameters):    # Necesario por error en GridSearch
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self   


### Ultimo nuevo trabajo
class tUltNT(TransformerMixin): 
    def __init__(self, tipo='c_v') -> pd.DataFrame:
        self.tipo = tipo
        self.imp = SimpleImputer(missing_values=np.nan, strategy='median' )
    def fit(self, X, y=None):        
        if self.tipo == 'c_v_NaN_median':    # Transformación graduada
            clave = ['1', '2', '3', '4', '>4', 'never']
            valor = [1, 2, 3, 4, 6, 9]
            temp_df = trans_clave_valor(X,'ultimo_nuevo_trabajo', clave, valor)    
            self.imp.fit(temp_df.values.reshape(-1,1))
        if self.tipo == 'c_vN_NaN_median':    # Transformación graduada
            clave = ['1', '2', '3', '4', '>4', 'never']
            valor = [0/8, 1/8, 2/8, 3/8, 5/8, 8/8]
            temp_df = trans_clave_valor(X,'ultimo_nuevo_trabajo', clave, valor)    
            self.imp.fit(temp_df.values.reshape(-1,1))

            
        return self
    def transform(self, X):
        if self.tipo == 'c_v':    # Transformación graduada
            clave = ['1', '2', '3', '4', '>4', 'never']
            valor = [1, 2, 3, 4, 5, 0]
            X['tUltNT'] = trans_clave_valor(X,'ultimo_nuevo_trabajo', clave, valor)
            return X
        if self.tipo == 'c_v_NaN_median':    # Transformación graduada
            clave = ['1', '2', '3', '4', '>4', 'never']
            valor = [1, 2, 3, 4, 6, 9]
            temp = trans_clave_valor(X,'ultimo_nuevo_trabajo', clave, valor)
            X['tUltNT'] = self.imp.transform(temp.values.reshape(-1,1))
            return X
        if self.tipo == 'c_vN_NaN_median':    # Transformación graduada
            clave = ['1', '2', '3', '4', '>4', 'never']
            valor = [0/8, 1/8, 2/8, 3/8, 5/8, 8/8]
            temp = trans_clave_valor(X,'ultimo_nuevo_trabajo', clave, valor)
            X['tUltNT'] = self.imp.transform(temp.values.reshape(-1,1))
            return X
        
        else :
            print('ERROR definition tUltNT.tipo')
    def set_params(self, **parameters):    # Necesario por error en GridSearch
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self  



### Horas de formacion
class tHoras(TransformerMixin): 
    
    # 'eq' para dejarlo igual
    # 'log' para el log de la variable   
   
    def __init__(self, tipo='eq') -> pd.DataFrame:
        self.tipo = tipo
        self.ss = StandardScaler() 
        
    def fit(self, X, y=None):
        if self.tipo == 'eqSS': 
            self.ss.fit(X['horas_formacion'].values.reshape(-1, 1) )
        elif self.tipo == 'logSS': 
            temp = np.log(X[['horas_formacion']]) 
            self.ss.fit(temp)

        return self
                        
    def transform(self, X):
        if self.tipo == 'eq': 
            X['tHoras'] = X['horas_formacion'] 
            return X
        elif self.tipo == 'eqSS': 
            X['tHoras'] = self.ss.transform(X['horas_formacion'].values.reshape(-1, 1) )
            return X       
        elif self.tipo == 'log':
            X['tHoras'] = np.log(X[['horas_formacion']])  
            return X
        elif self.tipo == 'logSS':
            temp = np.log(X[['horas_formacion']]) 
            X['tHoras'] = self.ss.transform(temp)  
            return X

        else :
            print('ERROR definition tHoras.tipo')
            
    def set_params(self, **parameters):    # Necesario por error en GridSearch
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self  