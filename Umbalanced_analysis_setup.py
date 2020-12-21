# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 11:12:36 2020

@author: Benito RC y Javier CM

Código que realiza un análisis robusto sobre 6 métodos de balanceo de clases en 19 bases de datos para clasificación binaria, usando
tres clasificadores diferentes.

"""

#Importamos librerías sobre métodos de balanceo de la librería imblearn 
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.base import BaseEstimator

#Esta librería es una implementación que realizamos, se carga directo del .py que debe estar en la misma carpeta
from adasyn2 import Adasyn

#Fijamos una semilla para tener replicabilidad de los resultados
from random import seed
seed(42)

#Clase principal que incorpora 6 métodos de balanceo de datos, 2 undersampling, 2 oversamplig, 2 híbridos
class imbalanced_handler(BaseEstimator):

    def __init__(self, method = 'ENN'):
        self.method = method

    #Undersamplig methods        
    def __None(self, X,y):
        pass

    def __ENN(self, X,y):
        balancer = EditedNearestNeighbours()
        self.ENN_ = balancer.fit_resample(X,y)
        return self
    
    def __TL(self,X,y):
        balancer = TomekLinks()
        self.TL_ = balancer.fit_resample(X,y)
        return self    

    #Oversampling methods
    def __Adasyn(self, X, y):
        balancer = Adasyn(X, y)
        self.Adasyn_ = balancer.oversampling()
        return self

    def __SM(self,X,y):
        balancer = SMOTE()
        self.SM_ = balancer.fit_resample(X,y)
        return self    

    # Hybrid Methods
    def __SM_TL(self,X,y):
        balancer = SMOTETomek()
        self.SM_TL_ = balancer.fit_resample(X,y)
        return self

    def __SM_ENN(self,X,y):
        balancer = SMOTEENN()
        self.SM_ENN_ = balancer.fit_resample(X,y)
        return self

    def fit(self, X, y):
        #Undersampling methods
        if self.method == 'ENN':
            self.__ENN(X,y)
        elif self.method == 'TL':
            self.__TL(X,y)
        #Oversampling methods            
        elif self.method == 'SM':
            self.__SM(X,y)
        elif self.method=='Adasyn':
            self.__Adasyn(X,y)
        #Hybrid methods
        elif self.method == 'SM_TL':
            self.__SM_TL(X,y)      
        elif self.method == 'SM_ENN':
            self.__SM_ENN(X,y)      
        elif self.method == 'None':
            self.__None(X,y)      
        else:
            raise ValueError('Unrecognized method')
        return self

    def transform(self, X,y):
        X_ = np.copy(X)
        y_ = np.copy(y)
        
        #Undersampling methods
        if self.method == 'ENN':
            X_,y_=self.ENN_

        elif self.method=='TL':
            X_,y_=self.TL_

        #Oversampling methods
        elif self.method=='Adasyn':
            X_,y_=self.Adasyn_

        elif self.method=='SM':
            X_,y_=self.SM_

        #Hybrid methods 
        elif self.method=='SM_TL':
            X_,y_=self.SM_TL_

        elif self.method=='SM_ENN':
            X_,y_=self.SM_ENN_
        
        elif self.method=='None':
            pass

        else:
            raise ValueError('Unrecognized method')
        return X_,y_  

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X,y)

#Líbrerías para realizar el set up que nos permitirá comparar el performance de los métodos

from sklearn.metrics import  roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

#Esta función nos permite hacer la clasificación y devuelv el score de la tarea de clasificación, 
#como parámetros toma el train y test sets, el clasificdor y el método de balanceo
def classification(X_train,y_train,X_test,y_test,clf,method,minor):
    classifier = clf
    imputer = imbalanced_handler(method = method)
    X_, y_ = imputer.fit_transform(X_train, y_train)
    y_pred = classifier.fit(X_, y_).predict(X_test)
    #print((y_pred==y_test).to_numpy()*(y_pred==minor))
    if method == "Adasyn":
      minor_score = ((y_pred==y_test)*(y_pred==minor)).sum()/(y_pred==minor).sum()
    else:
      minor_score = ((y_pred==y_test).to_numpy()*(y_pred==minor)).sum()/(y_pred==minor).sum()
    score = (y_pred==y_test).sum()/len(y_pred)
    roc_score = roc_auc_score(y_test,y_pred)
    return score,minor_score,roc_score

#Esta función realiza un repeated stratified k-fold de 2*5 para calcular una serie de scores y el área bajo la curva ROC 
#para cada método de balanceo, dado un clasificador se obtienen 10 scores por método devolviendo al final dos arreglo de 6x10  
def scores_rsfk(X,y,clf,minor):
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)
    scores1,scores2,scores3,scores4,scores5,scores6,scores7=[],[],[],[],[],[],[]
    minor_scores1,minor_scores2,minor_scores3,minor_scores4,minor_scores5,minor_scores6,minor_scores7=[],[],[],[],[],[],[]
    roc_scores1,roc_scores2,roc_scores3,roc_scores4,roc_scores5,roc_scores6,roc_scores7=[],[],[],[],[],[],[]
    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #Methods "ENN","TL","Adasyn","SMOTE","SMOTE+ENN","SMOTE+TL"
        scores_aux1,minor_scores_aux1, roc_scores_aux1 = classification(X_train,y_train,X_test,y_test,clf,"ENN",minor)
        scores_aux2,minor_scores_aux2, roc_scores_aux2 = classification(X_train,y_train,X_test,y_test,clf,"TL",minor)
        #scores_aux3,minor_scores_aux3, roc_scores_aux3 = classification(X_train,y_train,X_test,y_test,clf,"TL",minor)
        scores_aux3,minor_scores_aux3, roc_scores_aux3 = classification(np.array(X_train),np.array(y_train),np.array(X_test),np.array(y_test),clf,"Adasyn",minor)
        scores_aux4,minor_scores_aux4, roc_scores_aux4 = classification(X_train,y_train,X_test,y_test,clf,"SM",minor)
        scores_aux5,minor_scores_aux5, roc_scores_aux5 = classification(X_train,y_train,X_test,y_test,clf,"SM_ENN",minor)
        scores_aux6,minor_scores_aux6, roc_scores_aux6 = classification(X_train,y_train,X_test,y_test,clf,"SM_TL",minor)
        scores_aux7,minor_scores_aux7, roc_scores_aux7 = classification(X_train,y_train,X_test,y_test,clf,"None",minor)

        scores1 = np.append(scores1,scores_aux1)
        scores2 = np.append(scores2,scores_aux2)
        scores3 = np.append(scores3,scores_aux3)
        scores4 = np.append(scores4,scores_aux4)
        scores5 = np.append(scores5,scores_aux5)
        scores6 = np.append(scores6,scores_aux6)
        scores7 = np.append(scores7,scores_aux7)

        minor_scores1 = np.append(minor_scores1,minor_scores_aux1)
        minor_scores2 = np.append(minor_scores2,minor_scores_aux2)
        minor_scores3 = np.append(minor_scores3,minor_scores_aux3)
        minor_scores4 = np.append(minor_scores4,minor_scores_aux4)
        minor_scores5 = np.append(minor_scores5,minor_scores_aux5)
        minor_scores6 = np.append(minor_scores6,minor_scores_aux6)
        minor_scores7 = np.append(minor_scores7,minor_scores_aux7)

        roc_scores1 = np.append(roc_scores1,roc_scores_aux1)
        roc_scores2 = np.append(roc_scores2,roc_scores_aux2)
        roc_scores3 = np.append(roc_scores3,roc_scores_aux3)
        roc_scores4 = np.append(roc_scores4,roc_scores_aux4)
        roc_scores5 = np.append(roc_scores5,roc_scores_aux5)
        roc_scores6 = np.append(roc_scores6,roc_scores_aux6)
        roc_scores7 = np.append(roc_scores7,roc_scores_aux7)

    scores = np.append(scores1,scores2)    
    for i in [scores3,scores4,scores5,scores6,scores7]:
      scores = np.append(scores,i)
    scores = scores.reshape((7, 10))
    minor_scores = np.append(minor_scores1,minor_scores2)    
    for i in [minor_scores3,minor_scores4,minor_scores5,minor_scores6,minor_scores7]:
      minor_scores = np.append(minor_scores,i)
    minor_scores = minor_scores.reshape((7, 10))
    roc_scores = np.append(roc_scores1,roc_scores2)
    for i in [roc_scores3,roc_scores4,roc_scores5,roc_scores6,roc_scores7]:
      roc_scores = np.append(roc_scores,i)
    roc_scores = roc_scores.reshape((7, 10))
    
    return scores,roc_scores,minor_scores

#Esta función hace toda la chamba, toma como valores un dataframe y un clasificador, primero limpia algunas 
#posibles problemas del dataframe, separa en variable predictora y respuesta, manda a llamar a scores_rsfk
#y devuelve dos arreglos cada uno con los scores promediados del k-fold para cada método, así como la etiqueta correspondiente a la clase minoritaria

def statistics(df,clf):
    df = df.replace(" negative", 0)
    df = df.replace("negative", 0)
    df = df.replace(' negative ', 0)
    df = df.replace(' negative', 0)
    df = df.replace(" positive", 1)
    df = df.replace("positive", 1)
    df = df.replace(' positive ', 1)
    df = df.replace(' positive    ', 1)
    X = df.drop('Class', 1).to_numpy()
    y = df['Class']
    if (y==0).sum()<(y==1).sum():
      minor = 0
    else:
      minor = 1
    scores,roc_scores,minor_scores=scores_rsfk(X,y,clf,minor)
    return np.mean(roc_scores,axis=1),np.mean(scores,axis=1),np.mean(minor_scores,axis=1)

#Bloque final del código que lee los dataframes y realiza las tres tablas de resultados, presición, presición en la clase minoritaria y el área bajo las curvas ROC
    
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

files = ["Data bases/1_82_glass1-1.csv",
"Data bases/1_86_ecoli-0_vs_1-1.csv",
"Data bases/1_86_wisconsin-1.csv",
"Data bases/1_87_pima-1.csv",
"Data bases/2_0_iris0-1.csv",
"Data bases/2_06_glass0-1.csv",
"Data bases/2_46_yeast1-1.csv",
"Data bases/2_78_haberman-1.csv",
"Data bases/2_88_vehicle2-1.csv",
"Data bases/3_25_vehicle0-1.csv",
"Data bases/5_14_new-thyroid1-1.csv",
"Data bases/6_02_segment0-1.csv",
"Data bases/9_08_yeast-2_vs_4-2.csv",
"Data bases/16_0_abalone-2.csv",
"Data bases/30_yeast-1-2-8-9_vs_7-2.csv",
"Data bases/129_abalone2.csv",
"Data bases/BL_04clover5z-800-7.csv",
"Data bases/BL_04clover5z-600-5.csv",
"Data bases/BL_paw02a-800-7.csv"]

Data_frames=[]
for file in files: 
    Data_frames.append(pd.read_csv(file))

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

clf1 = DecisionTreeClassifier(random_state=42)
clf2 = SVC(kernel='linear', probability=True,max_iter=1000)
clf3 = KNeighborsClassifier(4)
classifiers=[clf1,clf2,clf3]
ACC, ROC_AUC, minor_ACC=[], [], []
i = 0
for df in Data_frames: 
  for clf in classifiers:
    roc_auc,acc,minor_acc = statistics(df,clf)
    minor_ACC.append(minor_acc)
    ACC.append(acc)
    ROC_AUC.append(roc_auc)
  print("Df",i)
  i= i+1

ACC = np.array(ACC)
minor_ACC = np.array(minor_ACC)
ROC_AUC = np.array(ROC_AUC)
table_ROC = pd.DataFrame(ROC_AUC, columns=["ENN","TL","Adasyn","SMOTE","SMOTE+ENN","SMOTE+TL","None"])
table_ROC["Classificator"]=["Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC"]
table_ROC["Database"] = ["1.82","1.82","1.82","1.86","1.86","1.86","1.86","1.86","1.86","1.87","1.87","1.87","2.0","2.0","2.0","2.06","2.06","2.06","2.46","2.46","2.46","2.78","2.78","2.78","2.88","2.88","2.88",
     "3.25","3.25","3.25","5.14","5.14","5.14","6.02","6.02","6.02","9.08","9.08","9.08","16.0","16.0","16.0","30","30","30","129","129","129",
     "BLpaw-7","BLpaw-7","BLpaw-7","BLclover-5","BLclover-5","BLclover-5","BLclover-7","BLclover-7","BLclover-7"]
table_ROC = table_ROC.set_index(['Database',"Classificator"])

table_acc = pd.DataFrame(ACC, columns=["ENN","TL","Adasyn","SMOTE","SMOTE+ENN","SMOTE+TL","None"])
table_acc["Classificator"]=["Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC"]
table_acc["Database"] = ["1.82","1.82","1.82","1.86","1.86","1.86","1.86","1.86","1.86","1.87","1.87","1.87","2.0","2.0","2.0","2.06","2.06","2.06","2.46","2.46","2.46","2.78","2.78","2.78","2.88","2.88","2.88",
     "3.25","3.25","3.25","5.14","5.14","5.14","6.02","6.02","6.02","9.08","9.08","9.08","16.0","16.0","16.0","30","30","30","129","129","129",
     "BLpaw-7","BLpaw-7","BLpaw-7","BLclover-5","BLclover-5","BLclover-5","BLclover-7","BLclover-7","BLclover-7"]
table_acc = table_acc.set_index(['Database',"Classificator"])

table_minor_acc = pd.DataFrame(minor_ACC, columns=["ENN","TL","Adasyn","SMOTE","SMOTE+ENN","SMOTE+TL","None"])
table_minor_acc["Classificator"]=["Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC","Decision Tree","SVC","KNC"]
table_minor_acc["Database"] = ["1.82","1.82","1.82","1.86","1.86","1.86","1.86","1.86","1.86","1.87","1.87","1.87","2.0","2.0","2.0","2.06","2.06","2.06","2.46","2.46","2.46","2.78","2.78","2.78","2.88","2.88","2.88",
     "3.25","3.25","3.25","5.14","5.14","5.14","6.02","6.02","6.02","9.08","9.08","9.08","16.0","16.0","16.0","30","30","30","129","129","129",
     "BLpaw-7","BLpaw-7","BLpaw-7","BLclover-5","BLclover-5","BLclover-5","BLclover-7","BLclover-7","BLclover-7"]
table_minor_acc = table_minor_acc.set_index(['Database',"Classificator"])


print("Tabla de accuracys")
print(table_acc)
print("Tabla de accuracys de la clase minoritaria")
print(table_minor_acc)
print("Tabla de AUC_ROC")
print(table_ROC)


table_acc.to_csv("ACC.csv",index=True)
table_minor_acc.to_csv("ACC_minor.csv",index=True)
table_ROC.to_csv("ROC.csv",index=True)






