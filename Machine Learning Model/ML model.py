import pandas as pd 
import numpy as np
import scipy 

import librosa
import librosa.display
import IPython

import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 
sns.set()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score
from sklearn.feature_selection import SelectKBest, f_classif

from catboost import CatBoostClassifier

# Set B
sb=pd.read_csv('E:/Final Project/set_b.csv')
sb.fname=sb.fname.str.replace('Btraining_','')
sb.head()
sb.shape
sb.dropna(subset=['label'],inplace=True)

# Set A
sa=pd.read_csv('E:/Final Project/set_a.csv')
sa.head()
sa.fname=sa.fname.str.replace('__','_')
sa.dropna(subset=['label'],inplace=True)

# # Set A timing
sa_timing=pd.read_csv('E:/Final Project/set_a_timing.csv')
sa_timing.head()

# # Concat and Cleaning
hb=pd.concat([sa,sb],axis=0)
hb.reset_index(inplace=True)
hb.head()
hb.sublabel.fillna('None',inplace=True)
hb['total']=np.where(hb.sublabel!='None',hb.label+'_'+hb.sublabel+'_',hb.label+'_')
hb.head()
hb.total.isna().sum()

np.where(hb.sublabel=='None',hb.fname.replace(str(hb.total), str(hb.total)+"_"),hb.fname)
hb.fname=hb.apply(lambda x: x.fname.replace(x.total, x.total+"_") if x.sublabel=='None' else x.fname, axis=1)
hb.head()

# Extract features for all files
def ExtractFeatures(audio):
    file_path=f'E:/Final Project/{audio}'
    x=librosa.load(file_path,sr=44000)
    mfcc=pd.DataFrame(librosa.feature.mfcc(x[0],sr=44000)).T
    result=np.mean(mfcc)
    return result

df=pd.DataFrame(columns=range(20))
for i,j in enumerate(hb.fname):
    df.loc[i]=ExtractFeatures(j)
cols=['X'+str(i) for i in range(20)]
df.columns=cols
#df.head()
dfinal=pd.concat([hb,df],axis=1)
dfinal.drop(['index','dataset','fname','sublabel','total'],axis=1,inplace=True)

# Prepare for modelling
# Relabelling target column

#dfinal.head()
dfinal.label.value_counts()
dct={'normal':1,
    'murmur':0,        
    'extrastole':0,    
    'artifact':0,   
    'extrahls':0}
dfinal.label=dfinal.label.map(dct)
dfinal.label.value_counts()

# Standard scaling
dfinal.drop('label',axis=1).describe().T
dfinal.corr()['label']
#plt.figure(figsize=(20,20))
#for i,col in enumerate(dfinal.columns,1):
#    plt.subplot(7,3,i)
#    sns.kdeplot(dfinal[col])
#plt.show()
X=dfinal.drop('label',axis=1)
y=dfinal.label
scaler=StandardScaler()
scaler.fit(X)
X=pd.DataFrame(scaler.transform(X),index=X.index,columns=X.columns)
#plt.figure(figsize=(20,20))

#for i,col in enumerate(X.columns,1):
#    plt.subplot(7,3,i)
#    sns.kdeplot(X[col])
#plt.show()


# Feature selection
X=dfinal.drop('label',axis=1)
y=dfinal.label
selector=SelectKBest(f_classif, k='all')
selection=selector.fit_transform(X,y)


# --Models--
# split
X_train, X_test,y_train,y_test = train_test_split(selection, dfinal.label, test_size=0.2, random_state=42, stratify=dfinal.label)

# Catboost
model=CatBoostClassifier()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
conf=confusion_matrix(y_test, [eval(str(i)) for i in y_pred])
acc=accuracy_score(y_test, [eval(str(i)) for i in y_pred])
rec=recall_score(y_test, [eval(str(i)) for i in y_pred])
pre=precision_score(y_test, [eval(str(i)) for i in y_pred])
f1=f1_score(y_test, [eval(str(i)) for i in y_pred])
print(conf)
print('Accuracy', acc)
print('Recall', rec)
print('Precision', pre)
print('F1', f1)


model = CatBoostClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('Acurracy of train ',(model.score(X_train, y_train)*100))
print('Acurracy of test ',(accuracy_score(y_test, y_pred)*100))

import pickle
pickle.dump(model, open('model.pkl', 'wb'))