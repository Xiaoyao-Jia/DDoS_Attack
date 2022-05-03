'''
-*- ecoding: utf-8 -*-
@ModuleName: ngbddos_code
@laiyuan:
@Author:XYJ
@Time: 2022/4/19 19:12
'''
from ngboost.distns import k_categorical
import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import BaggingClassifier
from deepforest import CascadeForestClassifier,CascadeForestRegressor
from sklearn.tree import DecisionTreeClassifier
import warnings
import ngboost
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import scikitplot as skplt
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTEN,RandomOverSampler,ADASYN,KMeansSMOTE,SMOTE,BorderlineSMOTE
from sklearn.ensemble import AdaBoostClassifier
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score,roc_auc_score,auc
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier as rf
import catboost as catb
import lightgbm as lgb
import pickle
from sklearn.ensemble import ExtraTreesClassifier as etc

df1 = pd.read_csv('../data/cic2019.csv',encoding='utf-8')

df = df1[~df1.isin([np.nan, np.inf, -np.inf]).any(1)]
df = df.sample(frac=1)
isDplicated=df.duplicated()
outlier_pd = pd.DataFrame(isDplicated, columns=['duplicated'])
data_merge = pd.concat((outlier_pd,df), axis=1)
df = data_merge[data_merge['duplicated']==False]
df = df.drop(['duplicated'],axis = 1)
print(df.select_dtypes(exclude=[np.number]))
hangye_mapping_1 = {
'BENIGN':0,
'DNS':1,
'LDAP':2,
'MSSQL':3,
'NTP':4,
'NetBIOS':5,
'SNMP':6,
'SSDP':7,
'UDP':8,
'Syn':9,
'WebDDoS':10,
'TFTP':11,
'UDPLag':12,
}
hangye_mapping_2 = {'BENIGN':0,
'DNS':1,
'LDAP':1,
'MSSQL':1,
'NTP':1,
'NetBIOS':1,
'SNMP':1,
'SSDP':1,
'UDP':1,
'WebDDoS':1,
'Portmap':1,
'Syn':1,
'TFTP':1,
'UDPLag':1,}
df['if_attract'] = df[' Label'].map(hangye_mapping_2)
df[' Label'] = df[' Label'].map(hangye_mapping_1)
obj = [
     ' Bwd PSH Flags',' Bwd URG Flags' ,'Fwd Avg Bytes/Bulk',' Fwd Avg Packets/Bulk',' Fwd Avg Bulk Rate'
       ,' Bwd Avg Bytes/Bulk' ,' Bwd Avg Packets/Bulk','Bwd Avg Bulk Rate'
       ]
df.drop(obj,axis=1,inplace=True)

x = df.loc[:,df.columns!=' Label']
y = df.loc[:,df.columns==' Label']

# import math
# def calc_entropy(column):
#     counts=np.bincount(column) #This code will return the number of each unique value in a column
#     probability=counts/(len(column))#here we calculate the probability of each value vy dividing the length of the entire column
#     entropy=0#we start 0 as the intial value of entropy
#     for prob in probability: # here we a for loop to go throuh each probability of each unique value in the column
#         if prob >0:
#             entropy += prob * math.log(prob, 2) # here calculate entropy of each value and add them to find the total emtropy
#     return -entropy # we should return - * entropy due to the formula
#
# def information_gain(data, split,target):
#     original_entropy=calc_entropy(data[target])
#     values=data[split].unique()
#     left_split=data[data[split]==values[0]]
#     right_split=data[data[split]==values[1]]
#     subract=0
#     for subset in [left_split,right_split]:
#         prob=(subset.shape[0])/data.shape[0]
#         subract += prob * calc_entropy(subset[target])
#     return  original_entropy - subract
#
# '''---------------------------------------------'''
# entro_ = {'information gain': {}}
# for index,i in enumerate(x.columns):
#     print(index,i)
#     tem = information_gain(df,i,' Label')
#     entro_['information gain'][i] = tem
# g = pd.DataFrame(entro_).sort_values(by='information gain',ascending=0)

selct =['Fwd IAT Total', 'Flow Bytes/s', ' Flow Duration', 'Fwd Packets/s',
       ' Flow IAT Max', ' Flow IAT Mean', ' Flow IAT Std', ' Flow Packets/s',
       ' Fwd IAT Max', ' Fwd IAT Mean', ' Fwd IAT Std', ' Subflow Fwd Bytes',
       'Total Length of Fwd Packets', ' Packet Length Mean',
       ' Avg Fwd Segment Size', ' Fwd Packet Length Mean',
       ' Max Packet Length', ' Fwd Packet Length Max', ' Average Packet Size',
       ' Fwd Packet Length Min', ' Min Packet Length', ' min_seg_size_forward',
       ' Fwd Header Length.1', ' Fwd Header Length', ' Flow IAT Min',
       ' act_data_pkt_fwd', ' Total Fwd Packets', 'Subflow Fwd Packets',
       ' Fwd IAT Min', ' Bwd Packets/s', ' Bwd Header Length',
       ' Total Backward Packets', ' Subflow Bwd Packets', 'if_attract',
       ' Packet Length Std', ' Packet Length Variance']
x = x[selct]


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=2022)
maxmin = MinMaxScaler()
X_train = maxmin.fit(X_train).transform(X_train)
X_test = maxmin.transform(X_test)

sm =SMOTEN()
x_res, y_res = sm.fit_resample(X_train,y_train)
x_res= np.array(x_res)
y_res= np.array(y_res)
print()

# base_models=[
#
#              ('XGBoost',xgb.XGBClassifier(random_state=0))
#             ,('DeepForest',CascadeForestClassifier(random_state=0,verbose=0))
#             ,('CatBoost',catb.CatBoostClassifier(verbose=0,random_state=0))
#             ,('LightGBM',lgb.LGBMClassifier(random_state=0, verbosity= -1))
#             ,('GradientBoost',GradientBoostingClassifier(random_state=0))
#             ,('RandomForest',rf(random_state=0))
#             , ('AdaBoost', AdaBoostClassifier(random_state=0))
#             , ('ExtraTree', etc(random_state=0))
#             ,('DecisionTree',DecisionTreeClassifier(random_state = 0))
#             ,("Bagging_model",BaggingClassifier())
#             ,('LogisticRegression',LogisticRegression(random_state=0))
#             ,('KNN',KNeighborsClassifier())
#             ,('SVM',SVC(probability=True,random_state = 0))
#              ]
#
# models_data = {'Accuracy': {}, 'AUC': {}, 'Precision': {}, 'Recall': {},'F1-score':{}}
# models_data_2 = {'Accuracy': {}, 'AUC': {}, 'Precision': {}, 'Recall': {},'F1-score':{}}
# for name, model in base_models:
#     print('开始'+name+'的训练：')
#     model.fit(x_res,y_res)
#     y_name = model.predict(np.array(X_test))
#     auc_name = roc_auc_score(np.array(y_test).ravel(), model.predict_proba(X_test),multi_class='ovr' )
#     print('->auc为',auc_name)
#     ans_dt = classification_report(y_test, y_name, digits=13)
#     print('->',name,ans_dt )
#     models_data_2['AUC'][name]=auc_name
#     acc_name= accuracy_score(y_test, y_name)
#     print( '-->acc为' ,acc_name)
#     models_data_2['Accuracy'][name]=acc_name
#     precision_s = precision_score(y_test, y_name,average='weighted')  # 精确度
#     print( '--->precision为', precision_s)
#     models_data_2['Precision'][name]=precision_s
#     recall_s = recall_score(y_test, y_name,average='weighted')
#     print( '---->recall为' ,recall_s)
#     models_data_2['Recall'][name]=recall_s
#     f1_s = f1_score(y_test, y_name ,average='weighted')
#     print('----->f1值为' , f1_s)
#     models_data_2['F1-score'][name]=f1_s
#
# models_df_2 = pd.DataFrame(models_data_2).sort_values(by='Accuracy',ascending=False)
# print(models_df_2)
# models_df_2.iplot(kind='bar')
# print()
#
model =ngboost.NGBClassifier(k_categorical(13),Base=xgb.XGBRegressor())
model.fit(x_res, y_res)
pickle.dump(model, open("pima.pickle.dat", "wb"))

# model = pickle.load(open("pima.pickle.dat", "rb"))

y_name = model.predict(X_test)
skplt.metrics.plot_roc(y_test, model.predict_proba(X_test))
plt.show()
auc_name = roc_auc_score(y_test, model.predict_proba(X_test),multi_class='ovr')
print('->auc为',auc_name)
acc_name= accuracy_score(y_test, y_name)
print('->auc为',auc_name)
ans_dt = classification_report(y_test, y_name, digits=13)
print( '-->acc为' ,acc_name)
precision_s = precision_score(y_test, y_name ,average='weighted')
print( '--->precision为', precision_s)
recall_s = recall_score(y_test, y_name,average='weighted')
print( '---->recall为' ,recall_s)
f1_s = f1_score(y_test, y_name,average='weighted')
print('----->f1值为' , f1_s)
print()

x_res = pd.DataFrame(x_res)
x_res.columns = selct
y_res = pd.DataFrame(y_res)
y_res.columns = ['Label']

explainer = shap.TreeExplainer(model)
shap_values_ = explainer.shap_values(x_res)
shap.summary_plot(shap_values_, x_res, max_display = 10)

shap.summary_plot(shap_values_[0], x_res, max_display = 10)
shap.summary_plot(shap_values_[1], x_res, max_display = 10)
shap.summary_plot(shap_values_[2], x_res, max_display = 10)
shap.summary_plot(shap_values_[3], x_res, max_display = 10)
shap.summary_plot(shap_values_[4], x_res, max_display = 10)
shap.summary_plot(shap_values_[5], x_res, max_display = 10)
shap.summary_plot(shap_values_[6], x_res, max_display = 10)
shap.summary_plot(shap_values_[7], x_res, max_display = 10)
shap.summary_plot(shap_values_[8], x_res, max_display = 10)
shap.summary_plot(shap_values_[9], x_res, max_display = 10)
shap.summary_plot(shap_values_[10], x_res, max_display = 10)
shap.summary_plot(shap_values_[11], x_res, max_display = 10)
shap.summary_plot(shap_values_[12], x_res, max_display = 10)