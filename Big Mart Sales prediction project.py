import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import normaltest
import seaborn as sns
#from scipy.stats import Sharpiro
#----------- Data Loading 
traing_set=pd.read_csv('train.csv')
testing_set=pd.read_csv('test.csv')
#----------- use Some of functions to get preview of dataset metadata

print(traing_set.columns)
print(traing_set.head())
print(traing_set.info())
print(traing_set.describe())

#Concate The two Data Sets to Apply Some Statistical methods on it %like union all in SQL 
Dataset=pd.concat([traing_set ,testing_set] , axis=0).reset_index(drop=True)
#-----------build a function To detect Outliers Using the Interquartile Range Statistical Rule 

def detect_outlier(features , Data):
    outlier_index=[]
    for each in features:
        Q1=np.percentile(Data[each] , 25)
        Q3=np.percentile(Data[each] , 75)
        IQR=Q3-Q1
        Min_Quartile=Q1-1.5*IQR
        Max_Quartile=Q3+1.5*IQR
        Outliers=Data[(Data[each] <Min_Quartile ) | (Data[each]>Max_Quartile)].index  #indexing using boolean condition 
        outlier_index.extend(Outliers)
    Outlier_index=Counter(Outliers)
    Outlier_Data=list( i for i , n in Outlier_index.items() if n>3 ) #In the case of the Feature have more than 3 datapoints Detected as an outliers we will Append it to A outlier list 
    return Outlier_Data
outlier_data=detect_outlier(['Item_Visibility' , 'Item_Weight' ,'Item_MRP' ,'Outlet_Establishment_Year'], Dataset )
print(Dataset.iloc[outlier_data])
Dataset=Dataset.drop(outlier_data , axis=0).reset_index(drop=True)
#Do some Visualization and Some Feature Analysis
#ploting the relation between All of the categorical independant variable and dependant variable

Group=Dataset[['Item_Type','Item_Outlet_Sales']].groupby(Dataset['Item_Type'] ,as_index=True ).sum() #To return an accurate Values rather than gussing when ploting the next factorplot
sns.factorplot(x='Item_Type' , y='Item_Outlet_Sales' , data=Dataset , kind ='bar' ,size=5)
plt.show()
sns.factorplot(x='Item_Fat_Content' , y='Item_Outlet_Sales' , data=Dataset , kind ='bar' ,size=5)
plt.show()
sns.factorplot(x='Outlet_Size' , y='Item_Outlet_Sales' , data=Dataset , kind ='bar' ,size=5)
plt.show()
sns.factorplot(x='Outlet_Type' , y='Item_Outlet_Sales' , data=Dataset , kind ='bar' ,size=7)
plt.show()
sns.factorplot(x='Outlet_Location_Type' , y='Item_Outlet_Sales' , data=Dataset , kind ='bar' ,size=3)
plt.show()
'''G=sns.FacetGrid(data=Dataset , row ='Item_Outlet_Sales')
G.map(sns.distplot , 'Item_MRP' , bins=25)
plt.show()'''
#processing Categorical Data to figure out the heat map pf feature and determine how far they are corrolated
# List of columns with their unique value in the Training DataFrame
Item_Fat_Content_Categories=Dataset['Item_Fat_Content'].unique()
Item_Type_Categories=Dataset['Item_Type'].unique()
Outlet_Size_Categories=Dataset['Outlet_Size'].unique()
Outlet_Type_Categories=Dataset['Outlet_Type'].unique()
Outlet_Location_Type_Categories=Dataset['Outlet_Location_Type'].unique()
# we will loop through all Categorical Variables and replace its values with A numerical ones
Dataset['Item_Fat_Content'].replace(Item_Fat_Content_Categories ,list(np.arange(len(Item_Fat_Content_Categories))) , inplace=True)
Dataset['Item_Type'].replace(Item_Type_Categories ,list(np.arange(len(Item_Type_Categories))) , inplace=True)
Dataset['Outlet_Size'].replace(Outlet_Size_Categories ,list(np.arange(len(Outlet_Size_Categories))) , inplace=True)
Dataset['Outlet_Type'].replace(Outlet_Type_Categories ,list(np.arange(len(Outlet_Type_Categories))) , inplace=True)
Dataset['Outlet_Location_Type'].replace(Outlet_Location_Type_Categories ,list(np.arange(len(Outlet_Location_Type_Categories))) , inplace=True)
#Correlation Between features
sns.heatmap(Dataset[['Item_Fat_Content' ,'Outlet_Establishment_Year' ,'Outlet_Size' ,'Outlet_Type' ,'Outlet_Location_Type' ,'Item_MRP'  ,'Item_Outlet_Sales']].corr() , annot=True)
plt.show()
#----------We have to do Normality test to our Dataset in order to dermine
#which statistical methods we to use-------------
# we will do Visual Normality Checks using Histogram plot for All Features need to be Normalized
for item in ['Item_Visibility' , 'Item_Weight' ,'Item_MRP' ,'Outlet_Establishment_Year']:
    plt.hist(Dataset[item])
    plt.show()    
#Because of Incurratance of usinge visualization than statistical methods we will Apply Statistical Normality Tests
for item in ['Item_Visibility']:
    Statis , P_value = normaltest(Dataset[item])
    print('Statistics=%.3f, p=%.3f' % (Statis, P_value))    
Alpha =0.5 # Is a Threshhold Value To interpret The Test Results 
if P_value>Alpha:
    print('Sample looks Gaussian (fail to reject H0)')
elif P_value<Alpha:
    print('Sample does not look Gaussian (reject H0)')
else:
    print("UNKNOWNEN")

#--------------Feature Scaling using Normalization Technique
norm=MinMaxScaler().fit(Dataset[['Item_Visibility']])
Dataset[['Item_Visibility' ]]=norm.transform(Dataset[['Item_Visibility' ]])
#---------------------Feature Scaling using standardization Technique
'''#-------data coping To apply another scaling 
Dataset2=Dataset.copy()
standard=StandardScaler().fit(traing_set2[['Item_Visibility' , 'Item_Weight' ,'Item_MRP' ,'Outlet_Establishment_Year']])
traing_set2[['Item_Visibility' , 'Item_Weight' ,'Item_MRP' ,'Outlet_Establishment_Year']]=standard.transform(traing_set2[['Item_Visibility' , 'Item_Weight' ,'Item_MRP' ,'Outlet_Establishment_Year']])
testing_set2[['Item_Visibility' , 'Item_Weight' ,'Item_MRP' ,'Outlet_Establishment_Year']]=standard.transform(testing_set2[['Item_Visibility' , 'Item_Weight' ,'Item_MRP' ,'Outlet_Establishment_Year']])'''
#dealing with null values
#------------------ filling missing values in numerical fields using imputer 
imputing_object=SimpleImputer(missing_values=np.nan , strategy='mean').fit(Dataset.iloc[: ,1:2])
Dataset['Item_Weight']=imputing_object.transform(Dataset.iloc[: ,1:2])
#most frequent categoriacl imputation to deal with missing values in Categorical features
#1. Function to replace NAN values with mode value
print(Dataset['Outlet_Size'].unique())
def impute_nan_most_frequent_category(data ,colname ):
    most_freq_category=data[colname].mode()[0]
    data[colname].fillna(most_freq_category , inplace=True)
#2. Call function to impute most occured category
impute_nan_most_frequent_category(Dataset , 'Outlet_Size')
#3. Drop actual columns
#Correlation Between features After filling Null values
sns.heatmap(Dataset[['Item_Fat_Content' ,'Item_Type' ,'Outlet_Establishment_Year' ,'Outlet_Type' ,'Outlet_Location_Type' ,'Item_MRP'  ,'Item_Outlet_Sales']].corr() , annot=True)
plt.show()
#Binning
Dataset['Item_MRP']=pd.qcut(Dataset['Item_MRP'] ,q=4)
Dataset['Item_MRP']=LabelEncoder().fit_transform(Dataset['Item_MRP'])
Dataset['Outlet_Establishment_Year']=pd.cut(Dataset['Outlet_Establishment_Year'] , bins=5)
Dataset['Outlet_Establishment_Year']=LabelEncoder().fit_transform(Dataset['Outlet_Establishment_Year'])
Dataset['Item_Weight']=pd.qcut(Dataset['Item_Weight'] , q=4)
Dataset['Item_Weight']=LabelEncoder().fit_transform(Dataset['Item_Weight'])
#Visualize  some Dataset Features after Binning
sns.factorplot(x='Item_Weight' , y='Item_Outlet_Sales' , data=Dataset , kind='bar' , size=4)
plt.show()
sns.factorplot(x='Item_MRP' , y='Item_Outlet_Sales' , data=Dataset , kind='bar' , size=4)
plt.show()
#----------model Building
# Deleting unused columns
Dataset.drop('Item_Identifier' , inplace=True , axis=1)
Dataset.drop('Outlet_Identifier' , inplace=True , axis=1)
from sklearn.model_selection import train_test_split
#I Have to splitting DatastManiualy
Training_Data =Dataset.iloc[0:8522 ,:]
Test_Data=Dataset.iloc[8523: ,:]
X_Training=Training_Data.iloc[: ,0:9]
Y_Training=Training_Data.iloc[: ,9:]
X_Testing=Test_Data.iloc[: ,0:9]
Y_Testing=Test_Data.iloc[: ,9:]
#XTraining_Data , XTesting_Data , YTraining_Data ,YTesting_Data = train_test_split(X,Y ,test_size=0.2 ,random_state=0 )
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
KNN=KNeighborsRegressor(n_neighbors=7)
#Fitting the model with KNeighborsRegressor
KNN.fit(X_Training ,Y_Training)
#predict
KNN_Y_pred=KNN.predict(X_Testing)
#------------------
#Fitting the model with DecisionTreeRegressor

DT=DecisionTreeRegressor(max_depth=10,random_state=27)
DT.fit(X_Training ,Y_Training)
DT_Y_Predict=DT.predict(X_Testing)

# Results
print('Decision Tree')
print(DT_Y_Predict)
print('kNN')
print(KNN_Y_pred)
# Creating the csv file for submitting the solu
Predictions=pd.DataFrame({'Actule_Prediction':DT_Y_Predict})
Predictions.to_csv('Solution.csv' , index=False)
















    
    



