#inmporting all the essential libraries
from flask import Flask
from flask_restful import Resource,Api
from flask_cors import CORS
import numpy as np      # the numpy library is for numpy arrays
import pandas as pd     # pandas is for preprossing the CSV files and converting into dataframes
import matplotlib.pyplot as plt  # matplotlib is for visualising the data
import seaborn as sns

import os
print(os.listdir())

import warnings
warnings.filterwarnings('ignore')
import json




app = Flask(__name__)
api=Api(app)



dataset = pd.read_csv("heart.csv") # importing the dataset heart.csv

from sklearn.model_selection import train_test_split

predictors = dataset.drop("target",axis=1)
target = dataset["target"]

X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,Y_train)

Y_pred_lr = lr.predict(X_test)

score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)

print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")






class Predict(Resource):
    def get(self):
        return "hello!!!"

class PredictOutput(Resource):
    def get(self,age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal):
        temp={'age':[age],'sex':[sex],'cp':[cp],'trestbps':[trestbps],'chol':[chol],'fbs':[fbs],'restecg':[restecg],'thalach':[thalach],'exang':[exang],'oldpeak':[oldpeak],'slope':[slope],'ca':[ca],'thal':[thal]}
        print(temp)
        ans = pd.DataFrame(data=temp)
        clasify=lr.predict(ans)
        print(clasify)
        return str(clasify)

api.add_resource(Predict,'/predict')
api.add_resource(PredictOutput,'/predictoutput/<int:age>,<int:sex>,<int:cp>,<int:trestbps>,<int:chol>,<int:fbs>,<int:restecg>,<int:thalach>,<int:exang>,<int:oldpeak>,<int:slope>,<int:ca>,<int:thal>')



if __name__ == '__main__':
    app.run(debug=True)































"""print("lets diagonos the disease")
print("enter the values of :")
temp={'age':[70],'sex':[1],'cp':[0],'trestbps':[145],'chol':[174],'fbs':[0],'restecg':[1],'thalach':[125],'exang':[1],'oldpeak':[2.6],'slope':[0],'ca':[0],'thal':[3]}
#print(temp)
error=False
for i in ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'] :
    
    if(i=='age'):
        print('enter your '+i+':')
        val=int(input())
        temp[i]=[val]
        continue
    if(i=='sex'):
        print("enter your sex 'male' or 'female' without caps")
        val=input()
        if(val=='male'):
            temp[i]=[1]
            continue
        if(val=='female'):
            temp[i]=[0]
            continue
        error=True
        break
    if(i=='cp'):
        print("chest pain types 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic please enter the value corresponding")
        val=int(input())
        if(val<=4 or val>=1):
            temp[i]=[val]
            continue
        error=True
        break
    if(i=='trestbps'):
        print("enter the resting blood pressure")
        val=int(input())
        temp[i]=[val]
        continue
    if(i=='chol'):
        print("enter the cholostrol levels")
        val=int(input())
        temp[i]=[val]
        continue
    if(i=='fbs'):
        print("enter the fasting suger levels if greater then 120 mg/dl [1 or 0]")
        val=int(input())
        temp[i]=[val]
        continue
    if(i=='restecg'):
        print("enter the electrocardiographic results (values 0,1,2) ECG")
        val=int(input())
        if(val<0 or val>2):
            error=True
            print("wrong values entered please enter the values between 0 to 2 inclusively")
            break
        temp[i]=[val]
        continue
    if(i=='thalach'):
        print("enter the maximum heart rate achieved")
        val=int(input())
        temp[i]=[val]  
        continue     
    if(i=='exang'):
        print("enter the exercise induced angina 0 or 1")
        val=int(input())
        temp[i]=[val]
        continue
    if(i=='oldpeak'):
        print("enter the  ST depression induced by exercise relative to rest")    
        val=float(input())
        temp[i]=[val]
        continue
    if(i=='slope'):
        print("enter the the slope of the peak exercise ST segment")
        val=int(input())
        temp[i]=[val]
        continue
    if(i=='ca'):
        print("enter the number of major vessels (0-3) colored by flourosopy")
        val=int(input())
        temp[i]=[val]
    if(i=='thal'):
        print("enter the values 1 if normal, 2 if fixed defect, 3 if reversable defect")    
        val=int(input())
        temp[i]=[val]
        continue    
    
print()
print()
print()    
print(temp)
print()
print()
ans = pd.DataFrame(data=temp)
clasify=lr.predict(ans)
if(clasify==1):
    print("YOUR HAVING HEART DISEASE")
else:
    print("YOUR HEART IS NORMAL")

print("THANKS FOR USING OUR SERVICES.")  """      

