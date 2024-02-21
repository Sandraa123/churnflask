from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data=pd.read_csv("cc.csv")
print(data.head(5))
data['class']=data['Churn'].apply(lambda x:1 if x== "Yes" else 0)

x=data[['tenure','MonthlyCharges']].copy()
y=data['class'].copy()



x_train, x_test, y_train, y_test=train_test_split(x,y , test_size= 0.2, random_state= 0)
clf=LogisticRegression(fit_intercept=True, max_iter=10000)
clf.fit(x_train,y_train)


from flask import Flask,render_template,request
churn = Flask(__name__)

@churn.route('/')
def home():
    return render_template('churn.html')


@churn.route('/prediction',methods=["GET","POST"])
def hom():
    tenure=request.form['tenure']
    MonthlyCharges=request.form['MonthlyCharges']
    ar=np.array([tenure,MonthlyCharges])
    ar=ar.astype(np.float64)
    predd=clf.predict([ar])

    if predd==1:
        result= "churned"
    else:
        result= "not churned"


    return render_template('prediction.html',predd=result)

if __name__ == '__main__':
    churn.run(debug=True)