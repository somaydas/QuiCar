from flask import Flask,render_template,request,redirect
import pickle
import pandas as pd
import numpy as np

#to deploy on heroku flask-scors

app=Flask(__name__) #Creating a object of flask

model=pickle.load(open("LinearRegressionModel.pkl",'rb'))

car=pd.read_csv('Cleaned_Car.csv') #Reading the csv

@app.route('/') #entry point of our application

def index():#this function will only be called when someone hits on that above route
    companies=sorted(car['company'].unique())
    car_model=sorted(car['name'].unique())
    year=sorted(car['year'].unique())
    fuel_type=sorted(car['fuel_type'].unique())
    companies.insert(0,"Select Company")

    return render_template('index.html',companies=companies,car_model=car_model,years=year,fuel_type=fuel_type) #Sending the data according the attributes.

@app.route('/predict',methods=['POST'])
def predict():
    company=request.form.get('company')
    car_model=request.form.get('car_model')
    year=int(request.form.get('year'))
    fuel_type=request.form.get('fuel_type')
    kms_driven=int(request.form.get('kilo_driven'))

    #print(company,car_model,year,fuel_type,kms_driven);

    prediction=model.predict(pd.DataFrame([[car_model,company , year, kms_driven, fuel_type]],
                              columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
    #print(prediction)
    return str(np.round(prediction[0],2))


if __name__=="__main__":
    app.run(debug=True)