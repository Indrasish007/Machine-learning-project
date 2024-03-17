
from flask import Flask,render_template,request
from numpy import int32
import pandas as pd
import pickle


app =Flask(__name__)
car=pd.read_csv('refine_car.csv')
model=pickle.load(open('LinearRegressionModel.pkl','rb'))
@app.route('/')

def index():
    companies=sorted(car['company'].unique())
    car_models=sorted(car['name'].unique())
    year= sorted(car['year'].unique(),reverse=True)
    fuel_type=car['fuel_type'].unique()
    kms_driven=car['kms_driven'].unique()
    
    return render_template("index.html",companies=companies,car_models=car_models,year=year,fuel_types=fuel_type,kms_driven=kms_driven)



@app.route("/predict", methods=['POST'])
def predict():
    companies= request.form.get('companies')
    car_models=request.form.get('car_models')
    year= int(request.form.get('year'))
    fuel_type=request.form.get('fuel_type')
    kms_driven=request.form.get('kms_driven')
    print(companies,car_models,year,fuel_type,kms_driven)
    prediction=model.predict(pd.DataFrame([[companies,car_models,year,fuel_type,kms_driven]],columns=['name','company','year','fuel_type','kms_driven']))
    print(prediction)
    return ""

if __name__=="__main__":
    app.run(debug=True)
# from flask import Flask
# import pandas as pd
# app = Flask(__name__)
# car=pd.read_csv('refine_car.csv')

# @app.route('/')
# def hello_world():
#     companies=sorted(car['company'].unique())
#     car_models=sorted(car['name'].unique())
#     year= sorted(car['year'].unique(),reverse=True)
#     fuel_type=car['fuel_type'].unique()
#     return str(len(car_models))

# if __name__=="__main__":
#      app.run(debug=True)
