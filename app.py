
from flask import Flask,render_template,request
from numpy import int32
import pandas as pd
import pickle

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


app =Flask(__name__)

car=pd.read_csv('notebooks/data/refine_car.csv')


@app.route('/')
def index():
    companies=sorted(car['company'].unique())
    car_models=sorted(car['name'].unique())
    fuel_type=car['fuel_type'].unique()
    kms_driven=car['kms_driven'].unique()
    
    return render_template("index.html",companies=companies,car_models=car_models,fuel_types=fuel_type,kms_driven=kms_driven)



@app.route("/predict", methods=['POST','GET'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(name=request.form.get('car_models'),
                          company=request.form.get('companies'),
                          year = request.form.get('year'),
                          kms_driven=request.form.get('kms_driven'),
                          fuel_type=request.form.get('fuel_type')
                )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print(results)
        return render_template('index.html', price=results[0])

if __name__=="__main__":
    app.run(debug=True)
