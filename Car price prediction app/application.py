
from flask import Flask,render_template
import pandas as pd

app =Flask(__name__)
car=pd.read_csv('refine_car.csv')

@app.route('/')

def index():
    companies=sorted(car['company'].unique())
    car_models=sorted(car['name'].unique())
    year= sorted(car['year'].unique(),reverse=True)
    fuel_type=car['fuel_type'].unique()
    
    return render_template("index.html",companies=companies,car_models=car_models,year=year,fuel_types=fuel_type)


if __name__=="__main__":
    app.run(debug=True)

app.route('/predict', methods=['POST'])
def predict():
    return

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
