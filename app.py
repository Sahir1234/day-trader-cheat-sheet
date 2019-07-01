
import os
import pandas as pd
from flask import Flask, render_template, url_for, request, redirect
from alpha_vantage.timeseries import TimeSeries
from modules.preprocessor import Preprocessor
from modules.model import Model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/', methods=['POST'])
def get_company():
    company = request.form['company']
    return redirect('/' + company)

@app.route('/<company>')
def get_data(company):
    
    ts = TimeSeries(key='G55XFTEJRRNFT53F', output_format='pandas')
    data, __ = ts.get_daily(symbol=company, outputsize='full')
    index_data, __ = ts.get_daily(symbol='INX', outputsize='full')

    prep = Preprocessor(data, index_data)
    prep.append_SP_data()
    prep.remove_nulls()
    prep.split_date_features()
    prep.convert_data_types()

    X_pred = prep.remove_current_data()
    x, y = prep.get_data()
    
    X_pred.to_csv('X_pred.csv', index=False)
    x.to_csv('x.csv', index= False)
    y.to_csv('y.csv', index=False)
    
    return render_template('page.html', company=company)

@app.route('/<company>', methods=['POST'])
def analyze(company):
    try:
        count = int(request.form['count'])
        if(count <= 0):
            count = 1
    except ValueError:
        count = 1

    X_pred = pd.read_csv('X_pred.csv')
    x = pd.read_csv('x.csv')
    y = pd.read_csv('y.csv')
    os.remove('X_pred.csv')
    os.remove('x.csv')
    os.remove('y.csv')

    predictions=[]
    errors=[]

    for i in range(0,count):
    
        reg = Model(len(x.columns))
    
        loss_history, prediction_history, rmse = reg.train(x, y, X_pred)
        
        Y_pred = reg.predict(X_pred)
        predictions.append(Y_pred)
        errors.append(rmse)

    true_prediction = sum(predictions)/len(predictions)
    true_error = sum(errors)/len(errors)

    print('')
    print('********************')
    print('TRUE PREDICTION: ' + str(true_prediction))
    print('********************')
    print('')
    print('True Error: ' + str(true_error))

    return str(true_prediction)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
