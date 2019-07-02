
import json
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
    
    try:
        ts = TimeSeries(key='G55XFTEJRRNFT53F', output_format='pandas')
        data, __ = ts.get_daily(symbol=company, outputsize='full')
        index_data, __ = ts.get_daily(symbol='INX', outputsize='full')
    except KeyError:
        return redirect('/')

    prep = Preprocessor(data, index_data)
    prep.append_SP_data()
    prep.remove_nulls()
    prep.split_date_features()
    prep.convert_data_types()

    global X_pred
    global x
    global y
    X_pred = prep.remove_current_data()
    x, y = prep.get_data()
    
    return render_template('load.html', company=company)

@app.route('/<company>', methods=['POST'])
def analyze(company):
    try:
        count = int(request.form['count'])
        if(count <= 0):
            count = 1
    except ValueError:
        count = 1

    predictions=[]
    errors=[]

    global prediction_json
    prediction_json = []

    for i in range(0,count):
    
        reg = Model(len(x.columns))
    
        prediction_history, rmse = reg.train(x, y, X_pred)
    
        prediction_history.insert(0, 'Model '+ str(i+1))
        prediction_json.append(prediction_history)
        
        Y_pred = reg.predict(X_pred)
        predictions.append(Y_pred)
        errors.append(rmse)

    global true_prediction
    global true_error
    true_prediction = sum(predictions)/len(predictions)
    true_error = sum(errors)/len(errors)

    print('')
    print('********************')
    print('TRUE PREDICTION: ' + str(true_prediction))
    print('********************')
    print('')
    print('True Error: ' + str(true_error))
    print('')

#    global prediction_json
#   prediction_json = json.dumps(prediction_data)

    return redirect('/' + company + '/' + str(count) + '/' 'results')

@app.route('/<company>/<count>/results')
def show_results(company, count):
    return render_template('results.html', company = company, count = count, true_prediction = true_prediction, true_error = true_error, prediction_json = prediction_json)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
