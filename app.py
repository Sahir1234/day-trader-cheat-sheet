
import json
import pandas as pd

from flask import Flask, render_template, url_for, request, redirect

from alpha_vantage.timeseries import TimeSeries

from modules.preprocessor import Preprocessor
from modules.model import Model


app = Flask(__name__)


# Start page for the app where users can enter the stock symbol of
# the company that they would like to analyze
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def get_company():
    
    # When the user submits which company's stock they would like to analyze,
    # this method reads their submission and directs them to the next page
    company = request.form['company']
    
    return redirect('/' + company)


@app.route('/<company>')
def get_data(company):
    
    """ The data is retrieved from the API based on teh stock symbol supplied by the user
        and run through the Preprocessor. If it all goes successfully, the user is directed
        to the load webpage. If not, they are returned to the home page.
        
        Parameters:
            company(str): stock symbol of the stock that the model analyzed
    """
    
    # if the user selected a stock that doesn't exist, they will be redirected back to
    # the start page and prompted to enter a stock symbol again
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

    # Global variables required so that the data can be used in multiple routes
    global X_pred
    global x
    global y
    X_pred = prep.remove_current_data()
    x, y = prep.get_data()
    
    return render_template('load.html', company=company)



@app.route('/<company>', methods=['POST'])
def analyze(company):
    
    """ This route responds when the user submits how many models they would like to train.
        It trains and predicts with a model as many times as the user specified and then
        redirects to the final results page.
        
        Parameters:
            company(str): stock symbol of the stock that the model will analyze
    """
    
    # Reads the user's submission of how many models they would like to train and sets
    # it to 1 if the user entered something besides a positive integer
    try:
        count = int(request.form['count'])
        if(count <= 0):
            count = 1
    except ValueError:
        count = 1

    # Stores the final prediction and error of each model after it has
    # completed all of the epochs of traing
    predictions=[]
    errors=[]

    # Stores the predictions each model makes after each epoch
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


    # Average the predictions to get the final or "true" prediction/error
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

    return redirect('/' + company + '/' + str(count) + '/' 'results')



@app.route('/<company>/<count>/results')
def show_results(company, count):
    
    """ Final Page of the app where the model prediction history, final predictions,
        and overall error is shown
        
        Parameters:
            company(str): stock symbol of the stock that the model analyzed
            count(int): the amount of models that the user asked to be trained
    """
    
    return render_template('results.html', company = company, count = count, true_prediction = true_prediction, true_error = true_error, prediction_json = prediction_json)



if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
