import sys
from flask import Flask,render_template,url_for,request, redirect
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
    print('DATA RETRIEVED!')
    
    prep = Preprocessor(data, index_data)
    prep.append_SP_data()
    prep.remove_nulls()
    prep.split_date_features()
    prep.convert_data_types()
    
    X_pred = prep.remove_current_data()
    x, y = prep.get_data()
    
    return render_template('page.html', company=company)

@app.route('/<company>', methods=['POST'])
def analyze(company):
    count = request.form['count']
    return count

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

#########################################
#
# END OF PREPROCESSING, BEGINNING REGRESSION
#
##########################################

count = int(input('How Many Models Would You Like to Train? '))

predictions=[]
errors=[]

for i in range(0,count):

    print('Constructing Model...')
    reg = Model(len(x.columns))

    loss_history, prediction_history, rmse = reg.train(x, y, X_pred)

    reg.plot_loss(loss_history)
    reg.plot_predictions(prediction_history)

    Y_pred = reg.predict(X_pred)
    predictions.append(Y_pred)
    errors.append(rmse)

true_prediction = sum(predictions)/len(predictions)
true_error = sum(errors)/len(errors)

print(prediction_history)
print('')
print('********************')
print('TRUE PREDICTION: ' + str(true_prediction))
print('********************')
print('')
print('True Error: ' + str(true_error))
