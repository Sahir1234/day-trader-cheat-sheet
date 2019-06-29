import sys

from alpha_vantage.timeseries import TimeSeries
from modules.preprocessor import Preprocessor
from modules.model import Model

import matplotlib.pyplot as plt

ts = TimeSeries(key='G55XFTEJRRNFT53F', output_format='pandas')
company = str(input('Enter Company Stock Symbol: '))
data, __ = ts.get_daily(symbol=company, outputsize='full')
index_data, __ = ts.get_daily(symbol='INX', outputsize='full')
print('DATA RETRIEVED!')

prep = Preprocessor(data, index_data)
prep.append_SP_data()
prep.remove_nulls()
prep.split_date_features()
prep.convert_data_types()

response = str(input('Plot Stock Data? (y/n) : '))

if(response == 'y'):
    prep.plot_data(company)

X_pred = prep.remove_current_data()
x, y = prep.get_data()

#########################################
#
# END OF PREPROCESSING, BEGINNING REGRESSION
#
##########################################

count = int(input('How Many Models Would You Like to Train? '))

prediction_history=[]

for i in range(0,count):

    print('Constructing Model...')
    reg = Model(len(x.columns))

    loss_history, prediction_history = reg.train(x, y, X_pred)

    response = str(input('Plot Loss Data? (y/n) : '))

    if(response == 'y'):
        reg.plot_loss(loss_history)
        reg.plot_prediction_history(prediction_history)

    Y_pred = reg.predict(X_pred)
    prediction_history.append(Y_pred)

true_prediction = sum(prediction_history)/len(prediction_history)

print(prediction_history)
print('')
print('********************')
print('TRUE PREDICTION: ' + str(true_prediction))
print('********************')
print('')

print('Recommended Course of Action: ')
if(true_prediction >= X_pred['Open'][0]):
    print('BUY')
else:
    print('SELL')
print('')
