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

print('Constructing Model...')
reg = Model(len(x.columns))

loss_history = reg.train_test(x, y)

response = str(input('Plot Loss Data? (y/n) : '))

if(response == 'y'):
    reg.plot_loss(loss_history)

Y_pred = reg.predict(X_pred)

print('')
print('Recommended Course of Action: ')
if(Y_pred >= X_pred['Open'][0]):
    print('BUY')
else:
    print('SELL')
