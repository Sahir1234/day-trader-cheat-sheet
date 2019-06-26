
#
# IMPORTANT: This preprocessor (and the model as a whole) is meant to be used
# on historical stock price datasets for individual companies found at Yahoo
# Finance. Currently, this preprocessor only works for complete datasets with
# no missing data and company stock data that begins after January 3, 1950.
# If you have questions, contact 'sahir.mody@gmail.com'.
#

# import necessary packages to preprocess the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# This class serves as a preprocessor that analyzes stock data in CSV files
# and prepares/formats it so that it can be fed directly into the model for
# training, testing, prediction, etc.
class Preprocessor (object):
 
    # Basic constructor that converts the data in a specified CSV file
    # into a Pandas dataframe.
    def __init__ (self, data, index_data):
        
        self.df = data
        
        # Remove the unnecessary data and restructure the date index of the data
        self.df.drop(columns = ['2. high', '3. low', '5. volume'], inplace=True)
        self.df = self.df.reset_index()
        self.df.rename(columns={'1. open':'Open', '4. close':'Close', 'date': 'Date'}, inplace=True)
    
        # Remove the unnecessary data and restructure the date index of the S&P 500 data
        index_data = index_data.reset_index()
        index_data.rename(columns={'1. open':'Open','date': 'Date'}, inplace=True)
        index_data.drop(columns = ['2. high', '3. low', '4. close', '5. volume'], inplace=True)
        
        # The S&P 500 data from Alpha Vantage uses data from after 1999, so we need to fill in the previous
        # 50 years worth of data with CSV data from Yahoo Finance
        self.market_data = pd.read_csv('./modules/SP500.csv')
        self.market_data = self.market_data.append(index_data, ignore_index=True)
        self.market_data.rename(columns={'Open': 'Index'}, inplace=True)
        
        # Print statements for visual verification of data processing
        print('Unnecessary Features Removed!')
        print(self.df.columns)
        print(self.market_data.columns)
    
    
    # Finds the index in the market data dataframe where the dates match with
    # the dates in the company data
    def find_start_date_pos(self):
        
        # Searches for the point in the S&P 500 data where the
        for i in range(0, len(self.market_data['Date'])):
            if(self.df['Date'][0] == self.market_data['Date'][i]):
                print('Data Aligns! Appending Index Data...')
                return i

    # Adds the index data from market_data with corresponding dates
    # to the dataframe with the company-specific data
    def append_SP_data(self):

        position = self.find_start_date_pos()
        
        # Slice the S&P 500 data with the relevant dates into a new Data Series
        index_values = self.market_data['Index'][position:]
        print('Relevant Index Values Identified...')
        print(index_values)
        
        # Reindex the S&P 500 data so that it can be added easily as a new colummn
        index_values.index = range(len(index_values))

        self.df['Index'] = index_values
        print('All Data Has Been Added')
    
    
    # Checks for and removes missing elements in the dataset just in case
    def remove_nulls(self):
        
        print('Removing Null Data...')
        
        # Find the indices in each column where there are null values and removes those
        # entire rows from the dataset
        for column in ['Open', 'Close']:
            for index in self.df[column].index[self.df[column].apply(np.isnan)].tolist():
                self.df.drop(labels=index,axis=0,inplace=True)
                print('NULL ELEMENT REMOVED!')

        print('All Null Data Removed Successfully!')


    # Separates the date data into three separate features: year, month, and day
    def split_date_features(self):

        # new data frame with split value columns
        times = self.df['Date'].str.split('-', n = 2, expand = True)
        print('Formatting Time Features...')
        
        self.df['Year'] = times[0]
        self.df['Month'] = times[1]
        self.df['Day'] = times[2]

        # Remove the now unnecessary date data
        self.df.drop(columns='Date', inplace=True)
        
        # Reorganize the data so that the features are on the left and the
        # output is on the rightmost column of the DataFrame
        self.df = self.df.reindex(columns=['Year','Month','Day','Open','Index','Close'])
        print('Time Data Formatted! Converting Data Types...')
        
        
    # Convert data types of the date data so that all of the inputs
    # have the same data type for model training
    def convert_data_types(self):
        
        self.df['Year'] = self.df['Year'].astype('float64')
        self.df['Month'] = self.df['Month'].astype('float64')
        self.df['Day'] = self.df['Day'].astype('float64')
        
        # Sanity check to make sure that all of the type conversions were successful
        for str in self.df.columns:
            print(type(self.df[str][0]))
            if not isinstance(self.df[str][0], float):
                raise RuntimeError('Unknown Error when converting data types')

        print('')
        print('**** PREPROCESSING COMPLETE ****')
        print('')
    
        print(self.df)

    
    # Getter for the data in the correct format so that it can
    # be fed directly into the model
    def get_data(self):
        
        x  = self.df.drop(columns='Close')
        y = self.df['Close']
    
        return x, y

    # Generates a plot of the historical stock price data over time so
    # that the data can be visualized more easily
    def plot_data(self, company_name):

        print('')
        print('CREATING DATA PLOT...')
        print('')

        prices = self.df['Close']

        plt.plot(range(0, len(prices)), prices)
        plt.xlabel('Business Days Since Stock Opened')
        plt.ylabel('Price')
        plt.title(company_name +' Stock Prices Over Time')
        plt.show()
