
import pandas as pd
import numpy as np

class Preprocessor (object):
 
 
    def __init__ (self, data, index_data):
        
        """ Basic constructor that removes unnecessary data and restructures the DataFrames given in
            the parameters so that they can go through the rest of the preprocessing steps. The resulting DataFrame
            will only have Date, Opening Price, and S&P 500 IndexPrice  data, which is all the model will learn from.
            
            Parameters:
                data (pd.DataFrame): should be a formatted DataFrame that comes from an
                    Alpha Vantage API call containing data on a specific company
            
                index_data (pd.DataFrame): historical data of the S&P 500 Stock Index from after 1999 (meant to represent the state of the overall market in the model's algorithm, which is why it is stored in market_data)
        """
        
        self.df = data
        print('DATA RETRIEVED!')
        
        
        # Remove the unnecessary data and restructure the date index of the data
        self.df.drop(columns = ['2. high', '3. low', '5. volume'], inplace=True)
        self.df = self.df.reset_index()
        self.df.rename(columns={'1. open':'Open', '4. close':'Close', 'date': 'Date'}, inplace=True)
        
        index_data = index_data.reset_index()
        index_data.rename(columns={'1. open':'Open','date': 'Date'}, inplace=True)
        index_data.drop(columns = ['2. high', '3. low', '4. close', '5. volume'], inplace=True)
        
        
        # The S&P 500 data from Alpha Vantage uses data from after 1999, so we need to fill in the previous
        # 50 years worth of data with CSV data from Yahoo Finance
        self.market_data = pd.read_csv('./modules/SP500.csv')
        self.market_data = self.market_data.append(index_data, ignore_index=True)
        self.market_data.rename(columns={'Open': 'Index'}, inplace=True)
        
        
        print('Unnecessary Features Removed!')
        print(self.df.columns)
        print(self.market_data.columns)
    
    

    def find_start_date_pos(self):
        
        """ Finds the position in the market data DataFrame where the dates match with the dates in the company DataFrame
            
            Returns:
                i (int): index in the market data where the dates begin to overlap with the company data dates
        """
        
        # Searches for the point in the S&P 500 data where the
        for i in range(0, len(self.market_data['Date'])):
            if(self.df['Date'][0] == self.market_data['Date'][i]):
                print('Data Aligns! Appending Index Data...')
                return i



    def append_SP_data(self):
        
        """ Since the S&P 500 data goes all the way back to the 1950s and most company data does not,
            we need to isolate the S&P 500 data whose dates overlap with the company data dates. That data
            is then added to the primary DataFrame with the company data
        """

        position = self.find_start_date_pos()
        
        # Slice the S&P 500 data with the relevant dates into a new Data Series
        index_values = self.market_data['Index'][position:]
        print('Relevant Index Values Identified...')
        print(index_values)
        
        index_values.index = range(len(index_values))
        self.df['Index'] = index_values
        print('All Data Has Been Added')
    
    
    
    def remove_nulls(self):
        
        """ Checks for and removes missing elements in the dataset just in case. Instead of filling
            in missing data, rows with missing data are simply removed from the DataFrame.
        """
        
        print('Removing Null Data...')
        
        for column in ['Open', 'Close']:
            for index in self.df[column].index[self.df[column].apply(np.isnan)].tolist():
                self.df.drop(labels=index,axis=0,inplace=True)
                print('NULL ELEMENT REMOVED!')

        print('All Null Data Removed Successfully!')


    
    def split_date_features(self):
        
        """ Separates the date data that is a single string into three separate
            strings for the Year, Month, and Day,each of which are there own feature
            when training the model
        """

        times = self.df['Date'].str.split('-', n = 2, expand = True)
        print('Formatting Time Features...')
        
        self.df['Year'] = times[0]
        self.df['Month'] = times[1]
        self.df['Day'] = times[2]

        self.df.drop(columns='Date', inplace=True)
        
        # Reorganize the data so that the 5 features are on the left and the 1
        # output is on the rightmost column of the DataFrame
        self.df = self.df.reindex(columns=['Year','Month','Day','Open','Index','Close'])
        print('Time Data Formatted! Converting Data Types...')
    
    
    
    def convert_data_types(self):
        
        """ Converts the date features into floats because that is the data type that
            the model is designed to take in as input.
            
        """
        
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
    
    
    
    def remove_current_data(self):
        
        """ Removes the data for the current day or the last day that the market was open from
            the DataFrame so that it can be used by the model for prediction of future stock prices
            
            Returns:
                current_data(pd.DataFrame): a single row DataFrame that only has input data for the models
        """
        
        current_data = self.df.tail(1)
        current_data = current_data.reset_index()
        current_data.drop(columns=['index', 'Close'], inplace=True)
        
        self.df.drop(self.df.tail(1).index,inplace=True)
        
        print(current_data)
        print('Current Data Successfully Removed!')
     
        return current_data
    
    
    
    def get_data(self):
        
        """ Getter for the data in the correct format so that it can be fed directly into the model
            
            Returns:
                x (pd.DataFrame): the inputs that our model should make predictions on

                y (pd.DataFrame): the outputs that can be used to train and test the model
        """
        
        x  = self.df.drop(columns='Close')
        y = self.df.drop(columns=['Year','Month','Day','Open','Index'])
    
        return x, y
