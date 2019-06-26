# Day Trading Cheat Sheet

This project is a machine learning application that uses historical stock data in order to predict the closing price of a stock on a given day. It is meant to be used by day traders when the market opens in order to make predictions about the price of a stock at the end of the day so that traders can make the most efficient and profitable decisions. The project requires Python 3 and the packages specified in the requirements text, which can be installed through pip.

## Data Preprocessing

The data for the model comes from Alpha Vantage through their API (limits to 5 API calls per minute). The data includes the S&P 500 stock index historical data as well as the historical data for a given company. The model currently uses unadjusted opening and closing prices that don't account for dividends and stock splits. The API returns a CSV that needs to be reindexed to make the date its own column. Also, Alpha Vantage only provides S&P 500 data from the year 2000 and onward, so I used Yahoo Finance Data to fill in the rest. The preprocessor reads all of this data and removes the unnecessary features first. It then puts all of the data into a single, organized DataFrame and converts the data types so that they are all consistent.

## Model Structure

The model currently uses an input layer, 3 fully connected hidden layers, and then an output layer. The output layer uses a linear activation function while the rest of the layers use a relu function. The input layer has 128 nodes while the hidden layers each have 256 layers. While this slows down the model a bit because there are simply so many parameters to train, it should also increase the accuracy of the model predictions.

Still in development...

## Application

At the moment, it can only be run through the command line. Front-end in development...
