# Day Trading Cheat Sheet

This project is a machine learning application that uses historical stock data in order to predict the closing price of a stock on a given day. It is meant to be used by day traders when the market opens in order to make predictions about the price of a stock at the end of the day so that traders can make the most efficient and profitable decisions. The project requires Python 3 and the packages specified in requirements.txt, which can be installed through pip3.

## Back-End

The data for the model comes from the Alpha Vantage API. The Alpha Vantage API has a limit of 5 API calls per minute, so the model can only be run twice per minute (trying to do more will result in a KeyError). The data includes the S&P 500 stock index historical data as well as the historical data for a given company. The preprocessor reads all of this data and removes the unnecessary features first. It then puts all of the data into a single, organized DataFrame and converts the data types so that they are all consistent. The model takes this data and feeds it through an input layer, several fully connected hidden layers, and then an output layer. The amount of neurons in each layer was picked after a bit of trial and error and could still be significantly improved. The user also has the option of training multiple models in order to get a more accurate prediction.

## Front-End

The application has a front-end written in HTML and JavaScript that connects to the back-end through Flask. The user can submit the stock symbol of the company they want to predict stock price for, which will start up the back-end processes that invoke the Alpha Vantage API. The user is then redirected to page.html, where they are prompted for the amount of models they would like to train, after which they begin training. The webpage then displays graphs of the loss and predictions made by each model after each epoch of training. In order to run the app again, the user must go back to the homepage and start from the beginning. One issue faced when developing the front-end was transferring Pandas DataFrames between Flask routes while maintaining the datatypes and the structure of the data. To do this, I had to save the DataFrames as CSVs in the local system in one route and then read them back and delete them in a different route. If the app runs start-to-finish, this is not a problem. It becomes an issue if a user closes the app before they select how many models they want to train.


## Deployment

Because I don't have a web server to deploy the application on, I have only been running it locally. Tthe application must be run from the command line by running the app.py file. This will then start up the Flask server, usually at port 5000 on the local host unless that port is already in use. In a browser, you can then open the local host at that specific port to see the front-end. Even though the app runs locally, a strong internet connection is still required because the API calls need a connection to work. When closing the app, you must close the browser and the terminal in order for the app to completely close.
