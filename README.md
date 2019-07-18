
# Day Trading Cheat Sheet


## Introduction

This project is a machine learning application that uses historical stock data in order to predict the closing price of a stock on a given day. It is meant to be used by day traders when the market opens in order to make predictions about the price of a stock at the end of the day so that traders can make the most efficient and profitable decisions. The purpose of this project was to experiment with machine learning and see if a model could be made using numerical data that actually made accurate predictions. Obviously, the stock market is a lot more than just numbers and is extremely volatile, so this is not the optimal method for analyzing stocks. However, when the model analyzes a company with a lot of historical data, it performs surprisingly well. 

This project requires Python 3 and the packages specified in requirements.txt, which can be installed through pip3.


## Back-End

The back-end consists of a Flask application, a preprocessor class, and a model class, all written in Python. The model looks at five features: Year, Month, Day, Opening Price of the Stock, and the Opening Value of the S&P 500 Index (adding more comprehensive metrics of the stock market is one place that this model could be improved). The model is supposed to use this information to predict the Close price of the stock on the day it has been given in its input. The data for the model comes from the Alpha Vantage API. The Alpha Vantage API has a limit of 5 API calls per minute, so the model can only be run twice per minute (trying to do more will result in a KeyError). The preprocessor reads all of this data and removes the unnecessary features first. It then puts all of the data into a single, organized DataFrame, splits the date into three separate features, and converts all of teh data to floating point values. This data is then given to the model for training. The model takes this data and feeds it through an input layer, several fully connected hidden layers, and finally an output layer. The amount of neurons in each layer was picked after a bit of trial and error and could still be significantly improved.


## Front-End/Deployment

The application has a front-end written in HTML and JavaScript that connects to the back-end through Flask. The user can submit the stock symbol of the company they want to predict stock price for, which will start up the back-end processes that invoke the Alpha Vantage API. The user is then redirected to the loading page, where they are prompted for the amount of models they would like to train (at the moment, the app can only handle three models), after which they begin training. The final results webpage then displays graphs predictions made by each model after each epoch of training to show how it improves. At the moment, this requires users to be patient and wait for all the models to finish training before loading the next page. this could be imprioved in future versions by incorporating multi-threading and some JS animations to make the app a little more interesting while the models train. In order to run the app again, the user must go back to the homepage and start from the beginning. Because I don't have a web server to deploy the application on, I have only been running it locally. The Flask server must be started by running "python3 app.py" in the project root directory. In a browser, you can then open the local host at the port specified when the server starts up to actually run the app.

## Future Dev

There are two main areas for future development. The first is finding a way to transfer the data used for training of the model between Flask routes without saving it to the local filesystem. At the moment, the data is too big to be stored as a session variable so I had to store it as CSV files on the local filesystem. Another area for development is to speed up model development time or have it return prediction data in batches instead of all at once so that the user doesn't have to wait so long for results. Contact me at sahir.mody@gmail.com if you wish to contribute to this project.
