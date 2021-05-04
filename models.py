from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, SGDRegressor
from keras import models, layers, optimizers
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
import pandas as pd

class Model:
    def __init__(self, n_features):
        self.features = n_features
        # maybe add something here -> input data?
    
    def format_data(self, data):
        print("this will format data")
        # randomize
        # split into x and y
        # scale x 
        # return 10-fold

    #def adjusted_r2(self, y_true, y_pred, n, p): not using this because it's too late... let's just use r^2
    # calculates adjusted r2
    # y_true and y_pred from model
    # n = number of observations in sample
    # p = number of independent variables
        #r2 = metrics.r2_score(y_true, y_pred)
        #adjusted = 1-(1-r2)*((n-1)/(n-p-1))
        #return adjusted

    def KNN(self, X, Y, total_neighbors = 7):
        cv = KFold(n_splits=10, shuffle=True)
        cross_scoring = ['neg_root_mean_squared_error', 'r2']
        model = KNeighborsRegressor(n_neighbors= total_neighbors)
        scores = cross_val_score(model, X, Y, scoring=cross_scoring, cv=cv, n_jobs=-1)
        #knn.fit(x_train, y_train)
        #y_pred = knn.predict(x_test)
        #rmse = metrics.mean_squared_error(y_test, y_pred, squared=False) # root mean squared error
        #print("KNN RSME = " + str(rmse))
        # do adjusted_r2
        #r2 = 0 #placeholder
        return scores

    def find_best_N(self, x_train, y_train): # will output best N -> not sure if we want to use this
        params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
        knn = KNeighborsRegressor()
        model = GridSearchCV(knn, params)
        model.fit(x_train, y_train)
        return model.best_params_

    def descent(self, X, Y):
        cv = KFold(n_splits=10, shuffle=True)
        cross_scoring = ['neg_root_mean_squared_error', 'r2']
        model = SGDRegressor()
        scores = cross_val_score(model, X, Y, scoring=cross_scoring, cv=cv, n_jobs=-1)
        #for i in range(0, fold):
        #    model.partial_fit(x_train[i], y_train[i])

        #once done get values
        #y_pred = model.predict(x_test)
        #rmse = metrics.mean_squared_error(y_test, y_pred, squared=False) # root mean squared error
        #print("Descent RSME = " + str(rmse))
        # do adjusted_r2
        #r2 = 0 #placeholder
        return scores

    def lRegression(self, X, Y):
        cv = KFold(n_splits=10, shuffle=True)
        cross_scoring = ['neg_root_mean_squared_error', 'r2']
        model = LinearRegression()
        scores = cross_val_score(model, X, Y, scoring=cross_scoring, cv=cv, n_jobs=-1)
        #model.fit(x_train,y_train)
        #y_pred = model.predict(x_test)
        #rmse = metrics.mean_squared_error(y_test, y_pred, squared=False) # root mean squared error
        #print("Linear Regression RSME = " + str(rmse))
        # do adjusted_r2
        #r2 = 0 #placeholder
        return scores

    def createNetwork(self):
        network = models.Sequential()
        network.add(layers.Dense(128, activation='relu', input_shape=(self.features,)))
        network.add(layers.Dense(64, activation='relu'))
        network.add(layers.Dense(1, activation='linear'))
        network.compile(optimizer='adam', loss='mse', metrics=["root_mean_squared_error", "mae"])
        return network
        
    """ def NeuralNetwork(self, x_train, y_train, x_test, y_test, n_shapes):
        network = models.Sequential()
        network.add(layers.Dense(24, activation='relu', input_shape=(n_shapes,)))
        network.add(layers.Dense(24, activation='relu'))
        network.add(layers.Dense(1))
        network.compile(optimizer=optimizers.RMSprop(lr=0.01), loss='mse', metrics=['mae'])
        output = network.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=18,batch_size=20)
        output_values = output.history
        sme = output_values['loss']
        validation_sme = output_values['val_loss']
        # will need to take sqrt of these validation values ^
        #calculate r2 
        r2 = 0
        return rmse """

    def doNeuralNetwork(self, X, Y):
        neural_network = KerasClassifier(build_fn=createNetwork, 
                                 epochs=15, 
                                 batch_size=500, 
                                 verbose=1)
        cv = KFold(n_splits=10, shuffle=True)
        results = cross_val_score(neural_network, X, Y, cv=cv)
        return results



