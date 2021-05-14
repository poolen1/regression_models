from sklearn import model_selection
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, learning_curve
from sklearn.linear_model import LinearRegression, SGDRegressor
from keras import models, layers, optimizers
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

class Model:
    def __init__(self, n_features):
        self.features = n_features # feature count
        self.cv = KFold(n_splits=10, shuffle=False)  # shuffle = false because data is shuffled in run_this.py
        self.cross_scoring = ['neg_root_mean_squared_error', 'r2'] # return r2 and RSME (negative)

    def KNN(self, X, Y, total_neighbors = 7):
        model = KNeighborsRegressor(n_neighbors= total_neighbors) # model
        scores = cross_validate(model, X, Y, scoring=self.cross_scoring, cv=self.cv, n_jobs=-1) # rmse and r2
        prediction = model_selection.cross_val_predict(model, X, Y, cv=self.cv, n_jobs=-1) # prediction values
        return scores, prediction

    def find_best_N(self, x_train, y_train): # will output best N 
        params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
        knn = KNeighborsRegressor()
        model = GridSearchCV(knn, params)
        model.fit(x_train, y_train)
        return model.best_params_

    def descent(self, X, Y, folds, n_chunks):
        gmodel = SGDRegressor(loss="huber",eta0=0.00000001) # model -- this loss rate gave us the best accuracy, huber gave us results as well

        # stochastic model -- was used previously
        """ scores = cross_validate(gmodel, X, Y, scoring=self.cross_scoring, cv=self.cv, n_jobs=-1)
        prediction = model_selection.cross_val_predict(gmodel, X, Y, cv=self.cv, n_jobs=-1) """

        # mini batch
        x_folds = np.array_split(X, folds)
        y_folds = np.array_split(Y, folds)
        scores = {}
        r2 = []
        rmse = []
        predicted_values = []
        #print(X)

        for i in range(0, folds):
            # split into test and training --> current i value will be the test set
            list_o_nums = [] # this is used to combine the training set into one np array (it is currently an array of 9 arrays)

            for k in range(0, folds):
                if k != i:
                    list_o_nums.append(k)

            y_test = y_folds[i]
            x_test = x_folds[i]

            # combine test data
            for j in range(0, len(list_o_nums)):
                if j == 0:
                    x_train = x_folds[list_o_nums[j]]
                    y_train = y_folds[list_o_nums[j]]
                else:
                    x_train = np.concatenate((x_train, x_folds[list_o_nums[j]]))
                    y_train = np.concatenate((y_train, y_folds[list_o_nums[j]]))

            # split training data into n_chunks for mini-batch
            x_chunk = np.array_split(x_train, n_chunks) 
            y_chunk = np.array_split(y_train, n_chunks)

            # mini batch
            for counter in range(0, n_chunks):
                gmodel.partial_fit(x_chunk[counter], y_chunk[counter])

            #once done get values

            y_pred = gmodel.predict(x_test)
            rmse.append(metrics.mean_squared_error(y_test, y_pred, squared=False)) # root mean squared error
            r2.append(metrics.r2_score(y_test, y_pred))
            predicted_values.append(y_pred)
            #plt.plot(x_test, y_pred, "r-")
            #plt.plot(x_test, y_test, "b.")
            #plt.show()

        flat_list = [item for sublist in predicted_values for item in sublist] # predictions is list of lists --> turn into one list
        prediction = np.array(flat_list) # np array
        scores["r2"] = r2 # save r2 in scores
        scores["rmse"] =rmse # save rmse ins cores
        return scores, prediction

    def lRegression(self, X, Y):
        model = LinearRegression()
        scores = cross_validate(model, X, Y, scoring=self.cross_scoring, cv=self.cv, n_jobs=-1)
        prediction = model_selection.cross_val_predict(model, X, Y, cv=self.cv, n_jobs=-1)
        return scores, prediction

    def create_Network(self): # creates network for cross_validate and cross_val_predict
        network = models.Sequential()
        network.add(layers.Dense(128, activation='relu', input_shape=(self.features,)))
        network.add(layers.Dense(64, activation='relu'))
        network.add(layers.Dense(1, activation='linear'))
        network.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(), "mae"]) 
        return network
        
    def doNeuralNetwork(self, X, Y):
        neural_network = KerasClassifier(build_fn = self.create_Network, 
                                 epochs=20, 
                                 batch_size=500, 
                                 verbose=1)
        results = cross_validate(neural_network, X, Y, scoring=self.cross_scoring, cv=self.cv)
        prediction = model_selection.cross_val_predict(neural_network, X, Y, cv=self.cv, n_jobs=-1)
        return results, prediction



