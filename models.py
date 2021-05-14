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
        self.features = n_features
        self.cv = KFold(n_splits=10, shuffle=True)
        self.cross_scoring = ['neg_root_mean_squared_error', 'r2']

    def KNN(self, X, Y, total_neighbors = 7):
        model = KNeighborsRegressor(n_neighbors= total_neighbors)
        scores = cross_validate(model, X, Y, scoring=self.cross_scoring, cv=self.cv, n_jobs=-1)
        return scores

    def find_best_N(self, x_train, y_train): # will output best N -> not sure if we want to use this
        params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
        knn = KNeighborsRegressor()
        model = GridSearchCV(knn, params)
        model.fit(x_train, y_train)
        return model.best_params_

    def descent(self, X, Y, folds, n_chunks):
        gmodel = SGDRegressor(loss="huber",eta0=0.00000001)
        scores = cross_validate(gmodel, X, Y, scoring=self.cross_scoring, cv=self.cv, n_jobs=-1)

        # graphing
        """ train_sizes, train_scores, test_scores = model_selection.learning_curve(estimator=gmodel, X=X, y=Y, cv=10)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
        axes[0].set_title("graph")
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")
        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1,
                            color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1,
                            color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                    label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                    label="Cross-validation score")
        axes[0].legend(loc="best")
        plt.show() """

        # mini batch
        """ x_folds = np.array_split(X, folds)
        y_folds = np.array_split(Y, folds)
        scores = {}
        r2 = []
        rmse = []
        #print(X)

        for i in range(0, folds):
            # split into test and training --> current i value will be the test set
            list_o_nums = []

            for k in range(0, folds):
                if k != i:
                    list_o_nums.append(k)

            y_test = y_folds[i]
            x_test = x_folds[i]
            for j in range(0, len(list_o_nums)):
                if j == 0:
                    x_train = x_folds[list_o_nums[j]]
                    y_train = y_folds[list_o_nums[j]]
                else:
                    x_train = np.concatenate((x_train, x_folds[list_o_nums[j]]))
                    y_train = np.concatenate((y_train, y_folds[list_o_nums[j]]))

            x_chunk = np.array_split(x_train, n_chunks)
            y_chunk = np.array_split(y_train, n_chunks)

            for counter in range(0, n_chunks):
                gmodel.partial_fit(x_chunk[counter], y_chunk[counter])

            #once done get values

            y_pred = gmodel.predict(x_test)
            rmse.append(metrics.mean_squared_error(y_test, y_pred, squared=False)) # root mean squared error
            r2.append(metrics.r2_score(y_test, y_pred))
            #plt.plot(x_test, y_pred, "r-")
            #plt.plot(x_test, y_test, "b.")
            #plt.show()

        scores["r2"] = r2
        scores["rmse"] =rmse """

        

        return scores

    def lRegression(self, X, Y):
        model = LinearRegression()
        scores = cross_validate(model, X, Y, scoring=self.cross_scoring, cv=self.cv, n_jobs=-1)
        return scores

    def create_Network(self):
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
        return results



