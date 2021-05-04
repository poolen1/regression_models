from models import Model
import pandas as pd
import numpy as np

# needs data to be added :) 
def main():
    features = 8
    doML = Model(features)

    data_set = pd.read_csv('data_set.csv')
    y_list = data_set.loc[:, 'amount']
    X_list = data_set.drop('amount', axis=1)
    Y = y_list.values
    X = np.array(X_list)

    #print(X)
    #print(Y)
    #exit()
    
    for i in range(0,1): # repeat each experiment 3 times
        print("Run # :"+str(i))
        print("Doing KNN")
        knn_results = doML.KNN(X, Y)
        print("KNN done -- Doing LR")
        LR_results = doML.lRegression(X, Y)
        print("LR done -- Doing Gradient")
        Gradient_results = doML.descent(X, Y)
        print("Gradient done -- Doing NN")
        NN_results = doML.doNeuralNetwork(X, Y)
        print("KNN results are: "+str(knn_results))
        print("Linear Regression results are: "+str(LR_results))
        print("Gradient results are: "+str(Gradient_results))
        print("NN results are: "+str(NN_results))



if __name__ == "__main__":
    main()