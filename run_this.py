from models import Model

# needs data to be added :) 
def main():
    features = 7
    doML = Model(features)

    # X = pd.read_csv('X.csv')
    # Y = pd.read_csv('Y.csv')
    
    # for i in range(0,3): # repeat each experiment 3 times
        # print("Run # :"+str(i))
        # print("Doing KNN")
        # knn_results = doML.KNN(X, Y)
        # print("KNN done -- Doing LR")
        # LR_results = doML.lRegression(X, Y)
        # print("LR done -- Doing Gradient")
        # Gradient_results = doML.descent(X, Y)
        # print("Gradient done -- Doing NN")
        # NN_results = doML.doNeuralNetwork(X, Y)
        # print("KNN results are: "+str(knn_results))
        # print("Linear Regression results are: "+str(LR_results))
        # print("Gradient results are: "+str(Gradient_results))
        # print("NN results are: "+str(NN_results))



if __name__ == "__main__":
    main()