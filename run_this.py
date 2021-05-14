from models import Model
import pandas as pd
import numpy as np

def main():
    datafile = 'newdata_set.csv'
    features = 9
    # activate model with features
    doML = Model(features)
    
    # read in data
    data_set = pd.read_csv(datafile) # read in data
    data_set = data_set.sample(frac=1).reset_index(drop=True) # shuffle
    y_list = data_set.loc[:, 'amount'] # y = amount 
    X_list = data_set.drop(['amount', 'offer_id'], axis=1) # x = rest of features aside from amount and offer id 
    # get values
    Y = y_list.values
    X = np.array(X_list)

    # promotion id analytics -- save id
    promotion_id = []
    promotion_id.append("ae264e3637204a6fb9bb56bc8210ddfd")
    promotion_id.append("4d5c57ea9a6940dd891ad53e9dbe8da0")
    promotion_id.append("9b98b8c7a33c4b65b9aebfe6a799e6d9")
    promotion_id.append("0b1e1539f2cc45b7b9fa7c272da2e1d7")
    promotion_id.append("2298d6c36e964ae4a3e7e9706d1fb8c2")
    promotion_id.append("fafdcd668e3743c1bb461111dcafc2a4")
    promotion_id.append("f19421c1d4aa40978ebb69ca19b0e20d")
    promotion_id.append("2906b810c7d4411798c6938adc9daaa5")
    promotion_id.append("0")

    # grab sums of each offer in transaction list
    idsum = []
    idsum.append(data_set.loc[data_set['offer_id'] == "ae264e3637204a6fb9bb56bc8210ddfd", 'amount'].sum())
    idsum.append(data_set.loc[data_set['offer_id'] == "4d5c57ea9a6940dd891ad53e9dbe8da0", 'amount'].sum())
    idsum.append(data_set.loc[data_set['offer_id'] == "9b98b8c7a33c4b65b9aebfe6a799e6d9", 'amount'].sum())
    idsum.append(data_set.loc[data_set['offer_id'] == "0b1e1539f2cc45b7b9fa7c272da2e1d7", 'amount'].sum())
    idsum.append(data_set.loc[data_set['offer_id'] == "2298d6c36e964ae4a3e7e9706d1fb8c2", 'amount'].sum())
    idsum.append(data_set.loc[data_set['offer_id'] == "fafdcd668e3743c1bb461111dcafc2a4", 'amount'].sum())
    idsum.append(data_set.loc[data_set['offer_id'] == "f19421c1d4aa40978ebb69ca19b0e20d", 'amount'].sum())
    idsum.append(data_set.loc[data_set['offer_id'] == "2906b810c7d4411798c6938adc9daaa5", 'amount'].sum())
    idsum.append(data_set.loc[data_set['offer_id'] == "0", 'amount'].sum())

    # grab indices of each ID post-shuffle
    indices = []
    index = data_set.index
    condition = data_set["offer_id"] == "ae264e3637204a6fb9bb56bc8210ddfd"
    indices.append(index[condition])
    condition = data_set["offer_id"] == "4d5c57ea9a6940dd891ad53e9dbe8da0"
    indices.append(index[condition])
    condition = data_set["offer_id"] == "9b98b8c7a33c4b65b9aebfe6a799e6d9"
    indices.append(index[condition])
    condition = data_set["offer_id"] == "0b1e1539f2cc45b7b9fa7c272da2e1d7"
    indices.append(index[condition])
    condition = data_set["offer_id"] == "2298d6c36e964ae4a3e7e9706d1fb8c2"
    indices.append(index[condition])
    condition = data_set["offer_id"] == "fafdcd668e3743c1bb461111dcafc2a4"
    indices.append(index[condition])
    condition = data_set["offer_id"] == "f19421c1d4aa40978ebb69ca19b0e20d"
    indices.append(index[condition])
    condition = data_set["offer_id"] == "2906b810c7d4411798c6938adc9daaa5"
    indices.append(index[condition])
    condition = data_set["offer_id"] == "0"
    indices.append(index[condition])
    
    for i in range(0,1): # repeat each experiment 3 times
        print("Run # :"+str(i))
        print("Doing KNN")
        knn_results, knn_pred = doML.KNN(X, Y)
        print("KNN done -- Doing LR")
        LR_results, LR_pred = doML.lRegression(X, Y)
        print("LR done -- Doing Gradient")
        Gradient_results, Gradient_pred = doML.descent(X, Y, 10, 2600)
        print("Gradient done -- Doing NN")
        NN_results, NN_pred = doML.doNeuralNetwork(X, Y)

        #calculate id sums
        knn_sums = []
        LR_sums = []
        Gradient_sums = []
        NN_sums = []
        for i in range(0, len(indices)):
            knn_sums.append(knn_pred[indices[i]].sum())
            LR_sums.append(LR_pred[indices[i]].sum())
            Gradient_sums.append(Gradient_pred[indices[i]].sum())
            NN_sums.append(NN_pred[indices[i]].sum())

        print("KNN results are: "+str(knn_results))
        print("Linear Regression results are: "+str(LR_results))
        print("Gradient results are: "+str(Gradient_results))
        print("NN results are: "+str(NN_results))

        print("Average of transactions = "+str(np.mean(idsum)))
        print("Average of KNN = "+str(np.mean(knn_sums)))
        print("Average of Gradient = "+str(np.mean(Gradient_sums)))
        print("Average of LR = "+str(np.mean(LR_sums)))
        print("Average of NN = "+str(np.mean(NN_sums)))


        # show sum of predictions -- which offer type is most valuable? 
        print("Sums")
        print("KNN Predictions")
        for i in range(0, len(indices)):
            print("Offer ID: " + str(promotion_id[i])+ " sum = "+str(knn_sums[i]))
        print("Linear Regression Predictions")
        for i in range(0, len(indices)):
            print("Offer ID: " + str(promotion_id[i])+ " sum = "+str(LR_sums[i]))

        print("Gradient Predictions")
        for i in range(0, len(indices)):
            print("Offer ID: " + str(promotion_id[i])+ " sum = "+str(Gradient_sums[i]))

        print("NN Predictions")
        for i in range(0, len(indices)):
            print("Offer ID: " + str(promotion_id[i])+ " sum = "+str(NN_sums[i]))



if __name__ == "__main__":
    main()