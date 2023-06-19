import pandas
import numpy as np

#returns a design matrix Xtilde for SGD
#trainingImages: a 2D array where each row is an entry, and each column is a statistic
#returns: a 2D array, equal to trainingImages but with an extra row of ones to compute the bias
def getDesignMatrix(trainingImages):
    xShape = np.shape(trainingImages) #shape = (#rows, #col); each row is an image, each col is a pixel in the image
    vectorOf1s = np.ones(shape = (xShape[0], 1)) #creates a 1D column vector of ones
    return(np.append(arr = trainingImages, values = vectorOf1s, axis = 1)) #added a vector of 1s for bias

# Given training and testing data, learning rate epsilon, batch size, and regularization strength alpha,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix Wtilde (785x10).
# Then return Wtilde.
def softmaxRegression (Xtilde, y, epsilon, batchSize, alpha, classes):
    epochCount = 1

    #the following code initializes Wtilde
    Wtilde = np.random.randn(np.shape(Xtilde)[1], classes) * 1e-5 #creates a 2D array where each row is the weight for a statistic, and each column is the weight vector
    #the following code shuffles the data
    rng_state = np.random.get_state() #lets me shuffle both the values and labels so they still match up
    np.random.shuffle(Xtilde) #randomly shuffles the rows
    np.random.set_state(rng_state) #resets the state to what it was for Xtilde
    np.random.shuffle(y) #shuffles y in the same way

    trainingLosses = [] #keeps track of the training loss throughout training
    for epoch in range(epochCount): #for every epoch out of epochCount do:
        startPoint = 0 #the starting index of entries we parse in a batch

        for j in range(np.int32(np.ceil(np.shape(Xtilde)[0]/batchSize))): #for every mini-batch j do:
            subsetRange = None #the entries in a batch

            #if the final batch is not >= batchSize, this batch contains whatever is left
            if(np.shape(Xtilde[startPoint:-1, :])[0] < batchSize):
                subsetRange = range(startPoint, j*batchSize + np.shape(Xtilde[startPoint:-1, :])[0])
            #else, we know that we have enough images for a full batch
            else:
                subsetRange = range(startPoint, j*batchSize + batchSize)

            xSubset = Xtilde[subsetRange] # select rows [subsetRange, +batchSize)
            ySubset = y[subsetRange, :] #gets the subset of ground truth values
            yhat = softmax(np.asarray(xSubset).dot(Wtilde)) #computes yhat for subset
            gradient = computeGradient(xSubset, yhat, ySubset, Wtilde, batchSize, alpha) #computes the regularized gradient
            Wtilde = Wtilde - (gradient*epsilon) #applies changes to Wtilde

            startPoint += batchSize #increases the startpoint by batchSize so we don't start from 0 again
        
        trainingLosses.append(fCE(Xtilde, y, Wtilde, alpha))
        print("Epoch", epoch + 1, "completed...")
        print("training loss: ", trainingLosses[epoch])

    return Wtilde

#returns a normalized yhat
#yhat: a 2D array where each row is the probability vector and each column is the probability that an entry survives
#returns: a 2D array that is a normalized yhat
def softmax(yhat):
    yhat = np.exp(yhat) #enforce non negativity
    summation = np.sum(yhat, axis = 1) #sums each row together

    for row in range(np.shape(yhat)[0]):
        yhat[row] = yhat[row] / summation[row]  #enforces the sum of each prediction for an image = 1
    return yhat


#computes the regularized gradient of fCE
#x: a 2D array, where each row is an entry, and each column is statistic
#yhat: a 2D array, where each row is a set of normalized probabilities that an entry survived
#y: a 2D array, where each row is the ground-truth classifications of an image
#Wtilde: a 2D array, where each row is the weights for each classification applied to an entry stat, and each column is an image
#n: an int; the batch size
#returns: a float64 of the gradient of fCE
def computeGradient(x, yhat, y, Wtilde, n, alpha):
    gradFCE = np.transpose(x).dot(yhat - y) / n #computes unregularized gradient
    regularize = alpha * np.mean(a = Wtilde[:-1], axis = 0) #gets the derivative of the L2 regularization without bias
    regularize = np.repeat(a = [regularize], repeats = np.shape(gradFCE)[0], axis = 0)
    return(gradFCE + regularize)

#1-hot encodes the labels
#labels: a 1D array of classifications
#classes: the number of labels
#returns: a 2D array where each column represents a classification
def oneHotEncode(labels, classes):
    y = np.zeros((len(labels), classes)) #creates a 2D array of the proper size
    for i in range(len(labels)): #sets the index of the label to 1
        y[i, labels[i]] = 1
    return y


def fCE(x, y, Wtilde, alpha):
    yhat = np.asarray(x).dot(Wtilde)
    yhat = softmax(yhat)
    sum = np.sum(y * np.log(yhat), axis = 0)
    unreg = np.mean(sum) * -1
    reg = alpha * 0.5 * np.mean(np.sum(Wtilde * Wtilde, axis = 0))
    return unreg + reg

def interpretPredictions(passengerIDs, probabilities):
    predictions = np.argmax(probabilities, axis = 1)

    if(np.shape(predictions) != np.shape(passengerIDs)):
        print("ERROR: INCOMPATIBLE SHAPES")

    else:
        entries = np.stack((passengerIDs, predictions), axis = 1)
        pandasDF = pandas.DataFrame(data = entries, columns = ["PassengerId","Survived"])
        pandasDF.to_csv('predictionsJReiss.csv', index = False)



#STARTER CODE=================================================================================
if __name__ == "__main__":
    # Load training data
    d = pandas.read_csv("train.csv")
    y = d.Survived.to_numpy()
    sex = d.Sex.map({"male":0, "female":1}).to_numpy()
    Pclass = d.Pclass.to_numpy()

    # Train model using part of homework 3.
    classes = 2 #0 = died, 1 = survived

    Xtilde = np.transpose(np.stack((sex, Pclass))) #merges sex and Pclass into 1 array
    Xtilde = getDesignMatrix(Xtilde) #gets the design matrix
    y = oneHotEncode(y, classes)
    
    Wtilde = softmaxRegression(Xtilde, y, epsilon=0.1, batchSize=100, alpha=.1, classes = classes)

    # Load testing data
    dTest = pandas.read_csv('test.csv')
    sexTest = dTest.Sex.map({"male":0, "female":1}).to_numpy()
    PclassTest = dTest.Pclass.to_numpy()

    # Compute predictions on test set
    datasetTest = np.transpose(np.stack((sexTest, PclassTest)))
    datasetTest = getDesignMatrix(datasetTest)
    predictions = np.asarray(datasetTest).dot(Wtilde)
    predictions = softmax(predictions)

    # Write CSV file of the format:
    # PassengerId, Survived
    interpretPredictions(dTest.PassengerId.to_numpy(), predictions)
