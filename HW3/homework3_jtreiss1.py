import numpy as np
import matplotlib.pyplot as plt

# Given training and testing data, learning rate epsilon, batch size, and regularization strength alpha,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix Wtilde (785x10).
# Then return Wtilde.
def softmaxRegression (trainingImages, trainingLabels, testingImages, testingLabels, epsilon, batchSize, alpha):
    classes = 10 #we will have 10 w vectors in w
    epochCount = 5

    #the following code initializes Wtilde
    Wtilde = np.random.randn(np.shape(trainingImages)[1], classes) * 1e-5 #creates a 785 x 10 array, each column refers to the weights of a class, each row is the weight of a pixel in a class, the last row is the bias
    #the following code shuffles the data
    rng_state = np.random.get_state() #lets me shuffle both the images and labels so they still match up
    np.random.shuffle(trainingImages) #randomly shuffles the rows (images)
    np.random.set_state(rng_state) #resets the state to what it was for Xtilde
    np.random.shuffle(trainingLabels) #shuffles trainingLabels in the same way

    trainingLosses = [] #keeps track of the training loss throughout training
    last20Batches = [] #keeps track of the training loss for the final 20 batches
    for epoch in range(epochCount): #for every epoch out of epochCount do:
        startPoint = 0 #the starting index of images we parse in a batch

        for j in range(np.int32(np.ceil(np.shape(trainingImages)[0]/batchSize))): #for every mini-batch j out of 60000/batchSize = 60 batches do:
            subsetRange = None #the images in a batch

            #if the final batch is not >= batchSize, this batch contains whatever is left
            if(np.shape(trainingImages[startPoint:-1, :])[0] < batchSize):
                subsetRange = range(startPoint, j*batchSize + np.shape(trainingImages[startPoint:-1, :])[0])
            #else, we know that we have enough images for a full batch
            else:
                subsetRange = range(startPoint, j*batchSize + batchSize)

            xSubset = trainingImages[subsetRange, :] # select rows [subsetRange, +batchSize)
            ySubset = trainingLabels[subsetRange, :] #gets the subset of ground truth values
            yhat = softmax(np.asarray(xSubset).dot(Wtilde)) #computes yhat for subset
            gradient = computeGradient(xSubset, yhat, ySubset, Wtilde, batchSize, alpha) #computes the regularized gradient
            Wtilde = Wtilde - (gradient*epsilon) #applies changes to Wtilde

            startPoint += batchSize #increases the startpoint by batchSize so we don't start from 0 again

            #keeps track of the final 20 batches
            if((epoch == epochCount - 1) and (j >= (np.shape(trainingImages)[0]/batchSize) - 20)):
                last20Batches.append(fCE(trainingImages, trainingLabels, Wtilde, alpha))
        
        trainingLosses.append(fCE(trainingImages, trainingLabels, Wtilde, alpha))
        print("Epoch", epoch + 1, "completed...")
        print("training loss: ", trainingLosses[epoch])

    print("\n==================\nTraining fCE for final 20 batches:")
    count = 1
    for v in last20Batches:
        print(str(count) + ':', v)
        count += 1
    return Wtilde

#returns a design matrix Xtilde for SGD
#trainingImages: a 2D array where each row is an image, and each column is a pixel
#returns: a 2D array, equal to trainingImages but with an extra row of ones to compute the bias
def getDesignMatrix(trainingImages):
    xShape = np.shape(trainingImages) #shape = (#rows, #col); each row is an image, each col is a pixel in the image
    vectorOf1s = np.ones(shape = (xShape[0], 1)) #creates a 1D column vector of size (60000, 1)
    return(np.append(arr = trainingImages, values = vectorOf1s, axis = 1)) #added a vector of 1s for bias     

#returns a normalized yhat
#yhat: a 2D array where each row is the probability vector of an image, and each column is the probability that an image is of a specific classification
#returns: a 2D array that is a normalized yhat
def softmax(yhat):
    yhat = np.exp(yhat) #enforce non negativity

    summation = np.sum(yhat, axis = 1) #sums each row together
    for row in range(np.shape(yhat)[0]):
        yhat[row] = yhat[row] / summation[row]  #enforces the sum of each prediction for an image = 1
    return yhat


#computes the regularized gradient of fCE
#x: a 2D array, where each row is an image, and each column is a pixel in an image
#yhat: a 2D array, where each row is a set of normalized probabilities that an image is of a classification
#y: a 2D array, where each row is the ground-truth classifications of an image
#Wtilde: a 2D array, where each row is the weights for each classification applied to a pixel in an image, and each column is an image
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

def fPC(yhat, y):
     predictions = np.argmax(yhat, axis = 1) #produces a vector of the indeces of the maximum value in each row
     print(str(np.mean(np.equal(predictions, y)) * 100) + '% Correct')

if __name__ == "__main__":
    # Load data
    trainingImages = np.load("fashion_mnist_train_images.npy") / 255.0  # Normalizing by 255 helps accelerate training
    trainingLabels = np.load("fashion_mnist_train_labels.npy")
    testingImages = np.load("fashion_mnist_test_images.npy") / 255.0  # Normalizing by 255 helps accelerate training
    testingLabels = np.load("fashion_mnist_test_labels.npy")

    # Append a constant 1 term to each example to correspond to the bias terms
    trainingImages = getDesignMatrix(trainingImages)
    # Change from 0-9 labels to "one-hot" binary vector labels. For instance, 
    # if the label of some example is 3, then its y should be [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ]
    trainingLabels = oneHotEncode(trainingLabels, 10)

    # Train the model
    Wtilde = softmaxRegression(trainingImages, trainingLabels, testingImages, testingLabels, epsilon=0.1, batchSize=100, alpha=.1)

    #c) computes PC accuracy on test set
    testingImages = getDesignMatrix(testingImages)
    yhatTest = np.asarray(testingImages).dot(Wtilde)
    yhatTest = softmax(yhatTest)
    
    print("\n==================\nfPC:")
    fPC(yhatTest, testingLabels)

    # Visualize the vectors
    for Wvector in np.transpose(Wtilde[:-1]):
        vector = np.reshape(a = Wvector, newshape = (28, 28))
        plt.imshow(vector)
        plt.show()