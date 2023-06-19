import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

IM_WIDTH = 48
NUM_INPUT = IM_WIDTH**2
NUM_HIDDEN = 20
NUM_OUTPUT = 1

#computes the half-mean-square error loss
#W1: an array of weights for the first hidden layer in the neural network
#b1: the biases for W1 as a vector
#W2: an array of weights for the output of the neural network
#b2: a vector of biases for W2
def fMSE(y, yhat):
    return 0.5 * np.mean(np.square(yhat - y))

def relu (z):
    return np.maximum(0, z)

def forward_prop (x, y, W1, b1, W2, b2):
    #next three lines are from equations in assignment
    z = np.add(np.asarray(W1).dot(x), b1[:, np.newaxis])
    h = relu(z)
    yhat = np.asarray(W2).dot(h) + b2
    loss = fMSE(y, yhat)

    return loss, x, z, h, yhat
   
def back_prop (X, y, W1, b1, W2, b2, alpha = 0):
    loss, X, z, h, yhat = forward_prop(X, y, W1, b1, W2, b2)
    g = np.multiply(np.transpose(yhat - y).dot(W2), np.heaviside(np.transpose(z), 0))
    g = np.transpose(g)
    gradW1 = g.dot(np.transpose(X)) + (alpha*W1)
    gradb1 = np.mean(g, axis = 1)
    gradW2 = np.asarray(yhat - y).dot(np.transpose(h)) + (alpha*W2)
    gradb2 = np.mean(yhat - y)
    #print('\ngradient shapes:\nW1:', np.shape(gradW1), '\nW2:', np.shape(gradW2), '\nb1:', np.shape(gradb1), '\nb2:', np.shape(gradb2))

    return gradW1, gradb1, gradW2, gradb2

def train(trainX, trainY, W1, b1, W2, b2, testX, testY, epsilon = 1e-3, batchSize = 256, numEpochs = 25):
    batches = range(np.int32(np.ceil(np.shape(trainX)[1]/batchSize)))
    for epoch in range(numEpochs):
        if(epoch % 100 == 0):
            epsilon = epsilon/10

        startPoint = 0
        for batch in batches:
            print('On epoch ' + str(epoch + 1) + ". Batch: " + str(batch + 1), end = '\r')
            subsetRange = None #the images in a batch

            #if the final batch is not >= batchSize, this batch contains whatever is left
            if(np.shape(trainX[:, startPoint:-1])[1] < batchSize):
                subsetRange = range(startPoint, batch*batchSize + np.shape(trainX[:, startPoint:-1])[1])
            #else, we know that we have enough images for a full batch
            else:
                subsetRange = range(startPoint, batch*batchSize + batchSize)

            xSubset = trainX[:, subsetRange] # select rows [subsetRange, +batchSize)
            ySubset = trainY[subsetRange] #gets the subset of ground truth values
            gradW1, gradb1, gradW2, gradb2 = back_prop(xSubset, ySubset, W1, b1, W2, b2, 1e-3)

            loss = forward_prop(xSubset, ySubset, W1, b1, W2, b2)[0] #inefficient; find better way

            W1 = W1 - (epsilon * gradW1)
            W2 = W2 - (epsilon * gradW2)
            b1 = b1 - (epsilon * gradb1) 
            b2 = b2 - (epsilon * gradb2)

            startPoint += batchSize
            print(end = '\x1b[2K')
            
            if(epoch == numEpochs - 1 and (batch >= batches[-1] - 20)):
                lossTesting = forward_prop(testX, testY, W1, b1, W2, b2)[0]
                print('batch ' + str(batch + 1) + ': TrainingLoss =', loss, '    TestingLoss =', lossTesting)

    return W1, b1, W2, b2

def show_weight_vectors (W1):
    # Show weight vectors in groups of 5.
    for i in range(NUM_HIDDEN//5):
        plt.imshow(np.hstack([ np.pad(np.reshape(W1[idx,:], [ IM_WIDTH, IM_WIDTH ]), 2, mode='constant') for idx in range(i*5, (i+1)*5) ]), cmap='gray'), plt.show()
    plt.show()

def loadData (which, mu = None):
    images = np.load("age_regression_X{}.npy".format(which)).reshape(-1, 48**2).T
    labels = np.load("age_regression_y{}.npy".format(which))

    if which == "tr":
        mu = np.mean(images)

    # TODO: you may wish to perform data augmentation (e.g., left-right flipping, adding Gaussian noise).

    return images - mu, labels, mu

def checkGradient():
    testW1 = np.load("testW1.npy")
    testb1 = np.load("testb1.npy")
    testW2 = np.load("testW2.npy")
    testb2 = np.load("testb2.npy")
    oneSampleX = np.load("oneSampleX.npy")
    oneSampley = np.load("oneSampley.npy")
    gradW1, gradb1, gradW2, gradb2 = back_prop(np.atleast_2d(oneSampleX).T, oneSampley, testW1, testb1, testW2, testb2)
    correctGradW1 = np.load("correctGradW1OnSample.npy")
    correctGradb1 = np.load("correctGradb1OnSample.npy")
    correctGradW2 = np.load("correctGradW2OnSample.npy")
    correctGradb2 = np.load("correctGradb2OnSample.npy")
    # The differences should all be <1e-5
    print("CORRECT GRAD CONFIRMATION ============")
    print(np.sum(np.abs(gradW1 - correctGradW1)))
    print(np.sum(np.abs(gradb1 - correctGradb1)))
    print(np.sum(np.abs(gradW2 - correctGradW2)))
    print(np.sum(np.abs(gradb2 - correctGradb2)))
    print('\n')

if __name__ == "__main__":
    # Load data
    if "trainX" not in globals():
        trainX, trainY, mu = loadData("tr")
        testX, testY, _ = loadData("te", mu)

    # Check the gradient value for correctness.
    # Note: the gradients shown below assume 20 hidden units.
    checkGradient()

    # Initialize weights to reasonable random values
    W1 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    b2 = np.mean(trainY)

    # Train NN
    W1, b1, W2, b2 = train(trainX, trainY, W1, b1, W2, b2, testX, testY)

    #Test NN
    lossTe = forward_prop(testX, testY, W1, b1, W2, b2)[0]
    lossTr = forward_prop(trainX, trainY, W1, b1, W2, b2)[0]
    print('\nfinal testing accuracy:', lossTe)
    print('final training accuracy:', lossTr)
    show_weight_vectors(W1)