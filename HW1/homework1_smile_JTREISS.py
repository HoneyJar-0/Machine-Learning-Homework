import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def fPC (y, yhat):
    return (np.mean(np.equal(y, yhat)))

def measureAccuracyOfPredictors (predictors, X, y):
    gj = []
    for feature in predictors:
        gjOfPhi = np.greater(X[feature[0], feature[1],:] - X[feature[2], feature[3],:], 0)
        gj.append(gjOfPhi)
    
    yhat = np.greater(np.mean(gj, axis = 0), 0.5)

    return(fPC(y, yhat))

def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels):
    X = np.transpose(trainingFaces)
    predictors = []
    
    for features in range(6):
        tuples = []
        accs = []
        for r1 in range(np.shape(X)[1]):
            for c1 in range(np.shape(X)[0]):
                for r2 in range(np.shape(X)[1]):
                    for c2 in range(np.shape(X)[0]):
                        if((r1*np.shape(X)[0] + c1) < (r2*np.shape(X)[0] + c2)):
                            testPredictors = predictors.copy()
                            testPredictors.append((r1, c1, r2, c2))

                            accs.append(measureAccuracyOfPredictors(testPredictors, X, trainingLabels))
                            tuples.append((r1,c1,r2,c2))

        indexOfMax = np.argmax(accs)
        predictors.append(tuples[indexOfMax])
        print("feature", features, " out of 5 completed.")

    print("training accuracy:", measureAccuracyOfPredictors(predictors, X, trainingLabels))
    print("testing accuracy:", measureAccuracyOfPredictors(predictors, np.transpose(testingFaces),testingLabels))

                


    

    show = True
    if show:
        im = testingFaces[0,:,:]
        fig,ax = plt.subplots(1)
        ax.imshow(im, cmap='gray')
        for p in predictors:
            r1 = p[0]
            c1 = p[1]
            r2 = p[2]
            c2 = p[3]
            # Show an arbitrary test image in grayscale
           
            # Show r1,c1
            rect = patches.Rectangle((c1 - 0.5, r1 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            # Show r2,c2
            rect = patches.Rectangle((c2 - 0.5, r2 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
        # Display the merged result
        plt.show()

def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")

for n in range(400, 2001, 200):
    print("calculating accuracy for n = ", n)
    stepwiseRegression(trainingFaces[0:n], trainingLabels[0:n], testingFaces[0:n], testingLabels[0:n])
    print(np.asarray(trainingFaces).shape)



'''def measureAccuracyOfPredictors (predictors, X, y):
    phis = []
    print(predictors)
    for p in predictors:
        phi = X[:, p[0]] - X[:, p[1]]
        phis.append(phi)
    gj = np.greater(phis, 0)
    yhat = np.greater(np.mean(gj), 0.5)
    return(fPC(y, yhat))

def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels):
    predictors = []
    images = np.transpose(trainingFaces)
    for feature in range(6):
        tuples = []
        accs = []
        for i in range(images.shape[1]):
            for j in range(images.shape[1]):
                if(j > i):
                    predictorsNew = predictors
                    predictorsNew.append((i,j))

                    acc = measureAccuracyOfPredictors(predictorsNew, trainingFaces, trainingLabels)
                    tuples.append((i,j))
                    accs.append(acc)
        
        maxIndex = np.asarray(np.argmax(accs))
        predictors.append(((tuples[maxIndex][0]), (tuples[maxIndex][1])))'''