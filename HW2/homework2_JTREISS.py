import numpy as np
import matplotlib.pyplot as plt  # to show images

# Given an array of faces (N x M x M, where N is number of examples and M is number of pixes along each axis),
# return a design matrix Xtilde ((M**2 + 1) x N) whose last row contains all 1s.
def reshapeAndAppend1s (faces):
    shape = np.shape(faces)
    designMatrix = np.transpose(faces)
    designMatrix = np.reshape(designMatrix, (shape[1] * shape[2], shape[0]), 'F') #turns the 3d matrix into a 2d matrix that fits the requirements

    appending1s = np.ones(shape = (1, shape[0]))
    designMatrix = np.append(arr = designMatrix, values = appending1s, axis = 0) #adds 1s to the end of the matrix to get Xtilde

    return designMatrix

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, return the (unregularized)
# MSE.
def fMSE (wtilde, Xtilde, y):
    yhat = np.transpose(Xtilde).dot(wtilde)

    return 0.5 * np.mean(np.square(yhat - y))

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, and a regularization strength
# alpha (default value of 0), return the gradient of the (regularized) MSE loss.
def gradfMSE (wtilde, Xtilde, y, alpha = 0.):
    yhat = np.transpose(Xtilde).dot(wtilde)
    yPenalty = np.asarray(Xtilde).dot(yhat - y) #calculates the gradient without the regularization
    wPenalty = (alpha * wtilde) #calculates the gradient of the regularization
    wPenalty[-1] = 0 #excludes bias

    return (yPenalty + wPenalty)/np.shape(Xtilde)[1] #adds the two penalties together and divides by n


# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using the analytical solution.
def method1 (Xtilde, y):
    xT = np.asarray(Xtilde).dot(np.transpose(Xtilde)) #squares Xtilde
    return np.linalg.solve(xT, (np.asarray(Xtilde).dot(y))) #performs (Xtilde*Xtilde)^-1 * Xtilde*y

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE.
def method2 (Xtilde, y):
    return gradientDescent(Xtilde, y)

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE
# with regularization.
def method3 (Xtilde, y):
    ALPHA = 0.1
    return gradientDescent(Xtilde, y, ALPHA)

# Helper method for method2 and method3.
def gradientDescent (Xtilde, y, alpha = 0.):
    EPSILON = 3e-3  # Step size aka learning rate
    T = 5000  # Number of gradient descent iterations

    w = 0.01 * np.random.randn(np.shape(Xtilde)[0]) #initializes w with random values

    for i in range(T): #iterates to change w
        gradWfMSE = gradfMSE(w, Xtilde, y, alpha)
        w = w - (EPSILON * gradWfMSE)
    return w

if __name__ == "__main__":
    # Load data
    Xtilde_tr = reshapeAndAppend1s(np.load("age_regression_Xtr.npy"))
    ytr = np.load("age_regression_ytr.npy")
    Xtilde_te = reshapeAndAppend1s(np.load("age_regression_Xte.npy"))
    yte = np.load("age_regression_yte.npy")

    w1 = method1(Xtilde_tr, ytr)
    w2 = method2(Xtilde_tr, ytr)
    w3 = method3(Xtilde_tr, ytr)

    # Report fMSE cost using each of the three learned weight vectors
    #part A
    print("Analytical solution training MSE:", fMSE(w1, Xtilde_tr, ytr))
    print("Analytical solution testing MSE:", fMSE(w1, Xtilde_te, yte))

    #part B
    print("Gradient Descent solution training MSE:", fMSE(w2, Xtilde_tr, ytr))
    print("Gradient Descent solution testing MSE:", fMSE(w2, Xtilde_te, yte))

    #part C
    print("Gradient Descent Regularized solution training MSE:", fMSE(w3, Xtilde_tr, ytr))
    print("Gradient Descent Regularized solution testing MSE:", fMSE(w3, Xtilde_te, yte))

    #Part D
    plt.title(label = 'w1 image')
    plt.imshow(np.asarray(w1[:-1]).reshape(48,48), cmap = 'gray')
    plt.show()

    plt.title(label = 'w2 image')
    plt.imshow(np.asarray(w2[:-1]).reshape(48,48), cmap = 'gray')
    plt.show()

    plt.title(label = 'w3 image')
    plt.imshow(np.asarray(w3[:-1]).reshape(48,48), cmap = 'gray')
    plt.show()

    dYHat = np.transpose(Xtilde_tr).dot(w3)
    errors = np.absolute(dYHat - ytr)

    fiveWorst = [-1,-1,-1,-1,-1]
    for j in range(len(errors)):
        if (errors[j] > fiveWorst[0]):
            fiveWorst[0] = j
        
        elif (errors[j] > errors[fiveWorst[1]]) and (errors[j] <= errors[fiveWorst[0]]):
            fiveWorst[1] = j
        
        elif (errors[j] > errors[fiveWorst[2]]) and (errors[j] <= errors[fiveWorst[1]]):
            fiveWorst[2] = j
        
        elif (errors[j] > errors[fiveWorst[3]]) and (errors[j] <= errors[fiveWorst[2]]):
            fiveWorst[3] = j
        
        elif (errors[j] > errors[fiveWorst[4]]) and (errors[j] <= errors[fiveWorst[3]]):
            fiveWorst[4] = j
    
    for i in range(5):
        index = fiveWorst[i]
        img = np.transpose(Xtilde_tr)[index][:-1]
        title = str(i + 1) + ": y = " + str(ytr[index]) + "  yhat = " + str(dYHat[index]) + "  error = " + str(errors[index])
        plt.title(title)
        plt.imshow(np.asarray(img).reshape(48,48))
        plt.show()