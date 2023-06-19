from cvxopt import solvers, matrix
import numpy as np
import sklearn.svm
import matplotlib.pyplot as plt

class SVM4342 ():
    def __init__ (self):
        pass

    # Expects each *row* to be an m-dimensional row vector. X should
    # contain n rows, where n is the number of examples.
    # y should correspondingly be an n-vector of labels (-1 or +1).
    def fit (self, X, y):
        #since yi(Xi^Tw + b) >= 1, we can infer
        # -yi(Xi^Tw + b) <= -1
        # = -yi(Xtildei^T * Wtilde) <= -1
        # Because of the associative property of matrix multiplication, we can rewrite this to:
                # (-yiXtildei^T)Wtilde <= -1
                # = Gx <= h
        #w^TPw = w^Tw, therefore P = identity matrix
        X = np.append(arr = X, values = np.ones((np.shape(X)[0], 1)), axis = 1) #adds 1s vector for bias
        y = np.transpose(np.repeat([y], np.shape(X)[1], axis = 0)) #required for calculating G
       
        G = -1 * np.multiply(X, y) #fits the formula for G
        h = -1 * np.ones(shape = (np.shape(G)[0])) #each row is a constraint
        P = np.identity(n = np.shape(X)[1])
        #P = np.asarray([0]) #w^T (x^T in QP equation) has shape (m+1, 1). If it was (1, m+1), we would use np.identity(n = m+1)
        q = np.zeros(shape = (np.shape(G)[1]))

        # Solve -- if the variables above are defined correctly, you can call this as-is:
        sol = solvers.qp(matrix(P, tc='d'), matrix(q, tc='d'), matrix(G, tc='d'), matrix(h, tc='d'))

        # Fetch the learned hyperplane and bias parameters out of sol['x']
        # To avoid any annoying errors due to broadcasting issues, I recommend
        # that you flatten() the w you retrieve from the solution vector so that
        # it becomes a 1-D np.array.
        self.w = np.array(sol['x'][:-1]).flatten()
        self.b = sol['x'][-1]

    # Given a 2-D matrix of examples X, output a vector of predicted class labels
    def predict (self, x):
        val = np.asarray(x).dot(self.w) + self.b
        return(np.divide(val, np.abs(val)))

def test1 ():
    # Set up test problem
    X = np.array([ [1,1], [2,1], [1,2], [2,3], [1,4], [2,4] ])
    y = np.array([-1,-1,-1,1,1,1])

    # Train your model
    svm4342 = SVM4342()
    svm4342.fit(X, y)
    print(svm4342.w, svm4342.b)

    # Plot the data as well as the optimal separating hyperplane
    xvals = np.arange(-2, 4, 0.01)  # This is a good range of x values

    #x^T * w + b = 0. w is the slope. since we know that w has 2 values, these values are the slopes for x and y.
    #x*w0 + y*w1 + b = 0 -> y = -(xw0 + b)/w1
    yvals = -1 * ((xvals * svm4342.w[0]) + svm4342.b) / svm4342.w[1]
    plt.plot(xvals,yvals)
    plt.scatter(X[y==1,0], X[y==1,1])  # positively labeled points
    plt.scatter(X[y==-1,0], X[y==-1,1])  # negatively labeled points
    plt.show()

    # Compare with sklearn
    svm = sklearn.svm.SVC(kernel='linear', C=1e15)  # 1e15 -- approximate hard-margin
    svm.fit(X, y)
    print(svm.coef_, svm.intercept_)
    acc = np.mean(svm4342.predict(X) == svm.predict(X))
    print("Acc={}".format(acc))

def test2 (seed):
    np.random.seed(seed)

    # Generate random data
    X = np.random.rand(20,3)
    # Generate random labels based on a random "ground-truth" hyperplane
    while True:
        w = np.random.rand(3)
        y = 2*(X.dot(w) > 0.5) - 1
        # Keep generating ground-truth hyperplanes until we find one
        # that results in 2 classes
        if len(np.unique(y)) > 1:
            break

    svm4342 = SVM4342()
    svm4342.fit(X, y)

    # Compare with sklearn
    svm = sklearn.svm.SVC(kernel='linear', C=1e15)  # 1e15 -- approximate hard margin
    svm.fit(X, y)
    diff = np.linalg.norm(svm.coef_.flatten() - svm4342.w) + np.abs(svm.intercept_ - svm4342.b)
    print(diff)

    acc = np.mean(svm4342.predict(X) == svm.predict(X))
    print("Acc={}".format(acc))

    if acc == 1 and diff < 1e-1:
        print("Passed")

if __name__ == "__main__": 
    test1()
    for seed in range(5):
        test2(seed)
