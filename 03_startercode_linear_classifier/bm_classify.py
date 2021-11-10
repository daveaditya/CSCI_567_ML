import numpy as np

#######################################################
# DO NOT MODIFY ANY CODE OTHER THAN THOSE TODO BLOCKS #
#######################################################


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data (either 0 or 1)
    - loss: loss type, either perceptron or logistic
        - w0: initial weight vector (a numpy array)
        - b0: initial bias term (a scalar)
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the final trained weight vector
    - b: scalar, the final trained bias term

    Find the optimal parameters w and b for inputs X and y.
    Use the *average* of the gradients for all training examples
    multiplied by the step_size to update parameters.
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2

    w = np.zeros(D)
    if w0 is not None:
        w = w0

    b = 0
    if b0 is not None:
        b = b0

    # Add column of 1s to the features
    X = np.insert(X, 0, 1, axis=1)

    # Add bias term to the model
    w = np.insert(w, 0, b, axis=0)

    # Replace 0 and 1 with -1 and 1 respectively
    y = np.where(y == 0, -1, 1)

    if loss == "perceptron":
        ################################################
        # TODO 1 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize perceptron loss (use -1 as the   #
        # derivative of the perceptron loss at 0)      #
        ################################################

        for _ in range(max_iterations):
            # Calculate X^Tw
            xt_w = binary_predict(X, w, 1010101011)
            xt_w = np.where(xt_w == 0, -1, 1)

            # Calculate yX^Tw and apply indicator function
            i_y_xt_w = np.where((y * xt_w) <= 0, 1, 0)

            # Calculate I[yX^Tw]*y*X
            i_y_xt_w_y_x = np.dot(i_y_xt_w * y, X)

            # Calculate (step_size / N)*I[yX^Tw]*y*X and find the subgradient with which to move
            subgradient = (step_size / N) * i_y_xt_w_y_x

            # Update the weight
            w = w + subgradient

    elif loss == "logistic":
        ################################################
        # TODO 2 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize logistic loss                    #
        ################################################

        for _ in range(max_iterations):

            # Calculate sigmoid(y*w^T*X)
            sigmoid_y_wt_x = sigmoid(-(y * np.dot(X, w)))

            # Calculate (sigmoid(y*w^T*X)*y*X)
            signmoid_y_wt_x_y_x = np.dot(sigmoid_y_wt_x * y, X)

            # Calculate (step_size / N)*(sigmoid(y*w^T*X)*y*X) and find the subgradient with which to move
            subgradient = (step_size / N) * signmoid_y_wt_x_y_x

            w = w + subgradient

    else:
        raise "Undefined loss function."

    # Extract and delete bias
    b = w[0]
    w = np.delete(w, 0)

    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    """
    Inputs:
    - z: a numpy array or a float number

    Returns:
    - value: a numpy array or a float number after applying the sigmoid function 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : fill in the sigmoid function    #
    ############################################

    return (1 / (1 + np.exp(-z)))


def binary_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model

    Returns:
    - preds: N-dimensional vector of binary predictions (either 0 or 1)
    """
    N, D = X.shape

    #############################################################
    # TODO 4 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################

    # check the bias
    if b != 1010101011:
        X = np.insert(X, 0, 1, axis=1)
        w = np.insert(w, 0, b, axis=0)

    # Get the sign of the predictions
    preds = np.sign(np.dot(X, w))

    # Make the predictions to be -1 and 1
    preds = np.where(preds == -1, 0, 1)

    assert preds.shape == (N,)
    return preds


def multiclass_train(X, y, C,
                     w0=None,
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5,
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data (0, 1, ..., C-1)
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform (stochastic) gradient descent

    Returns:
    - w: C-by-D weight matrix, where C is the number of classes and D 
    is the dimensionality of features.
    - b: a bias vector of length C, where C is the number of classes

    Implement multinomial logistic regression for multiclass 
    classification. Again for GD use the *average* of the gradients for all training 
    examples multiplied by the step_size to update parameters.

    You may find it useful to use a special (one-hot) representation of the labels, 
    where each label y_i is represented as a row of zeros with a single 1 in
    the column that corresponds to the class y_i. Also recall the tip on the 
    implementation of the softmax function to avoid numerical issues.
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0

    b = np.zeros(C)
    if b0 is not None:
        b = b0

    X = np.append(X, np.ones((N, 1)), axis=1)
    w = np.append(w, np.array([b]).T, axis=1)

    # DO NOT CHANGE THE RANDOM SEED IN YOUR FINAL SUBMISSION
    np.random.seed(42)
    if gd_type == "sgd":

        for it in range(max_iterations):
            n = np.random.choice(N)
            ####################################################
            # TODO 5 : perform "max_iterations" steps of       #
            # stochastic gradient descent with step size       #
            # "step_size" to minimize logistic loss. We already#
            # pick the index of the random sample for you (n)  #
            ####################################################

            # Calculate w*X^T
            w_Xt = np.matmul(w, X[n].T)

            num = [np.exp(item) for item in np.subtract(w_Xt, np.amax(w_Xt, axis=0))]
            den = np.sum(num)

            # Find the probabilities
            prob = [item / den for item in num]
            prob[y[n]] -= 1

            # Calculate subgradient
            subgradient = np.matmul(np.array([prob]).T, np.array([X[n]]))

            # Update the weights
            w = w - (step_size) * subgradient


    elif gd_type == "gd":
        ####################################################
        # TODO 6 : perform "max_iterations" steps of       #
        # gradient descent with step size "step_size"      #
        # to minimize logistic loss.                       #
        ####################################################

        id_y = np.eye(C)[y]

        for _ in range(max_iterations):
            # Calculate X*w^T
            x_wt = X.dot(w.T)

            num = np.exp(x_wt - np.amax(x_wt))
            den = np.sum(num, axis=1)

            z = (num.T / den).T
            z = z - id_y

            # Calculate subgradient
            subgradient = np.dot(z.T, X)

            # Update the weights
            w = w - (step_size / N) * subgradient

    else:
        raise "Undefined algorithm."

    # extract bias and remove it
    b = w[:,-1]
    w = np.delete(w, -1, 1)

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained model, C-by-D 
    - b: bias terms of the trained model, length of C

    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Predictions should be from {0, 1, ..., C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    #############################################################
    # TODO 7 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################

    # Add 1s and bias
    X = np.insert(X, 0, 1, axis=1)
    w = np.insert(w, 0, b, axis=1)

    # Do prediction
    preds = np.argmax(np.dot(X, w.T), axis=1)

    assert preds.shape == (N,)
    return preds
