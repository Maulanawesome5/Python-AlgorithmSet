import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

class Config:
    nn_input_dim = 2
    nn_output_dim = 2
    epsilon = 0.01
    reg_lambda = 0.01

def generate_data():
    np.random.seed(0)
    x, y = datasets.make_moons(200, noise=0.20)
    return x, y

def visualize(x, y, model):
    plt.scatter(x[:,0], x[:,1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()
    plot_decision_boundary(lambda x: predict(model, x), x, y)
    plt.title("Logistic Regression")

def plot_decision_boundary(pred_func, x, y):
    #Set min and max values and give it some padding
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    h = 0.01
    
    #Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h)), np.arange(y_min, y_max, h)

    #Preditc the function value for the whole grid
    z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    #Plot the contour and training example
    plt.contour(xx, yy, z, cmap=plt.cm.Spectral)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

#Helper function to evaluate the total loss on the dataset
def calculate_loss(model, x, y):
    num_example = len(x) #Training set size
    w1, b1, w2, b2 = model['w1'], model['b1'], model['w2'], model['b2']
    
    #Forward propagation to calculate our predictions
    z1 = x.dot(w1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(w2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    #Calculating the loss
    corect_logprobs = -np.log(probs[range(num_example), y])
    data_loss = np.sum(corect_logprobs)

    #Add regulatization term to loss (optional)
    data_loss += Config.reg_lambda / 2 * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    return 1. / num_example * data_loss

def predict(model, x):
    w1, b1, w2, b2 = model['w1'], model['b1'], model['w2'], model['b2']
    #Forward propagation
    z1 = x.dot(w1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(w2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

# This function learns parameter for the neural network and returns the model
# - nn_hdim = Number of nodes in the hidden layer
# - num_passes = Number of passes through the training data for gradient descent
# - print_loss = if true, print the loss every 1000 iterations
def build_model(x, y, nn_hdim, num_passes = 20000, print_loss = False):
    #Initialize the parameter to random values. We need to learn these.
    num_example = len(x)
    np.random.seed(0)
    w1 = np.random.randn(Config.nn_input_dim, nn_hdim) / np.sqrt(Config.nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    w2 = np.random.randn(nn_hdim, Config.nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, Config.nn_output_dim))
    
    #This is what we return at the end
    model = {}

    #Gradient descent. For each batch
    for i in range(0, num_passes):
        #Forward propagation
        z1 = x.dot(w1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(w2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        #Backpropagation
        delta3 = probs
        delta3[range(num_example), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(w2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(x.T, delta2)
        db1 = np.sum(delta2, axis=0)

        #Add Regularizations terms (b1 and b2 don't have regularization terms)
        dW2 += Config.reg_lambda * w2
        dW1 += Config.reg_lambda * w1

        #Gradient descent parameter update
        w1 += -Config.epsilon * dW1
        b1 += -Config.epsilon * db1
        w2 += -Config.epsilon * dW2
        b2 += -Config.epsilon * db2

        #Assign new parameters to the model
        model = {'W1' : w1, 'b1': b1, 'W2' : w2, 'b2' : b2}

        #Optionally print the loss.
        #This is expensive because it uses the whole dataset,
        #so we don't want to do ittoo often
        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i : %f" % (i, calculate_loss(model, x, y)))
    return model

def classify(x, y):
    clf = linear_model.LogisticRegressionCV()
    clf.fit(x, y)
    return clf
    pass

def main():
    x, y = generate_data()
    model = build_model(x, y, 3, print_loss = True)
    visualize(x, y, model)

if __name__ == "__main__":
    main

"""
NB : Mungkin terdapat error pada syntax dikarenakan penulis
buku menggunakan python versi 2.7.9 saat menulis bukunya.
"""