import numpy as np

#Best error: 0.00243744217489 with hidden_Size = 4 and Syn = 6 Alpha = 55

alphas1 = range(1,64,6)
alphas2 = np.arange(0.150,1,0.10)
alphas = []
alphas.extend(alphas1)
alphas.extend(alphas2)
alphas.sort()
hiddenSizes = range(4,48,6) + range(48,128+1,16)
synNum = range(4,24,1)

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)

X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])

y = np.array([[0],
			[1],
			[1],
			[0]])

best_error = 1
best_hiddenSize = -1
best_alpha = 1


def createSyn(n,hiddenSize):
    syn = []
    np.random.seed(1)
    syn.append(2*np.random.random((3,hiddenSize))-1)
    for i in range(1,n-1):
        syn.append(2*np.random.random((hiddenSize,hiddenSize))-1)
    syn.append(2*np.random.random((hiddenSize,1))-1)
    return syn

def feedLayers(n,X,syn):
    layers = []
    layers.append(X)
    for i in range(1,n+1):
        layers.append(sigmoid(np.dot(layers[i-1],syn[i-1])))
    return layers

def getDelta(y,layers,syn,n):
    layers_err = []
    layers_delt = []
    top = len(layers) - 1
    layers_err.append(layers[top] - y)
    layers_delt.append(layers_err[0] * sigmoid_output_to_derivative(layers[top]))
    for i in range(n-1,0,-1):
        k = 0 - i + n
        layers_err.insert(0,layers_delt[0].dot(syn[i].T))
        layers_delt.insert(0,layers_err[0] * sigmoid_output_to_derivative(layers[top-k]))

    return layers_delt, layers_err, top

def updateSyn(n,layers_delt,syn,alpha):
    for i in range(0,n):
        syn[i] -= alpha * (layers[i].T.dot(layers_delt[i]))
    return syn

for syns in synNum:

    print "\nTraining With Syn: " + str(syns)

    for alpha in alphas:

        for hiddenSize in hiddenSizes:

            syn = createSyn(syns,hiddenSize)

            for j in xrange(100):

                layers = feedLayers(syns,X,syn)

                layers_delt, layers_err, top = getDelta(y,layers,syn,syns)

                if j == 99:
                    if best_error > np.mean(np.abs(layers_err[top-1])):
                        best_error = np.mean(np.abs(layers_err[top-1]))
                        best_hiddenSize = hiddenSize
                        print "Best error: "+str(best_error)+" with hidden_Size = "+str(best_hiddenSize)+" and Syn = "+str(syns)+" Alpha = "+str(alpha)

                syn = updateSyn(syns,layers_delt,syn,alpha)
