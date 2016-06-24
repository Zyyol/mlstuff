import numpy as np
#Best error: 0.000613545258393 with hidden_Size = 36 -- Alpha : 19
#Best error: 0.0277984212559 -- Alpha : 25 -- Hidden size : 22 /// 100 loop
#Best error: 0.00491980788459 -- Alpha : 25 -- Hidden size : 22 /// 1k loop


alphas1 = range(1,64,2)
alphas2 = np.arange(0.001,1,0.01)
alphas = []
alphas.extend(alphas1)
alphas.extend(alphas2)
alphas.sort()
hiddenSizes = range(4,48,2) + range(48,1024+1,6)


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
best_error = 0.5

for alpha in alphas:

    print "\nTraining With Alpha:" + str(alpha)

    for hiddenSize in hiddenSizes:

        #print "\nTraining with hidden size:" + str(hiddenSize)

        np.random.seed(1)
        # randomly initialize our weights with mean 0
        synapse_0 = 2*np.random.random((3,hiddenSize)) - 1
        synapse_1 = 2*np.random.random((hiddenSize,1)) - 1


        for j in xrange(1000):

            # Feed forward through layers 0, 1, and 2
            layer_0 = X
            layer_1 = sigmoid(np.dot(layer_0,synapse_0))
            layer_2 = sigmoid(np.dot(layer_1,synapse_1))

            # how much did we miss the target value?
            layer_2_error = layer_2 - y

            if j == 999:
                #print "Error after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error)))
                if best_error > np.mean(np.abs(layer_2_error)):
                    best_error = np.mean(np.abs(layer_2_error))
                    best_alpha = alpha
                    best_hiddenSize = hiddenSize
                    print "Best error: "+ str(best_error)+" with hidden_Size = "+ str(best_hiddenSize)


            # in what direction is the target value?
            # were we really sure? if so, don't change too much.
            layer_2_delta = layer_2_error*sigmoid_output_to_derivative(layer_2)

            # how much did each l1 value contribute to the l2 error (according to the weights)?
            layer_1_error = layer_2_delta.dot(synapse_1.T)

            # in what direction is the target l1?
            # were we really sure? if so, don't change too much.
            layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

            synapse_1 -= alpha * (layer_1.T.dot(layer_2_delta))
            synapse_0 -= alpha * (layer_0.T.dot(layer_1_delta))


print "\nBest error: "+ str(best_error)+" -- Alpha : "+ str(best_alpha)+" -- Hidden size : "+ str(best_hiddenSize)
