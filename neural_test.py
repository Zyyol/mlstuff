import numpy as np
#Training With Alpha:22
#Best error: 0.000287974257015 with hidden_Size = 8

alphas1 = range(1,64,6)
alphas2 = np.arange(0.150,1,0.10)
alphas = []
alphas.extend(alphas1)
alphas.extend(alphas2)
alphas.sort()
hiddenSizes = range(4,48,6) + range(48,256+1,12)
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
best_alpha = 1

for alpha in alphas:

    print "\nTraining With Alpha:" + str(alpha)

    for hiddenSize in hiddenSizes:

        #print "\nTraining with hidden size:" + str(hiddenSize)

        np.random.seed(1)
        # randomly initialize our weights with mean 0
        synapse_0 = 2*np.random.random((3,hiddenSize)) - 1
        synapse_1 = 2*np.random.random((hiddenSize,hiddenSize)) - 1
        synapse_2 = 2*np.random.random((hiddenSize,hiddenSize)) - 1
        synapse_3 = 2*np.random.random((hiddenSize,1)) - 1


        for j in xrange(60000):

            # Feed forward through layers 0, 1, and 2
            layer_0 = X
            layer_1 = sigmoid(np.dot(layer_0,synapse_0))
            layer_2 = sigmoid(np.dot(layer_1,synapse_1))
            layer_3 = sigmoid(np.dot(layer_2,synapse_2))
            layer_4 = sigmoid(np.dot(layer_3,synapse_3))


            # how much did we miss the target value?
            layer_4_error = layer_4 - y

            if j == 59999:
                #print "Error after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error)))
                if best_error > np.mean(np.abs(layer_4_error)):
                    best_error = np.mean(np.abs(layer_4_error))
                    best_alpha = alpha
                    best_hiddenSize = hiddenSize
                    print "Best error: "+ str(best_error)+" with hidden_Size = "+ str(best_hiddenSize)


            layer_4_delta = layer_4_error*sigmoid_output_to_derivative(layer_4)



            layer_3_error = layer_4_delta.dot(synapse_3.T)

            layer_3_delta = layer_3_error * sigmoid_output_to_derivative(layer_3)



            layer_2_error = layer_3_delta.dot(synapse_2.T)

            layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

            layer_1_error = layer_2_delta.dot(synapse_1.T)

            layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

            synapse_3 -= alpha * (layer_3.T.dot(layer_4_delta))
            synapse_2 -= alpha * (layer_2.T.dot(layer_3_delta))
            synapse_1 -= alpha * (layer_1.T.dot(layer_2_delta))
            synapse_0 -= alpha * (layer_0.T.dot(layer_1_delta))


print "\nBest error: "+ str(best_error)+" -- Alpha : "+ str(best_alpha)+" -- Hidden size : "+ str(best_hiddenSize)
