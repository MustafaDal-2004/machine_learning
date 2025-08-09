def dot_product(u, v):
    return sum([ui * vi for ui, vi in zip(u, v)])

def sign(x):
    return 1 if x > 0 else -1

def perceptron_learning_algorithm(X, y, max_iterations=1000):
    
    X = [x + [1] for x in X]

    n_samples = len(X)
    n_features = len(X[0])
    
    # Initialize weights to zero
    w = [0] * n_features

    for _ in range(max_iterations):
        error_count = 0
        for i in range(n_samples):
            if sign(dot_product(w, X[i])) != y[i]:
                # Update weights: w = w + y[i] * X[i]
                w = [wi + y[i] * xi for wi, xi in zip(w, X[i])]
                error_count += 1
        if error_count == 0:
            break

    return w

#exponetial approximation using taylor series
def exp(x, terms=10):
    result = 1.0
    numerator = 1.0
    denominator = 1.0
    for i in range(1, terms):
        numerator *= x
        denominator *= i
        result += numerator / denominator
    return result

# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

seed = 42
#random number generator
def rand():
    global seed
    seed = (seed * 9301 + 49297) % 233280
    return seed / 233280

# Initialize network
def initialize_network():
    return {
        'hidden': [
            {'weights': [rand() * 2 - 1 for _ in range(2)], 'bias': rand() * 2 - 1},
            {'weights': [rand() * 2 - 1 for _ in range(2)], 'bias': rand() * 2 - 1},
        ],
        'output': {
            'weights': [rand() * 2 - 1 for _ in range(2)],
            'bias': rand() * 2 - 1
        }
    }

# Forward pass
def forward_pass(network, inputs):
    hidden_outputs = []
    for neuron in network['hidden']:
        z = sum(w * x for w, x in zip(neuron['weights'], inputs)) + neuron['bias']
        hidden_outputs.append(sigmoid(z))

    z_out = sum(w * h for w, h in zip(network['output']['weights'], hidden_outputs)) + network['output']['bias']
    output = sigmoid(z_out)
    return hidden_outputs, output

# Training using backpropagation
def train(network, data, labels, learning_rate=0.5, epochs=10000):
    for epoch in range(epochs):
        total_loss = 0
        for inputs, expected in zip(data, labels):
            hidden_outputs, output = forward_pass(network, inputs)

            error = expected - output
            total_loss += error ** 2

            # Output layer delta
            delta_output = error * sigmoid_derivative(output)

            # Update output layer weights
            for i in range(2):
                network['output']['weights'][i] += learning_rate * delta_output * hidden_outputs[i]
            network['output']['bias'] += learning_rate * delta_output

            # Update hidden layer
            for i in range(2):
                hidden_out = hidden_outputs[i]
                delta_hidden = delta_output * network['output']['weights'][i] * sigmoid_derivative(hidden_out)

                for j in range(2):
                    network['hidden'][i]['weights'][j] += learning_rate * delta_hidden * inputs[j]
                network['hidden'][i]['bias'] += learning_rate * delta_hidden

        if epoch % 1000 == 0:
            print("Epoch", epoch, "Loss:", total_loss)

# Test the network
def test(network, data):
    for inputs in data:
        _, output = forward_pass(network, inputs)
        print(f"Input: {inputs}, Output: {round(output, 3)}")


w = initialize_network()
print(w)

data = [(0,0),(0,1),(1,0),(1,1)]
labels = (0,1,1,0)

train(w,data,labels)

print(w)



