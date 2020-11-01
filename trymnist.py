from uwnet import *

def softmax_model():
    l = [make_connected_layer(784, 10, SOFTMAX)]
    return make_net(l)

def neural_net():
    l = [   make_connected_layer(784, 300, LRELU),
            make_connected_layer(300, 300, RELU),
            make_connected_layer(300, 100, RELU),
            make_connected_layer(100, 10, SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("mnist/mnist.train", "mnist/mnist.labels")
test  = load_image_classification_data("mnist/mnist.test", "mnist/mnist.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .0

m = neural_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

