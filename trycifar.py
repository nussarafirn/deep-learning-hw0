from uwnet import *

def softmax_model():
    l = [make_connected_layer(3072, 10, SOFTMAX)]
    return make_net(l)

def neural_net():
    l = [   make_connected_layer(3072, 350, LRELU),
            make_connected_layer(350, 350, RELU),
            make_connected_layer(350, 10, SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test", "cifar/cifar.labels")
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

