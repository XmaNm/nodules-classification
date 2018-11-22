import utils,os
from SDA_layers import StackedDA
import numpy as np

d = os.path.dirname(__file__)
parent_path = os.path.dirname(d)

def demo():
    # X, y = utils.load_mnist()
    X = np.load(parent_path + "/data_aug.npy")
    X = np.resize(X,[X.shape[0],X.shape[1]*X.shape[2]])
    y = np.load(parent_path + "/label_aug.npy")
    y = y-1

    indices = np.random.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]

    y = utils.makeMultiClass(y)

    # #
    # # building the SDA
    sDA = StackedDA([100])
    # #
    # # pre-trainning the SDA
    sDA.pre_train(X[:100], noise_rate=0.9, epochs=1)
    # saving a PNG representation of the first layer
    # W = sDA.Layers[0].W.T[:, 1:]
    # utils.saveTiles(W, img_shape=(32, 32), tile_shape=(10, 10), filename="results/res_dA.png")

    # adding the final layer
    sDA.finalLayer(X[:18000], y[:18000], epochs=1)

    # trainning the whole network
    sDA.fine_tune(X[:18000], y[:18000], epochs=1)

    # predicting using the SDA
    pred = sDA.predict(X[18000:]).argmax(1)

    y = y[18000:].argmax(1)

    e = 0.0
    sum = 0
    sumr = 0
    for i in range(len(y)):
        e += abs(y[i] - pred[i]) <=2
        sum += abs(y[i]-pred[i])
        sumr += abs(pred[i]-round(y[i]))
        with open("result.txt", "a+") as f:
            f.write(str(pred[i]))
            f.write(",")
            f.write(str(y[i]))
            f.write("\n")
    # printing the result, this structure should result in 80% accuracy
    print "accuracy: %2.2f%%" % (100 * e / len(y))
    # print pred.tolist()

    return sDA

if __name__ == '__main__':

    demo()