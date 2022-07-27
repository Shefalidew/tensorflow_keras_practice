import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from tensorflow import keras

tf.disable_v2_behaviour()


def load_mnist(path, kind):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

    with open(labels_path,'rb') as lbpath:
        magic, n =struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open (images_path, 'rb') as imgpath:
        magic , num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images =np.fromfile(imgpath, dtype =np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) *2
    return images , labels


x_train, y_train = load_mnist('/home/student/PycharmProjects/pythonProject2/mnist', kind ='train')
print('Rows: %d, columns:%d' %(x_train.shape[0],x_train.shape[1]))

x_test , y_test = load_mnist('/home/student/PycharmProjects/pythonProject2/mnist', kind ='t10k')
print('Rows: %d, columns:%d' %(x_test.shape[0],x_test.shape[1]))

##mnist_dataset = keras.datasets.mnist.load_data(path="mnist.npz")

##(x_train, y_train), (x_test, y_test) = mnist_dataset
##print('Rows: %d, Columns:%d' % (x_train.shape[0], x_train.shape[1]))
##print('Rows: %d, Columns: %d' % (x_test.shape[0], x_test.shape[1]))

##mean centering and normalization
mean_vals = np.mean(x_train, axis=0)
std_val = np.std(x_train)

x_train_centered = (x_train - mean_vals) / std_val
x_test_centered = (x_test - mean_vals) / std_val

del x_train, x_test

print(x_train_centered.shape, y_train.shape)
print(x_test_centered.shape, y_test.shape)

np.random.seed(123)
tf.set_random_seed(123)

##converting the class labels to one hot format
y_train_onehot = keras.utils.to_categorical(y_train)
print('First 3 labels: ', y_train[:3])
print('\n First 3 labels (one-hot) : \n', y_train_onehot[:3])

##implementing 3 layers
model = keras.models.Sequential()

model.add(
    keras.layers.Dense(
        units=50,
        input_dim=x_train_centered.shape[1],
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='tanh'))

model.add(
    keras.layers.Dense(
        units=50,
        input_dim=50,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='tanh'))

model.add(
    keras.layers.Dense(
        units=y_train_onehot.shape[1],
        input_dim=50,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='softmax'))

sgd_optimizer = keras.optimizers.SGD(
    learning_rate=0.001, decay=1e-7, momentum=.9)

model.compile(optimizer=sgd_optimizer,
              loss='categorical_crossentropy')

history = model.fit(x_train_centered, y_train_onehot,
                    batch_size=64, epochs=50,
                    verbose=1,
                    validation_split=0.1)
predict_x=model.predict(x_train_centered)
y_train_predict=np.argmax(predict_x,axis=1)
##y_train_predict = model.predict_classes(x_train_centered,verbose=0)
correct_preds = np.sum(y_train == y_train_predict, axis =0)
train_acc =correct_preds/y_train.shape[0]
print('First 3 predictions:', y_train_predict[:3])

print('Training accuracy: %.2f%%' %(train_acc*100))

predict_x1=model.predict(x_test_centered)
y_test_predict=np.argmax(predict_x1,axis=1)
correct_preds = np.sum(y_test == y_test_predict, axis =0)
test_acc =correct_preds /y_test.shape[0]
print('Test accuracy:%.2f%%' %(test_acc*100))
