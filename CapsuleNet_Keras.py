from keras import layers, models, optimizers, regularizers, constraints
from keras import backend as K
from capsulelayer_keras import Class_Capsule, Conv_Capsule, PrimaryCap, Length
from data_prepare import readdata
import time


def CapsNet(input_shape, n_class, num_routing):

    x = layers.Input(shape=input_shape)

    conv1 = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='valid', name='conv1')(x)
    conv1 = layers.BatchNormalization(momentum=0.9, name='bn1')(conv1)
    conv1 = layers.Activation('relu', name='conv1_relu')(conv1)

    conv2 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', name='conv2')(conv1)
    conv2 = layers.BatchNormalization(momentum=0.9, name='bn2')(conv2)
    conv2 = layers.Activation('relu', name='conv2_relu')(conv2)

    primarycaps = PrimaryCap(conv2, dim_vector=8, n_channels=4, kernel_size=4, strides=2, padding='valid')

    Conv_caps1 = Conv_Capsule(kernel_shape=[3, 3, 4, 8], dim_vector=8, strides=[1, 2, 2, 1],
                              num_routing=num_routing, batchsize=args.batch_size, name='Conv_caps1')(primarycaps)

    digitcaps = Class_Capsule(num_capsule=n_class, dim_vector=16, num_routing=num_routing, name='digitcaps')(Conv_caps1)

    out_caps = Length(name='out_caps')(digitcaps)

    return models.Model(x, out_caps)


def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, data, args):

    (x_train, y_train), (x_valid, y_valid) = data

    # callbacks
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size)
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-test.h5',
                                           save_best_only=True, save_weights_only=True, verbose=1)

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss],
                  metrics={'out_caps': 'accuracy'})

    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[x_valid, y_valid], callbacks=[tb, checkpoint], verbose=2)

    return model


def test(model, data):
    from sklearn.metrics import confusion_matrix
    x_test, y_test = data[0], data[1]
    n_samples = y_test.shape[0]
    add_samples = args.batch_size - n_samples % args.batch_size
    x_test = np.concatenate((x_test[0:add_samples, :, :, :], x_test), axis=0)
    y_test = np.concatenate((y_test[0:add_samples, :], y_test), axis=0)
    y_pred = model.predict(x_test, batch_size=args.batch_size)
    ypred = np.argmax(y_pred, 1)
    y = np.argmax(y_test, 1)
    matrix = confusion_matrix(y[add_samples:], ypred[add_samples:])
    return matrix


def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


if __name__ == "__main__":
    import numpy as np
    import os
    from keras import callbacks

    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--n_class', default=13, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--num_routing', default=3, type=int)  # num_routing should > 0
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--is_training', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--windowsize', default=27, type=int)
    args = parser.parse_args()

    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # file path of HSI dataset
    image_file = r'E:\KSC.mat'
    label_file = r'E:\KSC_gt.mat'

    data, test_shuffle_number = readdata(image_file, label_file, train_nsamples=200, validation_nsamples=100,
                                         windowsize=args.windowsize, istraining=True)

    (x_train, y_train), (x_valid, y_valid) = (data[0], data[1]), (data[2], data[3])

    # define model
    model = CapsNet(input_shape=[args.windowsize, args.windowsize, 108],
                    n_class=args.n_class,
                    num_routing=args.num_routing)
    model.summary()

    # training
    start = time.time()
    train(model=model, data=((x_train, y_train), (x_valid, y_valid)), args=args)
    end = time.time()
    print('train time:', end - start)

    # test
    start = time.time()
    model.load_weights('./result/weights-test.h5')
    i = 0
    test_nsamples = 0
    matrix = np.zeros([args.n_class, args.n_class], dtype=np.float32)
    while 1:
        data = readdata(image_file, label_file, train_nsamples=200, validation_nsamples=100,
                        windowsize=args.windowsize, istraining=False, shuffle_number=test_shuffle_number, times=i)
        if data == None:
            OA, AA_mean, Kappa, AA = cal_results(matrix)
            print('-' * 50)
            print('OA:', OA)
            print('AA:', AA_mean)
            print('Kappa:', Kappa)
            print('Classwise_acc:', AA)
            end = time.time()
            print('test time:', end - start)
            break
        test_nsamples += data[0].shape[0]
        matrix = matrix + test(model=model, data=(data[0], data[1]))
        i = i + 1
