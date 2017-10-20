import shelve
import pickle

import numpy as np
import math
import time

import keras
import keras.metrics as M
import keras.backend as K

from dataset import get_mnist_dataset, get_mnist_dataset_part, get_srcnn_mnist_dataset, get_srcnn_mnist_dataset_part, \
    get_srcnn_rgb_mnist_dataset, get_srcnn_rgb_mnist_dataset_part, get_srcnn_rgb_cifar10_dataset, \
    get_srcnn_rgb_cifar10_dataset_part, get_srcnn_rgb_cifar10_20000_dataset, get_srcnn_rgb_cifar10_20000_dataset_part, \
    get_pasadena_dataset, get_pasadena_dataset_part, get_hundred_dataset, get_hundred_dataset_part, \
    get_100_86_dataset, get_100_86_dataset_part, get_dataset_part
import metrics


def print_result(train_result, test_result):
    print()
    print('test_result :', test_result)
    print('train_result :', 'epochs :', train_result.epoch, 'history :', train_result.history)


def print_train_result(train_result):
    print('train_result :', 'epochs :', train_result.epoch, 'history :', train_result.history)


def print_test_result(test_result):
    print()
    print('test_result :', test_result)


def print_shape(name, train, test):
    if len(train.shape) == 3:
        num_train, height_train, width_train = map(str, train.shape)
        num_test, height_test, width_test = map(str, test.shape)
        depth_train = depth_test = 1
    elif len(train.shape) == 4:
        num_train, height_train, width_train, depth_train = map(str, train.shape)
        num_test, height_test, width_test, depth_test = map(str, test.shape)

    print(name + '_train : [ num : ' + num_train, 'height : ' + height_train, 'width : ' + width_train,
          'depth : ' + depth_train + ' ]',
          name + '_test : [ num : ' + num_test, 'height : ' + height_test, 'width : ' + width_test,
          'depth : ' + depth_test + ' ]', sep=', ')


def plot_results(train_result, test_result, dataset_name, show=False):
    print('plotting train and test results...')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    epoch, history = train_result.epoch, train_result.history
    epoch = [i + 1 for i in epoch]

    # subplots = [221, 222, 223]
    history_keys = sorted(history.keys())
    history_keys.remove('loss')
    test_result_dict = {'loss': test_result[0]}
    test_result_dict.update(zip(history_keys, test_result[1:]))

    titles = {'loss': 'Потери', 'acc': 'Точность', 'mean_squared_error': 'Среднеквадратичная ошибка',
              'psnr_3_channels': 'PSNR'}

    plt.title('Train and test results')
    # ro, r--, bs, g^, -, -., :
    plt.figure(1, figsize=(16.0, 10.0))
    # ax = plt.figure(1).gca()  # plt.figure(1)
    for i, (metric, values) in enumerate(history.items()):
        # plt.subplot(subplots[i])
        plt.subplot(221 + i)
        plt.title(titles[metric])
        plt.xlabel('Итерация')  # 'Epoches'
        plt.ylabel('Величина')  # 'Values'
        # train result
        plt.plot(epoch, values, 'r')
        min_value, max_value, test_result_item = min(values), max(values), test_result_dict[metric]
        dy = 0
        # max train result
        plt.plot(epoch, [max_value] * len(epoch), 'r:')
        plt.text(epoch[-2], max_value + dy, str(round(max_value, 4)))
        # min train result
        plt.plot(epoch, [min_value] * len(epoch), 'r:')
        plt.text(epoch[-2], min_value + dy, str(round(min_value, 4)))
        # test result
        plt.plot(epoch, [test_result_item] * len(epoch), 'b')
        plt.text(epoch[0], test_result_item + dy, str(round(test_result_item, 4)))

    plt.subplots_adjust(top=0.9, bottom=0.10, left=0.15, right=0.90, hspace=0.5,
                        wspace=0.4)
    if show:
        plt.show()
    print('saving results/' + dataset_name + '/plots/train-and-test-results.png')
    plt.savefig('results/' + dataset_name + '/plots/train-and-test-results.png')
    print('saving results/' + dataset_name + '/plots/train-and-test-results.svg')
    plt.savefig('results/' + dataset_name + '/plots/train-and-test-results.svg')


def plot_psnr_rgb_results(train_result, test_result, dataset_name, show=False):
    print('plotting psrn_3_channels train and test results...')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    epoch, history = train_result.epoch, train_result.history
    epoch = [i + 1 for i in epoch]

    # subplots = [221, 222, 223]
    history_keys = sorted(history.keys())
    history_keys.remove('loss')
    test_result_dict = {'loss': test_result[0]}
    test_result_dict.update(zip(history_keys, test_result[1:]))

    metric = 'psnr_3_channels'
    values = history[metric]

    plt.title('Пиковое отношение сигнала к шуму')
    # ro, r--, bs, g^, -, -., :
    plt.figure(2, figsize=(7.0, 4.0))  # 10 6
    # ax = plt.figure(1).gca()  # plt.figure(1)

    # plt.subplot(subplots[i])
    # plt.subplot(221)
    plt.title('Пиковое отношение сигнала к шуму')
    plt.xlabel('Итерации')
    plt.ylabel('Среднее PSNR')
    # train result
    plt.plot(epoch, values, 'r')
    min_value, max_value, test_result_item = min(values), max(values), test_result_dict[metric]
    dy = 0
    # max train result
    plt.plot(epoch, [max_value] * len(epoch), 'r:')
    plt.text(epoch[-2], max_value + dy, str(round(max_value, 4)))
    # min train result
    plt.plot(epoch, [min_value] * len(epoch), 'r:')
    plt.text(epoch[-2], min_value + dy, str(round(min_value, 4)))
    # test result
    plt.plot(epoch, [test_result_item] * len(epoch), 'b')
    plt.text(epoch[0], test_result_item + dy, str(round(test_result_item, 4)))

    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.10, right=0.90, hspace=0.5,
                        wspace=0.4)
    if show:
        plt.show()
    path = '/plots/psnr_3_channels'
    print('saving results/' + dataset_name + path + '.png')
    plt.savefig('results/' + dataset_name + path + '.png')
    print('saving results/' + dataset_name + path + '.svg')
    plt.savefig('results/' + dataset_name + path + '.svg')


def psnr_RGB(y_true, y_pred):
    # Peak signal-to-noise ratio
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    # NORMAL VALUES : [30, 40]
    print('PSNR (3 channels) metric running...')

    maxf = 1.0  # K.max(y_true)
    mse = keras.metrics.mean_squared_error(y_true, y_pred)
    eps = K.constant(1e-6)  # K.epsilon()
    arg = maxf / K.sqrt(mse + eps)  # + K.epsilon()
    ten = K.constant(10)
    return 20 * K.log(arg) / K.log(ten)


def custom_relu(x):
    return keras.activations.relu(x, max_value=1.0)


def normalise_ndarrrays(ndarrays):
    result = []
    for data in ndarrays:
        result.append(data / np.max(data))
    return np.array(result)


def fit():
    print('SRCNN tf running...')

    from keras.layers import Input, Conv2D
    from keras.models import Model
    from keras.regularizers import l2  # L2-regularisation

    use_rgb_dataset = True
    depth = c = 3
    (X_train, Y_train), (X_test, Y_test) = get_srcnn_rgb_cifar10_20000_dataset_part(train_part=1, test_part=1)
    dataset_name = 'CIFAR-10 20000 tf'

    num_X_train, height_X_train, width_X_train, _ = X_train.shape
    num_X_test, height_X_test, width_X_test, _ = X_test.shape
    print_shape('X', X_train, X_test)

    num_Y_train, height_Y_train, width_Y_train, _ = Y_train.shape
    num_Y_test, height_Y_test, width_Y_test, _ = Y_test.shape
    print_shape('Y', Y_train, Y_test)

    dataset_type = 'float32'

    X_train = X_train.astype(dataset_type)
    X_test = X_test.astype(dataset_type)
    Y_train = Y_train.astype(dataset_type)
    Y_test = Y_test.astype(dataset_type)

    X_train /= 255  # Normalise data to [0, 1] range
    X_test /= 255  # Normalise data ti [0, 1] range
    Y_train /= 255  # Normalise data to [0, 1] range
    Y_test /= 255  # Normalise data to [0, 1] range

    # making PSNR metric
    psnr_3_callback = metrics.MetricsCallbackPSNR(X=(X_train, X_test), Y=(Y_train, Y_test),
                                                  batch_size=batch_size)
    min_max_callback = metrics.MetricsCallbackMinMax(X=(X_train, X_test), Y=(Y_train, Y_test),
                                                     batch_size=batch_size)
    use_saved = False
    # path_to_model = 'models/srcnn-cifar10-20000-9_3_1_5-64_32_32-20epochs-he_uniform-custom_relu.h5'
    path_to_model = 'results/' + dataset_name + '/model/' + \
                    'srcnn-cifar10-20000-tf-20000images-9_3_1_5-64_32_32-3epochs-he_uniform-relu-nadam.h5'
    # 'srcnn-hundred-40-tf-40images-9_3_1_5-64_32_32-100epochs-he_uniform-custom_relu-adam.h5'
    # 'srcnn-cifar10-20000-tf-200images-9_3_1_5-64_32_32-200epochs-he_uniform-custom_relu-adam.h5'

    if use_saved:
        # returns a compiled model
        # identical to the previous one
        model = keras.models.load_model(path_to_model, custom_objects={'psnr_L': psnr_L})
    else:
        print('SRCNN tf fitting...')

        batch_size = 64  # in each iteration we consider 128 training examples at once
        num_epochs = 3
        f_1, f_2, f_2_2, f_3 = 9, 3, 1, 5
        n_1, n_2, n_2_2 = 64, 32, 32
        l2_lambda = 0.0001

        inp = Input(shape=(height_X_train, width_X_train, c))

        conv_1 = Conv2D(n_1, (f_1, f_1), padding='same', activation='relu', kernel_regularizer=l2(l2_lambda),
                        kernel_initializer='he_uniform')(inp)

        conv_2 = Conv2D(n_2, (f_2, f_2), padding='same', activation='relu', kernel_regularizer=l2(l2_lambda),
                        kernel_initializer='he_uniform')(conv_1)

        conv_2 = Conv2D(n_2_2, (f_2_2, f_2_2), padding='same', activation='relu',
                        kernel_regularizer=l2(l2_lambda),
                        kernel_initializer='he_uniform')(conv_2)

        conv_3 = Conv2D(c, (f_3, f_3), padding='same', activation=custom_relu,
                        kernel_regularizer=l2(l2_lambda), )(
            conv_2)

        # creating model
        model = Model(inputs=inp, outputs=conv_3)

        from keras.utils import plot_model
        plot_model(model, to_file='results/' + dataset_name + '/model/SRCNN-model.png', show_shapes=True,
                   show_layer_names=True, rankdir='TB')
        plot_model(model, to_file='results/' + dataset_name + '/model/SRCNN-model.svg', show_shapes=True,
                   show_layer_names=True, rankdir='TB')

        model.compile(loss='mean_squared_error',
                      optimizer='adam',  # nadam,  # 'nadam',
                      metrics=['accuracy',
                               psnr_RGB,
                               # M.mean_squared_logarithmic_error, # +++
                               # M.mean_squared_error,
                               # mse_L
                               ])

        print(model.summary())

        train_result = model.fit(X_train, Y_train,  # Train the model using the training set...
                                 batch_size=batch_size,
                                 epochs=num_epochs,
                                 verbose=1,
                                 validation_split=0.0,
                                 callbacks=[
                                     psnr_3_callback,
                                     # ssim_3_callback,
                                     min_max_callback,
                                 ])

        print_train_result(train_result)

        print('saving model to ' + path_to_model + '...')
        model.save(path_to_model)
        print('model ' + path_to_model + ' saved')

        # del model  # deletes the existing model

    # getting results
    test_result = model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set
    print_test_result(test_result)

    # making plot
    if not use_saved:
        plot_results(train_result, test_result, dataset_name, show=False)
        plot_psnr_rgb_results(train_result, test_result, dataset_name, show=False)

    print('PSNR history :', psnr_3_callback.history)
    print('PSNR epoch history : ', psnr_3_callback.epoch_history)

    print('MIN-MAX history :', min_max_callback.history)
    print('MIN-MAX epoch history : ', min_max_callback.epoch_history)

    prediction = model.predict(X_test, batch_size=batch_size, verbose=1)

    from keras.backend import clear_session
    clear_session()

def test(model, X_test, Y_test, verbose=0):
    print('neural network testing...')
    model.evaluate(X_test, Y_test, verbose=verbose)


def run(image_data):
    print('neural network running...')

    # preparing image_data
    test = np.array([image_data])
    test = test.astype('float32')
    test /= 255  # Normalise data to [0, 1] range
    test = test.reshape(1, 3, 32, 32)

    # loading model
    # path_to_model = 'models/srcnn-cifar10-20000-9_3_1_5-64_32_32-20epochs-he_uniform-custom_relu.h5'
    path_to_model = ''
    model = keras.models.load_model(path_to_model, custom_objects={
        'psnr_3_channels': psnr_3_channels,
        'custom_relu': custom_relu,
    })  # , custom_objects={'psnr_L': psnr_L})
    print(model.summary())

    batch_size = 64
    prediction = model.predict(test, batch_size=batch_size, verbose=1)

    from keras.backend import clear_session
    clear_session()

    prediction.resize((1, 32, 32, 3))
    prediction = np.rint(prediction * 255).astype('uint8')
    print_ndarray_info(prediction, verbose=True)  # reshape((3, 4)) => a ; resize((2,6)) => on place
    prediction_image_data = prediction[0]
    print_ndarray_info(prediction_image_data, verbose=True)

    from image_handler import get_image
    prediction_image = get_image(prediction_image_data, mode='RGB')
    # prediction_image.show()

    return prediction_image_data


def pickle_mnist():
    from keras.datasets import mnist
    mnist_data = mnist.load_data()
    print(mnist_data[:3])
    # (X_train, y_train), (X_test, y_test) = mnist_data
    # print(type(X_train), type(y_train))

    mnist_data_file = open('mnist-data.pkl', 'wb')
    pickle.dump(mnist_data, mnist_data_file)
    mnist_data_file.close()
    # fitted_model = pickle.load(neural_network_model_file)


def print_ndarray_info(ndarray, verbose=False):
    if verbose:
        print('ndim :', ndarray.ndim, ' shape :', ndarray.shape, ' size :', ndarray.size, ' dtype :', ndarray.dtype,
              ' itemsize :', ndarray.itemsize)
    else:
        print(ndarray.ndim, ndarray.shape, ndarray.size, ndarray.dtype, ndarray.itemsize)


def handle_time_range(secs):
    msecs, secs = math.modf(secs)
    msecs = int(msecs * 1000)
    secs = int(secs)
    mins = secs // 60
    secs %= 60
    return mins, secs, msecs


def convert_handled_time_range_to_str(handled_time_range):
    mins, secs, msecs = handled_time_range
    return '{} m {} s {} ms'.format(mins, secs, msecs)


if __name__ == '__main__':
    print('neural_network module running...')
    begin_time = time.time()

    fit()

    print('SRCNN running lasted', convert_handled_time_range_to_str(handle_time_range(time.time() - begin_time)))
