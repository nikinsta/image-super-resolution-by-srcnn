import pickle
import numpy as np
from keras.datasets import mnist, cifar10, cifar100, boston_housing, imdb, reuters
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageTk


def validate_value_in_range(name, value, begin, end):
    print('validation...')
    if begin > end:
        print('VALIDATION ERROR : begin less than end')
        exit()
    if not (begin <= value <= end):
        print(name + ' ' + str(value) + ' is invalid! Valid range : [' + str(begin) + ', ' + str(end) + ']')
        exit()
    print('ACCEPTED')


def make_srcnn_dataset_based_on_mnist():
    print('making SRCNN dataset using MNIST...')

    from image_handler import get_image, get_image_data, zoom_out_image, zoom_up_image

    (Y_train, _), (Y_test, _) = mnist.load_data()  # 60000, 10000

    # making X_train list
    print('making X_train list...')
    X_train = []
    for item in Y_train:
        image_data = item.tolist()
        image = get_image(image_data, mode='L')
        zoomed_out_image = zoom_out_image(image, times=2)
        zoomed_up_image = zoom_up_image(zoomed_out_image, times=2)
        zoomed_up_image_data = get_image_data(zoomed_up_image)
        X_train.append(zoomed_up_image_data)

    # making X_test list
    print('making X_test list...')
    X_test = []
    for item in Y_test:
        image_data = item.tolist()
        image = get_image(image_data, mode='L')
        zoomed_out_image = zoom_out_image(image, times=2)
        zoomed_up_image = zoom_up_image(zoomed_out_image, times=2)
        zoomed_up_image_data = get_image_data(zoomed_up_image)
        X_test.append(zoomed_up_image_data)

    # X_train, X_test = [], []
    # for X, Y in [(X_train, Y_train), (X_test, Y_test)]:
    #     for item in Y:
    #         image_data = item.tolist()
    #         image = get_image(image_data, mode='L')
    #         zoomed_out_image = zoom_out_image(image, times=2)
    #         zoomed_up_image = zoom_up_image(zoomed_out_image, times=2)
    #         zoomed_up_image_data = get_image_data(zoomed_up_image)
    #         X.append(zoomed_up_image_data)

    dtype = 'uint8'
    dataset = (np.array(X_train, dtype=dtype), np.array(Y_train, dtype=dtype)), \
              (np.array(X_test, dtype=dtype), np.array(Y_test, dtype=dtype))
    print('saving dataset to srcnn-mnist-dataset.npz...')
    np.savez('datasets/srcnn-mnist-dataset.npz', dataset)


def make_srcnn_rgb_dataset_based_on_mnist():
    print('making SRCNN-RGB dataset using SRCNN dataset based on MNIST...')

    (X_train, Y_train), (X_test, Y_test) = get_srcnn_mnist_dataset_part(train_part=0.2)

    from image_handler import convert_l_image_data_to_rgb_image_data

    # X_train
    print('making X_train list...')
    rgb_image_data_list = []
    for l_image_data in X_train:
        rgb_image_data = convert_l_image_data_to_rgb_image_data(l_image_data)
        rgb_image_data_list.append(rgb_image_data)
    X_train = np.array(rgb_image_data_list)

    # Y_train
    print('making Y_train list...')
    rgb_image_data_list = []
    for l_image_data in Y_train:
        rgb_image_data = convert_l_image_data_to_rgb_image_data(l_image_data)
        rgb_image_data_list.append(rgb_image_data)
    Y_train = np.array(rgb_image_data_list)

    # X_test
    print('making X_test list...')
    rgb_image_data_list = []
    for l_image_data in X_test:
        rgb_image_data = convert_l_image_data_to_rgb_image_data(l_image_data)
        rgb_image_data_list.append(rgb_image_data)
    X_test = np.array(rgb_image_data_list)

    # Y_test
    print('making Y_test list...')
    rgb_image_data_list = []
    for l_image_data in Y_test:
        rgb_image_data = convert_l_image_data_to_rgb_image_data(l_image_data)
        rgb_image_data_list.append(rgb_image_data)
    Y_test = np.array(rgb_image_data_list)

    dtype = 'uint8'
    dataset = (np.array(X_train, dtype=dtype), np.array(Y_train, dtype=dtype)), \
              (np.array(X_test, dtype=dtype), np.array(Y_test, dtype=dtype))
    print('saving dataset to srcnn-rgb-mnist-dataset.npz...')
    np.savez('datasets/srcnn-rgb-mnist-dataset.npz', dataset)


def make_srcnn_rgb_dataset_based_on_cifar10():
    print('making SRCNN-RGB dataset using CIFAR-10...')
    (Y_train, _), (Y_test, _) = get_dataset_part(cifar10.load_data(), train_part=0.2)

    from image_handler import get_image, get_image_data, zoom_out_image, zoom_up_image

    # X_train
    print('making X_train list...')
    X_train = []
    for item in Y_train:
        image_data = item.tolist()
        image = get_image(image_data, mode='RGB')
        zoomed_out_image = zoom_out_image(image, times=2)
        zoomed_up_image = zoom_up_image(zoomed_out_image, times=2)
        zoomed_up_image_data = get_image_data(zoomed_up_image)
        X_train.append(zoomed_up_image_data)

    # X_test
    print('making X_test list...')
    X_test = []
    for item in Y_test:
        image_data = item.tolist()
        image = get_image(image_data, mode='RGB')
        zoomed_out_image = zoom_out_image(image, times=2)
        zoomed_up_image = zoom_up_image(zoomed_out_image, times=2)
        zoomed_up_image_data = get_image_data(zoomed_up_image)
        X_test.append(zoomed_up_image_data)

    dtype = 'uint8'
    dataset = (np.array(X_train, dtype=dtype), np.array(Y_train, dtype=dtype)), \
              (np.array(X_test, dtype=dtype), np.array(Y_test, dtype=dtype))
    print('saving dataset to srcnn-rgb-cifar10-dataset.npz...')
    np.savez('datasets/srcnn-rgb-cifar10-dataset.npz', dataset)


def make_srcnn_rgb_dataset_based_on_cifar10_20000():
    print('making SRCNN-RGB (20000) dataset using CIFAR-10..')
    (Y_train, _), (Y_test, _) = get_dataset_part(cifar10.load_data(), train_part=0.4, test_part=1)

    from image_handler import get_image, get_image_data, zoom_out_image, zoom_up_image

    # X_train
    print('making X_train list...')
    X_train = []
    for item in Y_train:
        image_data = item.tolist()
        image = get_image(image_data, mode='RGB')
        zoomed_out_image = zoom_out_image(image, times=2)
        zoomed_up_image = zoom_up_image(zoomed_out_image, times=2)
        zoomed_up_image_data = get_image_data(zoomed_up_image)
        X_train.append(zoomed_up_image_data)

    # X_test
    print('making X_test list...')
    X_test = []
    for item in Y_test:
        image_data = item.tolist()
        image = get_image(image_data, mode='RGB')
        zoomed_out_image = zoom_out_image(image, times=2)
        zoomed_up_image = zoom_up_image(zoomed_out_image, times=2)
        zoomed_up_image_data = get_image_data(zoomed_up_image)
        X_test.append(zoomed_up_image_data)

    dtype = 'uint8'
    dataset = (np.array(X_train, dtype=dtype), np.array(Y_train, dtype=dtype)), \
              (np.array(X_test, dtype=dtype), np.array(Y_test, dtype=dtype))
    print('saving dataset to srcnn-rgb-cifar10-20000-dataset.npz...')
    np.savez_compressed('datasets/srcnn-rgb-cifar10-20000-dataset.npz', dataset)


def make_pasadena_dataset():
    print('making PASADENA dataset...')

    from image_handler import get_image_data, get_image, zoom_out_image, zoom_up_image

    path = 'saved_images/Pasadena Dataset/'
    filename_prefix = path + 'dcp_24'
    size = 10
    result = []

    for i in range(size):
        filename = filename_prefix + str(12 + i) + '.jpg'
        image = Image.open(filename)
        print('Image', filename.rpartition('/')[2], 'opened')
        image_data = get_image_data(image)
        result.append(image_data)

    print('making Y_train and Y_test...')
    train_size = 7
    Y_train, Y_test = result[:train_size], result[train_size:]

    # X_train
    print('making X_train list...')
    X_train = []
    for item in Y_train:
        image_data = item  # .tolist()
        image = get_image(image_data, mode='RGB')
        zoomed_out_image = zoom_out_image(image, times=2)
        zoomed_up_image = zoom_up_image(zoomed_out_image, times=2)
        zoomed_up_image_data = get_image_data(zoomed_up_image)
        X_train.append(zoomed_up_image_data)

    # X_test
    print('making X_test list...')
    X_test = []
    for item in Y_test:
        image_data = item  # .tolist()
        image = get_image(image_data, mode='RGB')
        zoomed_out_image = zoom_out_image(image, times=2)
        zoomed_up_image = zoom_up_image(zoomed_out_image, times=2)
        zoomed_up_image_data = get_image_data(zoomed_up_image)
        X_test.append(zoomed_up_image_data)

    dtype = 'uint8'
    dataset = (np.array(X_train, dtype=dtype), np.array(Y_train, dtype=dtype)), \
              (np.array(X_test, dtype=dtype), np.array(Y_test, dtype=dtype))
    print('saving dataset to pasadena-dataset.npz...')
    np.savez_compressed('datasets/pasadena-dataset.npz', dataset)
    print('pasadena-dataset.npz saved')


def make_hundred_dataset():
    print('making HUNDRED dataset...')

    from image_handler import get_image_data, get_image, zoom_out_image, zoom_up_image

    path = 'images/Hundred Dataset/images/'
    size = 53
    result = []

    for i in range(size):
        filename = path + str(1 + i) + '.png'
        image = Image.open(filename)
        print('Image', filename.rpartition('/')[2], 'opened')
        image_data = get_image_data(image)
        result.append(image_data)

    print('making Y_train and Y_test...')
    train_size = 40
    Y_train, Y_test = result[:train_size], result[train_size:]

    # X_train
    print('making X_train list...')
    X_train = []
    for item in Y_train:
        image_data = item  # .tolist()
        image = get_image(image_data, mode='RGB')
        zoomed_out_image = zoom_out_image(image, times=2)
        zoomed_up_image = zoom_up_image(zoomed_out_image, times=2)
        zoomed_up_image_data = get_image_data(zoomed_up_image)
        X_train.append(zoomed_up_image_data)

    # X_test
    print('making X_test list...')
    X_test = []
    for item in Y_test:
        image_data = item  # .tolist()
        image = get_image(image_data, mode='RGB')
        zoomed_out_image = zoom_out_image(image, times=2)
        zoomed_up_image = zoom_up_image(zoomed_out_image, times=2)
        zoomed_up_image_data = get_image_data(zoomed_up_image)
        X_test.append(zoomed_up_image_data)

    dtype = 'uint8'
    dataset = (np.array(X_train, dtype=dtype), np.array(Y_train, dtype=dtype)), \
              (np.array(X_test, dtype=dtype), np.array(Y_test, dtype=dtype))
    print('saving dataset to hundred-dataset.npz...')
    np.savez('datasets/hundred-dataset.npz', dataset)  # _compressed
    print('hundred-dataset.npz saved')


def make_100_86_dataset():
    print('making 100-86 dataset...')

    from image_handler import get_image_data, get_image, zoom_out_image, zoom_up_image

    path = 'images/Hundred Dataset/images/'
    size = 53
    result = []

    for i in range(size):
        filename = path + str(1 + i) + '.png'
        image = Image.open(filename)
        print('Image', filename.rpartition('/')[2], 'opened')
        image_data = get_image_data(image)
        result.append(image_data)

    print('making Y_train (100x100) and Y_test (100x100)...')
    train_size = 40
    Y_train, Y_test = result[:train_size], result[train_size:]

    # X_train
    print('making X_train list from Y_train (100x100)...')
    X_train = []
    for item in Y_train:
        image_data = item  # .tolist()
        image = get_image(image_data, mode='RGB')
        zoomed_out_image = zoom_out_image(image, times=2)
        zoomed_up_image = zoom_up_image(zoomed_out_image, times=2)
        zoomed_up_image_data = get_image_data(zoomed_up_image)
        X_train.append(zoomed_up_image_data)

    # X_test
    print('making X_test list from Y_test (100x100)...')
    X_test = []
    for item in Y_test:
        image_data = item  # .tolist()
        image = get_image(image_data, mode='RGB')
        zoomed_out_image = zoom_out_image(image, times=2)
        zoomed_up_image = zoom_up_image(zoomed_out_image, times=2)
        zoomed_up_image_data = get_image_data(zoomed_up_image)
        X_test.append(zoomed_up_image_data)

    k = 100 / 86
    print('making Y_train (86x86)...')
    temp = []
    for image_data in Y_train:
        image = get_image(image_data, mode='RGB')
        zoomed_out_image = zoom_out_image(image, times=k)
        zoomed_out_image_data = get_image_data(zoomed_out_image)
        temp.append(zoomed_out_image_data)
    Y_train = temp

    print('making Y_test (86x86)...')
    temp = []
    for image_data in Y_test:
        image = get_image(image_data, mode='RGB')
        zoomed_out_image = zoom_out_image(image, times=k)
        zoomed_out_image_data = get_image_data(zoomed_out_image)
        temp.append(zoomed_out_image_data)
    Y_test = temp

    dtype = 'uint8'
    dataset = (np.array(X_train, dtype=dtype), np.array(Y_train, dtype=dtype)), \
              (np.array(X_test, dtype=dtype), np.array(Y_test, dtype=dtype))
    print('saving dataset to 100-86-dataset.npz...')
    np.savez('datasets/100-86-dataset.npz', dataset)  # _compressed
    print('100-86-dataset.npz saved')


def handle_mnist():
    from image_handler import get_image, get_image_data, zoom_out_image
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    dataset_size = 60000

    dataset_X, dataset_Y = [], []
    for i, X_train_item in enumerate(X_train[:dataset_size]):
        image_data = X_train_item.tolist()
        image = get_image(image_data, mode='L')
        zoomed_out_image = zoom_out_image(image, times=2)
        zoomed_out_image_data = get_image_data(zoomed_out_image)

        dataset_X.append(zoomed_out_image_data)
        dataset_Y.append(image_data)

        print('image ' + str(i) + ', height : ' + str(len(image_data)), 'width : ' + str(len(image_data[0])), sep=', ')
        print('zoomed_out_image ' + str(i) + ', height : ' + str(len(zoomed_out_image_data)),
              'width : ' + str(len(zoomed_out_image_data[0])), sep=', ')
        # print('image_data :', image_data)
        # print('zoomed_out_image_data :', zoomed_out_image_data)
        # image.show()
        # zoomed_out_image.show()

    dataset = (dataset_X, dataset_Y)

    import pickle
    mnist_dataset_file = open('datasets/mnist-dataset.pkl', 'wb')
    pickle.dump(dataset, mnist_dataset_file)
    mnist_dataset_file.close()


def convert_mnist_dataset_to_ndarrays():
    mnist_dataset_file = open('datasets/mnist-dataset.pkl', 'rb')
    dataset_X, dataset_Y = pickle.load(mnist_dataset_file)
    mnist_dataset_file.close()

    dataset_size = len(dataset_X) // 1
    dataset_X, dataset_Y = dataset_X[:dataset_size], dataset_Y[:dataset_size]

    train_size = int(len(dataset_X) * 0.9)
    X_train, X_test = dataset_X[:train_size], dataset_X[train_size:]
    Y_train, Y_test = dataset_Y[:train_size], dataset_Y[train_size:]

    mnist_dataset = (np.array(X_train, dtype='uint8'), np.array(Y_train, dtype='uint8')), \
                    (np.array(X_test, dtype='uint8'), np.array(Y_test, dtype='uint8'))

    np.savez('datasets/mnist-dataset', mnist_dataset)  # .npz

    # return (np.array(X_train, dtype='uint8'), np.array(Y_train, dtype='uint8')), \
    #        (np.array(X_test, dtype='uint8'), np.array(Y_test, dtype='uint8'))


def show_srcnn_mnist_dataset_example(count=1):
    from image_handler import get_image
    (X_train, Y_train), (X_test, Y_test) = get_srcnn_mnist_dataset()

    for i in range(count):
        get_image(X_train[i], mode='L').show()
        get_image(Y_train[i], mode='L').show()

    for i in range(count):
        get_image(X_test[i], mode='L').show()
        get_image(Y_test[i], mode='L').show()


def show_mnist_example(count=1):
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    from image_handler import get_image
    for i in range(count):
        get_image(X_train[i].tolist())


def show_cifar10_example(count=1):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # num_train, height, width, depth = X_train.shape
    print(X_train.shape)
    # print(X_train[0].tolist())

    from image_handler import get_image
    for i in range(count):
        get_image(X_train[i].tolist(), mode='RGB').show()


def show_cifar100_example(count=1):
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()

    # num_train, height, width, depth = X_train.shape
    print(X_train.shape)
    # print(X_train[0].tolist())

    from image_handler import get_image
    for i in range(count):
        get_image(X_train[i].tolist(), mode='RGB').show()


def show_srcnn_rgb_cifar10_dataset_example(count=1):
    (X_train, Y_train), (X_test, Y_test) = get_srcnn_rgb_cifar10_dataset()
    from image_handler import get_image
    for i in range(count):
        get_image(X_train[i], mode='RGB').show()
        get_image(Y_train[i], mode='RGB').show()
        # get_image(X_test[i], mode='RGB').show()
        # get_image(Y_test[i], mode='RGB').show()


def show_srcnn_rgb_cifar10_20000_dataset_example(count=1):
    (X_train, Y_train), (X_test, Y_test) = get_srcnn_rgb_cifar10_20000_dataset()
    from image_handler import get_image
    for i in range(count):
        # get_image(X_train[i], mode='RGB').show()
        # get_image(Y_train[i], mode='RGB').show()
        # get_image(X_test[i], mode='RGB').show()
        get_image(Y_test[i], mode='RGB').show()

        get_image(X_train[i], mode='RGB').save('saved_images/cifar10_20000_X_train[0].png')
        get_image(Y_train[i], mode='RGB').save('saved_images/cifar10_20000_Y_train[0].png')


def show_pasadena_dataset_example(count=1):
    (X_train, Y_train), (X_test, Y_test) = get_pasadena_dataset()
    from image_handler import get_image
    for i in range(count):
        get_image(X_train[i], mode='RGB').show()
        get_image(Y_train[i], mode='RGB').show()
        # get_image(X_test[i], mode='RGB').show()
        # get_image(Y_test[i], mode='RGB').show()

        # get_image(X_train[i], mode='RGB').save('saved_images/pasadena_X_train[0].png')
        # get_image(Y_train[i], mode='RGB').save('saved_images/pasadena_Y_train[0].png')


def show_hundred_dataset_example(count=1):
    (X_train, Y_train), (X_test, Y_test) = get_hundred_dataset()
    from image_handler import get_image
    for i in range(count):
        # get_image(X_train[i], mode='RGB').show()
        # get_image(Y_train[i], mode='RGB').show()
        get_image(X_test[i], mode='RGB').show()
        get_image(Y_test[i], mode='RGB').show()

        # get_image(X_train[i], mode='RGB').save('saved_images/pasadena_X_train[0].png')
        # get_image(Y_train[i], mode='RGB').save('saved_images/pasadena_Y_train[0].png')


def show_dataset_example(dataset, count=1):
    (X_train, Y_train), (X_test, Y_test) = dataset
    from image_handler import get_image
    for i in range(count):
        # get_image(X_train[i], mode='RGB').show()
        # get_image(Y_train[i], mode='RGB').show()
        get_image(X_test[i], mode='RGB').show()
        get_image(Y_test[i], mode='RGB').show()

        # get_image(X_train[i], mode='RGB').save('saved_images/pasadena_X_train[0].png')
        # get_image(Y_train[i], mode='RGB').save('saved_images/pasadena_Y_train[0].png')


def show_100_86_dataset_example(count=1):
    show_dataset_example(get_100_86_dataset(), count=count)


def get_dataset_part(dataset, train_part=1.0, test_part=1.0):
    validate_value_in_range('train_part', train_part, 0, 1)
    validate_value_in_range('test_part', test_part, 0, 1)

    (X_train, Y_train), (X_test, Y_test) = dataset
    train_size = int(len(X_train) * train_part)
    test_size = int(len(X_test) * test_part)
    return (X_train[:train_size], Y_train[:train_size]), (X_test[:test_size], Y_test[:test_size])


def get_mnist_dataset(path='datasets/mnist-dataset.npz'):
    print('getting mnist dataset...')
    npzfile = np.load(path)
    return npzfile['arr_0']


def get_mnist_dataset_part(train_part=1.0, test_part=1.0):
    return get_dataset_part(get_mnist_dataset(), train_part=train_part, test_part=test_part)


def get_srcnn_mnist_dataset(path='datasets/srcnn-mnist-dataset.npz'):
    print('getting SRCNN mnist dataset...')  # (60000, 28, 28) (10000, 28, 28)
    npzfile = np.load(path)
    return npzfile['arr_0']


def get_srcnn_mnist_dataset_part(train_part=1.0, test_part=1.0):
    return get_dataset_part(get_srcnn_mnist_dataset(), train_part=train_part, test_part=test_part)


def get_srcnn_rgb_mnist_dataset(path='datasets/srcnn-rgb-mnist-dataset.npz'):
    print('getting SRCNN-RGB mnist dataset...')  # (12000, 28, 28, 3) (10000, 28, 28, 3)
    npzfile = np.load(path)
    return npzfile['arr_0']


def get_srcnn_rgb_mnist_dataset_part(train_part=1.0, test_part=1.0):
    return get_dataset_part(get_srcnn_rgb_mnist_dataset(), train_part=train_part, test_part=test_part)


def get_srcnn_rgb_cifar10_dataset(path='datasets/srcnn-rgb-cifar10-dataset.npz'):
    print('getting SRCNN-RGB CIFAR-10 dataset...')  # (10000, 32, 32, 3) (10000, 32, 32, 3)
    print('opening ' + path + '...')
    npzfile = np.load(path)
    return npzfile['arr_0']


def get_srcnn_rgb_cifar10_dataset_part(train_part=1.0, test_part=1.0):
    return get_dataset_part(get_srcnn_rgb_cifar10_dataset(), train_part=train_part, test_part=test_part)


def get_srcnn_rgb_cifar10_20000_dataset(path='datasets/srcnn-rgb-cifar10-20000-dataset.npz'):
    print('getting SRCNN-RGB CIFAR-10 (20000) dataset ...')  # (10000, 32, 32, 3) (10000, 32, 32, 3)
    print('opening ' + path + '...')
    npzfile = np.load(path)
    return npzfile['arr_0']


def get_srcnn_rgb_cifar10_20000_dataset_part(train_part=1.0, test_part=1.0):
    return get_dataset_part(get_srcnn_rgb_cifar10_20000_dataset(), train_part=train_part, test_part=test_part)


def get_pasadena_dataset(path='datasets/pasadena-dataset.npz'):
    print('getting PASADENA dataset ...')  # (10000, 1760, 1168, 3) (10000, 32, 32, 3)
    print('opening ' + path + '...')
    npzfile = np.load(path)
    return npzfile['arr_0']


def get_pasadena_dataset_part(train_part=1.0, test_part=1.0):
    return get_dataset_part(get_pasadena_dataset(), train_part=train_part, test_part=test_part)


def get_hundred_dataset(path='datasets/hundred-dataset.npz'):
    print('getting HUNDRED dataset ...')  # (40, 100, 100, 3) (13, 100, 100, 3)
    print('opening ' + path + '...')
    npzfile = np.load(path)
    return npzfile['arr_0']


def get_hundred_dataset_part(train_part=1.0, test_part=1.0):
    return get_dataset_part(get_hundred_dataset(), train_part=train_part, test_part=test_part)


def get_100_86_dataset(path='datasets/100-86-dataset.npz'):
    print('getting 100-86 dataset ...')  # (40, 100, 100, 3) -> (40, 86, 86, 3) | (13, 100, 100, 3) -> (13, 86, 86, 3)
    print('opening ' + path + '...')
    npzfile = np.load(path)
    return npzfile['arr_0']


def get_100_86_dataset_part(train_part=1.0, test_part=1.0):
    return get_dataset_part(get_100_86_dataset(), train_part=train_part, test_part=test_part)

def get_srcnn_rgb_cifar10_20000_dataset_interpolation_metrics():
    from image_handler import make_interpolated_image_from, get_images_difference_metrics, get_image

    (X_train, Y_train), (X_test, Y_test) = get_srcnn_rgb_cifar10_20000_dataset()

    for image_data in Y_train:
        get_image()


if __name__ == '__main__':
    print('dataset module running...')
    # make_srcnn_dataset_based_on_mnist()
    # make_srcnn_rgb_dataset_based_on_mnist()
    # show_srcnn_mnist_dataset_example()

    # show_cifar100_example(5)
    # make_srcnn_rgb_dataset_based_on_cifar10()

    # dataset = get_srcnn_rgb_cifar10_dataset()
    # print(dataset[0][0].shape, dataset[1][0].shape)

    # show_srcnn_rgb_cifar10_dataset_example()

    # make_srcnn_rgb_dataset_based_on_cifar10_20000()
    # show_srcnn_rgb_cifar10_20000_dataset_example(count=1)

    # make_pasadena_dataset()
    # show_pasadena_dataset_example()

    # make_hundred_dataset()
    # show_hundred_dataset_example(count=2)

    # make_100_86_dataset()
    # show_100_86_dataset_example(count=3)