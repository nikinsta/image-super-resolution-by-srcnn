from PIL import Image

import neural_network

import numpy as np
import math
import metrics

from skimage.measure import compare_psnr, compare_ssim, compare_mse


def print_image_data1(image):
    print('Image data:')
    print('format:', image.format)
    print('mode:', image.mode)
    print('width:', image.width)
    print('height:', image.height)


def print_image_data(image):
    print('IMAGE DATA:')
    print('format:', image.format, 'mode:', image.mode)
    print('width:', image.width, 'height:', image.height)


def handle_image(image):
    print('image handling running...')

    print_image_data(image)
    mode = image.mode

    image_data = get_image_data(image)
    handled_image_data = neural_network.run(image_data)

    return get_image(handled_image_data, mode=mode)


def get_image(image_data, mode='L'):
    image_data = np.array(image_data).astype('uint8')
    image = Image.fromarray(image_data, mode=mode)
    # image.show()
    return image


def convert_to_white_black(image):
    print('converting to white-black running...')
    return image.convert('L')


def convert_to_rgb(image):
    # print('converting to rgb running...')
    return image.convert('RGB')


def convert_to_YCbCr(image):
    print('converting to YCbCr running...')
    return image.convert('YCbCr')


def convert_l_image_data_to_rgb_image_data(l_image_data):
    image = get_image(l_image_data, mode='L')
    rgb_image = convert_to_rgb(image)
    return get_image_data(rgb_image)


def get_images_difference(true_image, pred_image, metric='psnr'):
    if metric == 'psnr':
        true_image_data = get_image_data(true_image)
        pred_image_data = get_image_data(pred_image)
        return metrics.psnr(true_image_data, pred_image_data)
    else:
        print('Unknown metric ' + str(metric))
        exit()


def get_images_difference_metrics(true_image, pred_image):
    print('computing images difference metrics...')
    true_image = convert_to_rgb(true_image)
    test_image = convert_to_rgb(pred_image)
    true_image_data = get_image_data(true_image)
    test_image_data = get_image_data(test_image)
    true_image_data = np.array(true_image_data, dtype='uint8')
    test_image_data = np.array(test_image_data, dtype='uint8')

    psnr = compare_psnr(true_image_data, test_image_data)
    ssim = compare_ssim(true_image_data, test_image_data, multichannel=True)
    mse = compare_mse(true_image_data, test_image_data)

    return psnr.item(), ssim.item(), mse.item()


def get_image_data(image, mode='L'):
    image_data = list(image.getdata())
    image_data = np.array(image_data)
    if image_data.ndim == 1:
        # print('getting L image data...')
        image_data.shape = (image.height, image.width)
    elif image_data.ndim == 2:
        # print('getting RGB image data...')
        image_data.shape = (image.height, image.width, 3)
    return image_data.tolist()


def get_image_head_data(image):
    return list(image.getdata())[:10]


def zoom_out_image(image, times=2.0):
    # print('zooming out the image')
    return image.resize((int(image.width / times), int(image.height / times)))


def zoom_up_image(image, times=2.0):
    # print('zooming up the image')
    return image.resize((int(image.width * times), int(image.height * times)))


def make_interpolated_image_from(image, times=2.0, interpolation=Image.BICUBIC, data=(0,0,1,1), method=Image.EXTENT):
    # Interpolation
    # Image.BICUBIC
    # Image.BILINEAR
    # Image.NEAREST
    # Methods
    # Image.AFFINE
    # Image.EXTENT
    # Image.QUAD
    # Image.PERSPECTIVE
    # image = Image.open('')
    data = (0, 0, image.width, image.height)
    zoomed_out_image = zoom_out_image(image, times=times)
    zoomed_up_image = zoom_up_image(zoomed_out_image, times=2.0)
    return zoomed_up_image.transform(image.size, method=method, data=data, resample=interpolation)


if __name__ == '__main__':
    print('image_handler module running...')

    # image_data = [[255,123,10,0]]
    # image = get_image(image_data, mode='L')
    # # image.show()
    #
    # image_data = [[0,0,0, 0], [100,100,100,100], [255,255,255]]
    # image_data = [[[0, 50, 0], [255, 255, 0], [234, 0, 234]],
    #               [[0, 50, 0], [255, 255, 0], [234, 0, 234]]]
    # image = get_image(image_data, mode='RGB')
    # image = Image.open('results/CIFAR-10 20000 tf/test/4_original_zoomed_out.png')
    # # image.show()
    # interpolated_image = make_interpolated_image_from(image, times=2.0)
    # interpolated_image.show()
    # image.show()

    # handle_mnist()
    # from neural_network import get_mnist_dataset

    # dataset_X, dataset_Y = get_mnist_dataset()
    # print('dataset_X', len(dataset_X))
    # print('dataset_Y', len(dataset_Y))
    # (X_train, Y_train), (X_test, Y_test) = get_mnist_dataset()
    # print('X_train', len(X_train))
    # print('Y_train', len(Y_train))
    # print('X_test', len(X_test))
    # # print('Y_test', len(Y_test))
    #
    # path_to_images = 'images/'
    # get_images_difference()

    # make_srcnn_dataset_based_on_mnist()


    # show_mnist_example()
    # show_cifar10_example()


    # image_data = [[255, 255, 255], [0, 0, 0], [100, 100, 100], [100, 100, 100]]
    # image_data = np.array(image_data).astype('uint8')
    # print(image_data)
    # # image_data = np.reshape(image_data, 256, 256)
    # image = Image.fromarray(image_data, mode='L')
    # image.show()

    # preparing the image dataset


    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator


    plt.title('')
    # ro, r--, bs, g^, -, -., :
    plt.figure(1, figsize=(10, 12))  # 10 6
    # ax = plt.figure(1).gca()  # plt.figure(1)

    # plt.subplot(subplots[i])
    # plt.subplot(221)
    # plt.title('Функции активации')
    plt.xlabel('z')
    plt.ylabel('σ(z)')

    X = np.arange(-2, 2.1, 0.01).tolist()
    #y = range(-2, 2.1, 0.5)
    def identity(x):
        return x
    def tanh(x):
        return math.tanh(x)
    def logistic(x):
        return 1 / (1 + math.exp(-x))
    def relu(x):
        return max(0, x)
    Y = []
    for x in X:
        Y.append(identity(x))
    plt.plot(X, Y, 'blue', label='Тождественная функция')

    Y = []
    for x in X:
        Y.append(logistic(x))
    plt.plot(X, Y, 'red', label='Логистическая функция')

    Y = []
    for x in X:
        Y.append(tanh(x))
    plt.plot(X, Y, 'orange', label='Гиперболический тангенс')

    Y = []
    for x in X:
        Y.append(relu(x))
    plt.plot(X, Y, 'green', label='Полулинейная функция')

    plt.legend()
    # plt.legend((lab1, lab3, lab2, lab4, lab2_5), frameon=False)


    # train result
    # plt.plot(epoch, values, 'r')
    # dy = 0
    # # max train result
    # plt.plot(epoch, [max_value] * len(epoch), 'r:')
    # plt.text(epoch[-2], max_value + dy, str(round(max_value, 4)))
    # # min train result
    # plt.plot(epoch, [min_value] * len(epoch), 'r:')
    # plt.text(epoch[-2], min_value + dy, str(round(min_value, 4)))
    # # test result
    # plt.plot(epoch, [test_result_item] * len(epoch), 'b')
    # plt.text(epoch[0], test_result_item + dy, str(round(test_result_item, 4)))

    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.10, right=0.90, hspace=0.5,
                        wspace=0.4)

    plt.show()
    path = 'saved_images/activations.png'
    print('saving results to ' + path)
    plt.savefig(path)