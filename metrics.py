import math
import numpy as np
from keras.callbacks import Callback
from skimage.measure import compare_psnr, compare_ssim

"""Metrics for evaluating image difference"""


def mse(true_image_data, pred_image_data):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    if true_image_data.shape != pred_image_data.shape:
        print('Trying to compare images with different dimensions.')
        exit()

    result = np.sum((true_image_data.astype('float32') - pred_image_data.astype('float32')) ** 2)
    result /= float(true_image_data.size)

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return result


def psnr(true_image_data, pred_image_data):
    return 20 * math.log10(np.max(true_image_data) / math.sqrt(mse(true_image_data, pred_image_data)))


class MetricsCallbackPSNR(Callback):
    def __init__(self, X, Y, batch_size):
        print('making PSNR callback metric...')
        super().__init__()
        self.X_train, self.X_test = X
        self.Y_train, self.Y_test = Y
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        print('PSNR callback metric running...')
        self.losses = []
        self.history = []
        self.epoch_history = []

    def on_epoch_end(self, epoch, logs=None):
        eval_on_train = True
        if eval_on_train:
            train_prediction = self.model.predict(self.X_train, batch_size=self.batch_size, verbose=0)
            result = self.psnr(self.Y_train, train_prediction)
        else:
            test_prediction = self.model.predict(self.X_test, batch_size=self.batch_size, verbose=0)
            result = self.psnr(self.Y_test, test_prediction)
        self.history.append(result)
        print(' - PSNR:', result)

    def on_train_end(self, logs=None):
        test_prediction = self.model.predict(self.X_test, batch_size=self.batch_size, verbose=0)
        self.test_result = self.psnr(self.Y_test, test_prediction)
        print('PSNR test result :', self.test_result)

    def psnr(self, true_data, pred_data):
        # Peak signal-to-noise ratio
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        # NORMAL VALUES : [30, 40]
        # print('PSNR metric running...')

        return compare_psnr(true_data, pred_data)

    def mse(self, true_data, pred_data):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        # NORMAL VALUES : [30, 40]
        # print('PSNR metric running...')
        dif = true_data - pred_data
        result = np.sum(dif ** 2)
        result /= float(true_data.size)
        return result


class MetricsCallbackSSIM(Callback):
    def __init__(self, X, Y, batch_size, mode='L'):
        print('making SSIM callback metric...')
        super().__init__()
        self.X_train, self.X_test = X
        self.Y_train, self.Y_test = Y
        self.batch_size = batch_size
        self.mode = mode

    def on_train_begin(self, logs={}):
        print('SSIM callback metric running...')
        self.losses = []
        self.history = []
        self.epoch_history = []

    def on_epoch_end(self, epoch, logs=None):
        eval_on_train = True
        if eval_on_train:
            train_prediction = self.model.predict(self.X_train, batch_size=self.batch_size, verbose=0)
            result = self.ssim(self.Y_train, train_prediction)
        else:
            test_prediction = self.model.predict(self.X_test, batch_size=self.batch_size, verbose=0)
            result = self.ssim(self.Y_test, test_prediction)
        self.history.append(result)
        print(' - SSIM:', result)

    def on_train_end(self, logs=None):
        test_prediction = self.model.predict(self.X_test, batch_size=self.batch_size, verbose=0)
        self.test_result = self.ssim(self.Y_test, test_prediction)
        print('SSIM test result :', self.test_result)

    def ssim(self, true_data, pred_data):
        if self.mode == 'L':
            item_results = []
            for i in range(len(true_data)):
                item_result = compare_ssim(true_data[i][0], pred_data[i][0])
                item_results.append(item_result)
            return np.average(item_results)
        return np.inf


class MetricsCallbackMinMax(Callback):
    def __init__(self, X, Y, batch_size):
        print('making MIN-MAX callback metric...')
        super().__init__()
        self.X_train, self.X_test = X
        self.Y_train, self.Y_test = Y
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        print('MIN-MAX callback metric running...')
        self.losses = []
        self.history = []
        self.epoch_history = []

    def on_epoch_end(self, epoch, logs=None):
        eval_on_train = True
        if eval_on_train:
            train_prediction = self.model.predict(self.X_train, batch_size=self.batch_size, verbose=0)
            result = self.min_max(self.Y_train, train_prediction)
        else:
            test_prediction = self.model.predict(self.X_test, batch_size=self.batch_size, verbose=0)
            result = self.min_max(self.Y_test, test_prediction)
        self.history.append(result)
        # print(' - MIN-MAX:', result)

    def on_train_end(self, logs=None):
        test_prediction = self.model.predict(self.X_test, batch_size=self.batch_size, verbose=0)
        self.test_result = self.min_max(self.Y_test, test_prediction)
        print('MIN-MAX test result :', self.test_result)

    def min_max(self, true_data, pred_data):
        return np.min(pred_data), np.max(pred_data)