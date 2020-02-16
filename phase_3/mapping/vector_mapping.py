import cv2
import numpy as np
import scipy
from skimage.feature import local_binary_pattern


# Base class for vector mapping
class VectorMapping:

    def __init__(self):
        pass

    @staticmethod
    def get_yuv_for_image(_image_path):
        _image = cv2.imread(_image_path)
        _image_yuv = cv2.cvtColor(_image, cv2.COLOR_BGR2YUV)
        return _image_yuv

    @staticmethod
    def get_greyscale_for_image(_image_path):
        _image = cv2.imread(_image_path)
        _image_greyscale = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
        return _image_greyscale

    @staticmethod
    def get_image_windows(_image):
        _windows = []
        for i in range(0, _image.shape[0], 100):  # for every pixel take 100 at a time
            for j in range(0, _image.shape[1], 100):  # for every pixel take 100 at a time
                _windows.append(_image[i:(i + 100), j:(j + 100)])

        return np.array(_windows)


# LBP Vector mapping impl
class LbpVectorMapping(VectorMapping):

    def __init__(self):
        VectorMapping.__init__(self)

    @staticmethod
    def get_lbp_from_image_window(_image_window):
        radius = 2
        no_points = 8 * radius
        _lbp = np.array(local_binary_pattern(_image_window, no_points, radius, method='uniform'))
        (hist, _) = np.histogram(_lbp.ravel(), bins=np.arange(0, no_points + 3), range=(0, no_points + 2))
        hist = hist.astype("float")
        hist = hist / (hist.sum() + 1e-7)
        return hist

    def get_lbp_from_image(self, _image_path):
        image_greyscale = VectorMapping.get_greyscale_for_image(_image_path)
        image_windows_greyscale = VectorMapping.get_image_windows(image_greyscale)
        _lbp_image = np.array([self.get_lbp_from_image_window(window) for window in image_windows_greyscale]).ravel()
        return _lbp_image


class ColorMomentsVectorMapping(VectorMapping):

    def __init__(self):
        VectorMapping.__init__(self)

    @staticmethod
    def get_color_moments_from_image_window(_image_window_yuv):
        image_window_y, image_window_u, image_window_v = cv2.split(_image_window_yuv)

        image_window_color_moments = np.ravel(
            [[np.mean(channel), np.math.sqrt(np.var(channel)), scipy.stats.skew(channel.ravel())] for channel in
             [image_window_y, image_window_u, image_window_v]])

        return image_window_color_moments

    def get_color_moments_from_image(self, _image_path):
        _image_yuv = VectorMapping.get_yuv_for_image(_image_path)
        image_windows_yuv = VectorMapping.get_image_windows(_image_yuv)

        _cm_image = np.ravel(
            [self.get_color_moments_from_image_window(image_window_yuv) for image_window_yuv in image_windows_yuv]
        )

        return _cm_image


class HogVectorMapping(VectorMapping):

    def __init__(self):
        VectorMapping.__init__(self)

    @staticmethod
    def get_hog_from_image(_image_path):
        image = cv2.imread(_image_path)
        image = scipy.misc.imresize(image, 0.1)
        win_size = (8, 8)
        block_size = (8, 8)
        block_stride = (8, 8)
        cell_size = (2, 2)
        no_bins = 9
        deriv_aperture = 1
        win_sigma = 4.
        histogram_norm_type = 0
        l2_hys_threshold = 2.0000000000000001e-01
        gamma_correction = 0
        nlevels = 64
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, no_bins, deriv_aperture, win_sigma,
                                histogram_norm_type, l2_hys_threshold, gamma_correction, nlevels)

        win_stride = (8, 8)
        padding = (8, 8)
        locations = ((10, 20),)
        hist = hog.compute(image, win_stride, padding, locations)
        hist = hist.ravel()

        return hist


class SiftVectorMapping(VectorMapping):

    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()
        VectorMapping.__init__(self)

    def get_sift_from_image(self, _image_path):
        vector_size = 32

        _image_greyscale = self.get_greyscale_for_image(_image_path)
        key_points = self.sift.detect(_image_greyscale)

        key_points.sort(key=lambda x: x.response, reverse=True)

        key_points, descriptors = self.sift.compute(_image_greyscale, key_points[:vector_size])
        descriptors = descriptors.ravel()

        no_features = (vector_size * 128)
        # Making descriptor of same size - 128 is arbitrary value
        if descriptors.size < no_features:
            # if we have less the 128 descriptors then just adding zeros at the
            # end of our feature vector
            m=np.mean(descriptors)
            temp=np.asarray([m]*(no_features-descriptors.size))

            descriptors = np.concatenate([descriptors, temp])

        #if descriptors.size < no_features:
        #    descriptors = np.concatenate([descriptors, np.zeros(no_features - descriptors.size)])

        return descriptors
