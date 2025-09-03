import numpy as np


class NumpyImage(np.ndarray):
    """
    Enhanced numpy array subclass for convenient image dimension access.

    NumpyImage extends numpy.ndarray to provide intuitive property-based access
    to array dimensions using image-oriented terminology (width, height, depth)
    instead of generic shape indexing.

    This class is particularly useful for computer vision and image processing
    workflows where you frequently need to access image dimensions.

    Attributes:
        width (int): The width of the image (shape[1]).
        height (int): The height of the image (shape[0]).
        depth (int): The depth/channels of the image (shape[2]).

    Note:
        For compatibility with libraries like OpenCV that expect regular numpy
        arrays, use the as_array() method to convert back to numpy.ndarray.
    """

    def __new__(cls, input_array):
        """
        Create a new NumpyImage instance.

        Args:
            input_array: Array-like input that can be converted to numpy array.
                        Can be a list, tuple, numpy array, or any array-like object.

        Returns:
            NumpyImage: A new NumpyImage instance viewing the input data.
        """
        obj = np.asarray(input_array).view(cls)
        return obj

    @property
    def width(self):
        """
        Get the width of the image.

        Returns:
            int: Width of the image (second dimension, shape[1]).
                 Returns 1 for 1D arrays.
        """
        return self.shape[1] if len(self.shape) > 1 else 1

    @property
    def height(self):
        """
        Get the height of the image.

        Returns:
            int: Height of the image (first dimension, shape[0]).
        """
        return self.shape[0]

    @property
    def depth(self):
        """
        Get the depth/number of channels of the image.

        Returns:
            int: Depth of the image (third dimension, shape[2]).
                 Returns 1 for 1D and 2D arrays.
        """
        return self.shape[2] if len(self.shape) > 2 else 1

    def as_array(self):
        """
        Convert back to regular numpy array for compatibility.

        This method is essential for using the NumpyImage with libraries
        like OpenCV, scikit-image, or other tools that expect standard
        numpy arrays.

        Returns:
            numpy.ndarray: A regular numpy array view of the same data.
        """
        return np.asarray(self)