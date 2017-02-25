import theano
import theano.tensor as T
import numpy as np

class Layer:
    """
    Base layer class

    name: string
        A descriptive name of the layer
    """
    def __init__(self, name=""):
        self.name = name
        self.params = []
        self.output = None

    def get_params(self):
        return self.params

    def get_output(self):
        return self.params

class Conv2DLayer(Layer):
    """
    Basic 2D convolution layer
    """
    def __init__(self, input_tensor, filter_shape, mask_type=None,
            activation=None):
        super("Conv2D")

        # Setup conv filters
        conv_filter = Conv2DHelper.build_filter(filter_shape, mask_type)

        # Setup bias
        bias = Conv2DHelper.build_bias(filter_shape)

        conv_out = T.nnet.conv2d(input_tensor, conv_filter)
        self.output = conv_out + bias

        if activation is not None:
            if activation == 'relu':
                self.output = T.nnet.relu(self.output)
            elif activation == 'tanh':
                self.output = T.tanh(self.output)
            else:
                raise Exception("Activation: {}, is not implemented".format(activation))

        self.params.extend([conv_filter, bias])

class Conv2DHelper:
    """
    Helper for building the conv layer
    """
    @classmethod
    def build_filter(cls, filter_shape, mask_type):
        """
        Builds the filter of the 2D conv layer

        filter_shape: tuple
            A tuple that describes the shape of the filter
            (output_channels, input_channels, filter_width, filter_height)
        mask_type: string
            Conv filter mask
        """
        fan_in = np.prod(filter_shape[1:])
        fan_out = np.prod(filter_shape[0:2:-1])
        w_std = np.sqrt(2.0 / (fan_in + fan_out))

        filter_init = cls.uniform(w_std, filter_shape)

        if mask_type is not None:
            filter_init *= floatX(np.sqrt(2.))

        conv_filter = theano.shared(filter_init, name='filters')

        if mask_type is not None:
            mask = np.ones(
                    filter_shape,
                    dtype=theano.config.floatX)

            for i in range(filter_shape[2]):
                for j in range(filter_shape[3]):
                    if i > filter_shape[2]//2:
                        mask[:,:,i,j] = floatX(0.0)

                    if i == filter_shape[2]//2 and j >
                    filter_shape[3]//2:
                        mask[:,:,i,j] = floatX(0.0)

            if mask_type == 'a':
                mask[:,:,filter_shape[2]//2,filter_shape[3]//2] =
                floatX(0.0)

            conv_filter = conv_filter*mask

        return conv_filter


   @classmethod
   def build_bias(cls, filter_shape):
       """
        Builds the bias of the 2D conv layer

        filter_shape: tuple
            A tuple that describes the shape of the filter
            (output_channels, input_channels, filter_width, filter_height)
        """
        bias_init = np.zeros(filter_shape[0]).astype(theano.config.floatX)
        return theano.shared(bias_init, name="bias")

    @classmethod
    def uniform(cls, stdev, size):
        """
        Uniform distribution with the given stdev and size
        """
        return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
                ).astype(theano.config.floatX)

