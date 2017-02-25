import theano
import theano.tensor as T

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
        self.filter = Conv2DHelper.build_filter(filter_shape, mask_type)

        # Setup bias
        self.bias = Conv2DHelper.build_bias(filter_shape)

        conv_out = T.nnet.conv2d(input_tensor, self.filter)
        self.output = conv_out + self.bias

        if activation is not None:
            if activation == 'relu':
                self.output = T.nnet.relu(self.output)
            elif activation == 'tanh':
                self.output = T.tanh(self.output)
            else:
                raise Exception("Activation: {}, is not
                        implemented".format(activation))

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
        pass

    @classmethod
    def build_bias(cls, filter_shape):
        """
        Builds the bias of the 2D conv layer

        filter_shape: tuple
            A tuple that describes the shape of the filter
            (output_channels, input_channels, filter_width, filter_height)
        """
        pass

