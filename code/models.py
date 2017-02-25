import pickle
import numpy as np

class Model:
    """
    The model that will keep track of all the layers and their params
    """

    def __init__(self, name=""):
        self.name = name
        self.layers = []
        self.params = []

    def add_layer(self, layer):
        self.layers.append(layer)
        for p in layer.params:
            self.params.append(p)

    def print_layers(self):
        for layer in self.layers:
            print(layer.name)

    def get_params(self):
        return self.params

    def print_params(self):
        total_params = 0
        for p in self.params:
            curr_params = np.prod(np.shape(p.get_value()))
            total_params += curr_params
            print("{} ({})".format(p.name, curr_params))

    def save_params(self, file_name):
        params = {}
        for p in self.params:
            params[p.name] = p.get_value()
        pickle.dump(params, open(file_name, 'wb'))

    def load_params(self, file_name):
        params = pickle.load(open(file_name, 'rb'))
        for p in self.params:
            p.set_value(params[p.name])
