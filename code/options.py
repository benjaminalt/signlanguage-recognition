from itertools import product
import os
import time
import torch


class Options:

    def __init__(self):
        self._root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)  # absolute path
        self._data_dir = os.path.join(self._root_dir, "data")
        self._results_dir = os.path.join(self._root_dir, "results")
        self._output_dir = os.path.join(self._results_dir, time.strftime("%Y%m%d-%H%M%S"))

        # enables grid search
        self.gridSearch = False
        self.interactiveGUI = False

        # settings for single training
        self.use_cuda              = torch.cuda.is_available()
        self.batch_size            = 256
        self.n_epochs              = 10
        self.learning_rate         = 0.001
        self.weight_decay          = 0.0001
        self.shuffleTestData       = False
        self.shuffleTrainData      = True
        self.dropout_probability_1 = 0.2
        self.dropout_probability_2 = 0.2
        self.dropout_probability_3 = 0.2
        self.use_batchnorm_cnn     = True
        self.use_batchnorm_linear  = False

        # changes of above setting during grid search
        self._variety = {
            "batch_size":            [128, 64, 32, 16, 8, 1],
            "learning_rate":         [0.1, 0.001, 0.0001],
            "weight_decay":          [0.001, 0.0005, 0.0001],
            "dropout_probability_1": [0.0, 0.25, 0.5],
            "dropout_probability_2": [0.0, 0.25, 0.5],
            "use_batchnorm_cnn":     [True, False],
            "use_batchnorm_linear":  [True, False]
        }

    def iter(self):
        listOfVariations = []

        for var_name, variety in self._variety.items():
            if hasattr(self, var_name):
                listOfVariations.append([(var_name, x) for x in variety])
            else:
                print("WARNING: Option.'{}' is no valid attribute and will be ignored!".format(var_name))

        numIterations = len(list(product(*listOfVariations)))
        print("INFO: Grid search will test {} configurations.".format(numIterations))

        for config in product(*listOfVariations):
            for (var_name, val) in config:
                setattr(self, var_name, val)
            yield self

    def __str__(self):
        return 'Options: ' + ', '.join(str(e) + "=" + str(getattr(self, e)) for e in self.var_names())

    def var_names(self):
        return [x for x in vars(self) if not x.startswith("_")]

    def values(self):
        return dict(zip(self.var_names(), [getattr(self, e) for e in self.var_names()]))

    def root_dir(self):
        return self._root_dir

    def data_dir(self):
        return self._data_dir

    def results_dir(self):
        return self._results_dir

    def output_dir(self):
        return self._output_dir

    def root_path(self, relative_path):
        return os.path.join(self._root_dir, relative_path)

    def root_relpath(self, absolute_path):
        return os.path.relpath(absolute_path, self._root_dir)

    def data_path(self, relative_path):
        return os.path.join(self._data_dir, relative_path)

    def output_path(self, relative_path):
        return os.path.join(self._output_dir, relative_path)
