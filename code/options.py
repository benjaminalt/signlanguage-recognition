from itertools import product
import os, time

RESULT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, "results")


class Options:

    def __init__(self):
        self._output_dir = os.path.join(RESULT_DIR, time.strftime("%Y%m%d-%H%M%S"))

        # enables grid search
        self.gridSearch = False
        self.interactiveGUI = False

        # settings for single training
        self.batch_size            = 32
        self.n_epochs              = 1
        self.learning_rate         = 0.001
        self.weight_decay          = 0.0001
        self.shuffleTestData       = False
        self.shuffleTrainData      = True
        self.dropout_probability_1 = 0.3
        self.dropout_probability_2 = 0.3

        # changes of above setting during grid search
        self._variety = {
            "batch_size":            [1, 8, 16, 32, 64, 128],
            "learning_rate":         [0.1, 0.001, 0.0001],
            "weight_decay":          [0.001, 0.0005, 0.0001],
            "dropout_probability_1": [0.0, 0.25, 0.5],
            "dropout_probability_2": [0.0, 0.25, 0.5]
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
    
    def output_dir(self):
        return self._output_dir


