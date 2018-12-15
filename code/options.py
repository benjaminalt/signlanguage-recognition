from itertools import product


class Options:

    def __init__(self):
        # enables grid search
        self.gridSearch = False
        self.interativeGUI = True

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
        var_names = [x for x in vars(self) if not x.startswith("_")]
        return 'Options: ' + ', '.join(str(e) + "=" + str(getattr(self, e)) for e in var_names)

    def toFileName(self):
        var_names = [x for x in vars(self) if not x.startswith("_") and x in self._variety]
        return 'Output_' + '_'.join(str(e) + "_" + str(getattr(self, e)).replace(".", "_") for e in var_names) + '.png'



