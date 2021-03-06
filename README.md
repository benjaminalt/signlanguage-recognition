# signlanguage-recognition

A set of neural network models and visualizations developed for the practical course on Neural Networks at the Karlsruhe Institute of Technology.

## Installation

### Clone the repository with submodules
```
git clone --recurse-submodules git@github.com:benjaminalt/signlanguage-recognition.git
```

### Install dependencies
We recommend to use the Python distribution [Anaconda](https://www.anaconda.com/) and a custom virtual environment:
```
conda create --name signlanguage-env
source activate signlanguage-env
conda install graphviz python-graphviz matplotlib ipython opencv pandas scipy
conda install pytorch torchvision -c pytorch
cd signlanguage-recognition
pip install -r requirements.txt
```

## Training & testing

Before running anything from the repository, make sure the right virtual environment is active:
```
source activate signlanguage-env
```

### Training a model
To train the neural network, simply run `main.py`:
```
cd signlanguage-recognition/code
python main.py train
```
This will create a timestamped directory in `signlanguage-recognition/results`, which contains

- A CSV file with the parameters of the neural network and the training results
- A visual representation of the neural network
- A matplotlib plot of training and test loss and accuracy
- A pytorch (.pt) file containing the weights of the model

You can tweak the parameters of the model by adapting `code/options.py`.

### Testing a model
```
cd signlanguage-recognition/code
python main.py test --weights path/to/weights.pt --data path/to/data.csv
```
This will construct a model with the given weights and use it to classify the provided data.
The resulting accuracy will be printed.

### Visualizing a model
```
cd signlanguage-recognition/code
python main.py visualize --weights path/to/weights.pt --data path/to/data.csv
```
This will generate a [Grad-CAM](https://arxiv.org/abs/1610.02391) visualization for all convolutional layers of the model for all input images. A lot of images will be generated, so choose your input data wisely (use for example `data/sign_mnist_test_small.csv`). The results will be in a generated folder under `results`.
