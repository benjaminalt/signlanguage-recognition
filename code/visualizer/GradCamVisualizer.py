import numpy as np
import torch
import argparse
import os, sys
from PIL import Image
import cv2

sys.path.append(os.getcwd())
from nets.CNNs import CNN_5
from datasets.SignMNISTDataset import SignMNISTDataset
import options

gradcam_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, "pytorch-cnn-visualizations", "src"))
sys.path.append(gradcam_dir)
import gradcam
import misc_functions as gradcam_misc

class GradCamVisualizer(object):
    def __init__(self, options):
        self.options = options

    def visualize(self, model, target_layer, input_img, label):
        grad_cam = gradcam.GradCam(model, target_layer=target_layer)
        preprocessed_img = self._preprocess_img(input_img)
        preprocessed_img.save(self.options.output_path("gradcam_input_img.jpg"))
        inputs = torch.from_numpy(np.expand_dims(input_img,0))
        inputs = inputs.type(torch.FloatTensor)
        # Generate cam mask
        cam = grad_cam.generate_cam(inputs, label)
        # Save mask
        gradcam_misc.save_class_activation_images(preprocessed_img, cam, self.options.output_path("gradcam_{}".format(target_layer)))
        return

    def _preprocess_img(self, img):
        reshaped = np.reshape(img, (28,28))
        denormalized = np.interp(reshaped, [0,1], [0,255])
        resized = cv2.resize(denormalized, (224,224))
        return Image.fromarray(resized).convert("RGB")

def main(args):
    if not os.path.exists(args.weights) or not os.path.isfile(args.weights):
        raise ValueError("Invalid model path: {}".format(args.weights))
    opts = options.Options()
    model = CNN_5(opts)
    model.load_state_dict(torch.load(args.weights))

    # Load input data
    os.makedirs(opts.output_dir(), exist_ok=True)

    train_path = opts.data_path("sign_mnist_train.csv")
    train_set = SignMNISTDataset(opts, train_path)

    visualizer = GradCamVisualizer(opts)
    for n_layer in range(6):
        visualizer.visualize(model, n_layer, train_set[0]["image"], train_set[0]["label"])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=str)
    main(parser.parse_args())
