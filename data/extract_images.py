import argparse
import os
import numpy as np
from PIL import Image

def save_image(img, output_dir):
    seqnr = len(os.listdir(output_dir))
    img = np.delete(img, 0)
    img = np.reshape(img, (28,28))
    pil_img = Image.fromarray(img)
    pil_img.save(os.path.join(output_dir, "{}.png".format(seqnr)))

def main(args):
    if not os.path.isfile(args.dataset):
        raise ValueError("Invalid dataset: {}".format(args.dataset))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    data = np.genfromtxt(args.dataset, dtype="uint8", skip_header=1, delimiter=",")
    print("Extracting {} images to {}...".format(data.shape[0], args.output_dir))
    np.apply_along_axis(lambda img: save_image(img, args.output_dir), 1, data)
    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("output_dir", type=str)
    main(parser.parse_args())