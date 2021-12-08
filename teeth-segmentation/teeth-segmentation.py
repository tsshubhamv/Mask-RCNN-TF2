"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 teeth-segmentation.py train --dataset=/path/to/teeth/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 teeth-segmentation.py train --dataset=/path/to/teeth/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 teeth-segmentation.py train --dataset=/path/to/teeth/dataset --weights=imagenet

    # Apply color splash to an image
    python3 teeth-segmentation.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 teeth-segmentation.py splash --weights=last --video=<URL or path to file>
"""
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


class TeethConfig(Config):
    """
  Configuration for training on teeth dataset.
  Derives from the base Config class and overrides some values.
  """
    # Give the configuration a recognizable name
    NAME = "teeth"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + teeth

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 85% confidence
    DETECTION_MIN_CONFIDENCE = 0.85


############################################################
#  Dataset
############################################################


class TeethDataset(utils.Dataset):
    def load_teeth(self, dataset_dir, subset):
        """Load a subset of the Teeth dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or vals
        """
        # Add classes. We have only one class to add.
        self.add_class("teeth", 1, "teeth")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Handle annotations of teeth dataset images

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a teeth dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "teeth":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"],
                         len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        