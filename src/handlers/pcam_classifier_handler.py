import torch
from torchvision import transforms
from ts.torch_handler.image_classifier import ImageClassifier
import numpy as np


class PCAMClassifierHandler(ImageClassifier):
    """
    PCAm Classifier handler class. This handler extends class ImageClassifier from image_classifier.py, a
    default handler. This handler takes an image and returns the the binary prediction indicating presence of metastatic tissue.
    Here method postprocess() has been overridden while others are reused from parent class.
    """

    image_processing = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def postprocess(self, data):
        """The post process of PCAM Classification converts the predicted output response to a label.
        Args:
            data (list): The predicted output from the Inference with probabilities is passed
            to the post-process function
        Returns:
            list : A list of with an appropriate prediction.
        """
        pred = 1.0 / (1.0 + np.exp(-data.item()))
        return [int(pred > 0.5)]

        # return data.argmax(1).tolist()
