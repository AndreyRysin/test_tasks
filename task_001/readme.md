# Description

The task includes two problems.
* Binary classification problem. Recognizing the kind of an animal: a cat or a dog.
* Regression problem. Finding coordinates of the bounding box around the animal's face.

The loss function is the custom one. It is the combination of BCE loss (for classification) and Completed IoU loss (for regression).
Implementation of the Completed IoU loss is based on ideas from the article https://arxiv.org/pdf/1911.08287.pdf

There are several solutions of the task:
* two custom models (pytorch);
* one transfer learning model (pytorch);
* one custom model without any DL framework (numpy).

Augmentation is implemented with numpy.

# Essential

* deep learning (computer vision)
* ResNet50
* custom loss function
* augmentation

# Stack

* numpy
* pytorch
* matplotlib
* opencv
* numba
