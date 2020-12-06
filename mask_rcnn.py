import glob
import os
import cv2
import imutils
import numpy as np
import pandas as pd
from mrcnn import visualize
from mrcnn.config import Config
from mrcnn import model as modellib
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical

images_path = 'C:/Users/S&M/Desktop/data/'
images = glob.glob(images_path + "*.png")
images.sort()
data = []
width = 200
height = 200
for img in images:
    image = cv2.imread(img)
    image = cv2.resize(image, (width, height))
    data.append(image)

data = np.array(data) / 255.0

labels_csv = pd.read_csv(r'C:/Users/S&M/Desktop/data_label.csv')
labels = list(labels_csv.labels)
# data = np.array(data) / np.max(data)
labels = np.array(labels)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# load the class label names from disk, one label per line
CLASS_NAMES = open(labels).read().strip().split("\n")
# (thanks to Matterport Mask R-CNN for the method!)
hsv = [(i / len(CLASS_NAMES), 1, 1.0) for i in range(len(CLASS_NAMES))]


class SimpleConfig(Config):
    # give the configuration a recognizable name
    NAME = "coco_inference"
    # set the number of GPUs to use along with the number of images
    # per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = len(CLASS_NAMES)


# initialize the inference configuration
config = SimpleConfig()
# initialize the Mask R-CNN model for inference and then load the weights
print("[INFO] loading Mask R-CNN model...")
model = modellib.MaskRCNN(mode="inference", config=config,
                          model_dir=os.getcwd())
model.load_weights(args["weights"], by_name=True)

# load the input image, convert it from BGR to RGB channel
# ordering, and resize the image
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = imutils.resize(image, width=512)
# perform a forward pass of the network to obtain the results
print("[INFO] making predictions with Mask R-CNN...")
r = model.detect([image], verbose=1)[0]

# loop over of the detected object's bounding boxes and masks
for i in range(0, r["rois"].shape[0]):
    # extract the class ID and mask for the current detection, then
    # grab the color to visualize the mask (in BGR format)
    classID = r["class_ids"][i]
    mask = r["masks"][:, :, i]
    # visualize the pixel-wise mask of the object
    image = visualize.apply_mask(image, mask, alpha=0.5)


image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# loop over the predicted scores and class labels
for i in range(0, len(r["scores"])):
    # extract the bounding box information, class ID, label, predicted
    # probability, and visualization color
    (startY, startX, endY, endX) = r["rois"][i]
    classID = r["class_ids"][i]
    label = CLASS_NAMES[classID]
    score = r["scores"][i]
    # draw the bounding box, class label, and score of the object
    cv2.rectangle(image, (startX, startY), (endX, endY), 2)
    text = "{}: {:.3f}".format(label, score)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, 2)

cv2.imshow("Output", image)
cv2.waitKey()
