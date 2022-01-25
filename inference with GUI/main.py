import os
import random
import sys
import cv2
from PyQt5.QtGui import QImage
from keras.preprocessing import image
from keras import backend as K
from imageio import imread
from keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt
from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from PyQt5 import QtWidgets, QtGui

from GUI import Ui_MainWindow


class Main:
    def __init__(self):
        self.image_path = ""
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)

        self.img_height = 300
        self.img_width = 300

        self.classes = ['background', 'diseased', 'healthy']

        K.clear_session()
        self.model = ssd_300(image_size=(self.img_height, self.img_width, 3),
                             n_classes=3,
                             mode='inference',
                             l2_regularization=0.0005,
                             scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
                             # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                             aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                                      [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                      [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                      [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                      [1.0, 2.0, 0.5],
                                                      [1.0, 2.0, 0.5]],
                             two_boxes_for_ar1=True,
                             steps=[8, 16, 32, 64, 100, 300],
                             offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                             clip_boxes=False,
                             variances=[0.1, 0.1, 0.2, 0.2],
                             normalize_coords=True,
                             subtract_mean=[123, 117, 104],
                             swap_channels=[2, 1, 0],
                             confidence_thresh=0.9,
                             iou_threshold=0.75,
                             top_k=200,
                             nms_max_output_size=400)

        if os.path.exists("Mango.h5"):
            weights_path = 'Mango.h5'
        else:
            sys.exit()

        self.model.load_weights(weights_path, by_name=True)
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        self.model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

        self.ui.pushButton.clicked.connect(self.upload_img)
        self.ui.pushButton_2.clicked.connect(self.predict)

    def upload_img(self):
        # open the dialogue box to select the file
        options = QtWidgets.QFileDialog.Options()

        # open the Dialogue box to get the images paths
        image = QtWidgets.QFileDialog.getOpenFileName(caption="Select the image", directory="",
                                                      filter="Image Files (*.jpg);;Image Files (*.png);;All files (*.*)",
                                                      options=options)

        # If user don't select any image then return without doing any thing
        if image[0] == '':
            self.image_path = image[0]
            return

        self.ui.label_3.setPixmap(QtGui.QPixmap(image[0]))
        self.image_path = image[0]

    def predict(self):

        if self.image_path == "":
            return

        orig_images = []  # Store the images here.
        input_images = []

        orig_images.append(cv2.resize(imread(self.image_path), (300, 300)))
        img = image.load_img(self.image_path, target_size=(self.img_height, self.img_width))
        img = image.img_to_array(img)

        input_images.append(img)
        input_images = np.array(input_images)

        y_pred = self.model.predict(input_images)

        confidence_threshold = 0.5
        y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]
        colors = []

        for i in range(len(self.classes)):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            colors.append((r, g, b))

        for box in y_pred_thresh[0]:
            # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
            xmin = int(box[2] * orig_images[0].shape[1] / self.img_width)
            ymin = int(box[3] * orig_images[0].shape[0] / self.img_height)
            xmax = int(box[4] * orig_images[0].shape[1] / self.img_width)
            ymax = int(box[5] * orig_images[0].shape[0] / self.img_height)
            color = colors[int(box[0])]
            label = '{}: {:.2f}'.format(self.classes[int(box[0])], box[1])

            cv2.rectangle(orig_images[0], (xmin, ymin), (xmax, ymax), color, 3, 3)
            cv2.putText(orig_images[0], label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        result = orig_images[0].copy()
        height, width, channel = result.shape
        step = channel * width
        qImg = QImage(result.data, width, height, step, QImage.Format_RGB888)
        self.ui.label_4.setPixmap(QtGui.QPixmap(qImg))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    obj = Main()
    obj.MainWindow.show()
    sys.exit(app.exec_())
