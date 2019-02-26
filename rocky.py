import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from PIL import Image

import cv2
import time


# This is needed since the notebook is stored in the object_detection folder.
# sys.path.append("..")
from object_detection.utils import ops as utils_ops

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

#if tf.__version__ < '1.4.0':
#    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

print("OpenCV version :  {0}".format(cv2.__version__))

# Should run under docker container from tensorflow_object_detection
ROOT = '/opt/tf_model/research/object_detection/'

# Download pre-train SSD-MobileNet model from
# http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
# What model to download.
MODEL_NAME = 'object_detection/ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    t1 = cv2.getTickCount()
    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    t2 = cv2.getTickCount()
    print((t2 - t1) / cv2.getTickFrequency())

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    return image_np


if __name__ == '__main__':
    # This is needed since the notebook is stored in the object_detection folder.

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print('No video camera found')
        exit()
  ##_____________________Ven___Displaying the fps________________________##
        
         # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
     
    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.
     
    if int(major_ver)  < 3 :
        fps = video_capture.get(cv2.cv.CV_CAP_PROP_FPS)
        print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
     
 
    # Number of frames to capture
    num_frames = 120;
     
     
    #print ("Capturing {0} frames".format(num_frames))
 
    # Start time
    #start = time.time()
     
    # Grab a few frames
    #for i in range(0, num_frames) :
    #   ret, frame = video_capture.read()
     
    # End time
    #end = time.time()
 
    # Time elapsed
    #seconds = end - start
    #print ("Time taken : {0} seconds".format(seconds))
 
    # Calculate frames per second
    #fps  = num_frames / seconds;
    #print ("Estimated frames per second : {0}".format(fps));
    ##___________________________________________________________##

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)

    category_index = label_map_util.create_category_index(categories)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            while True:
                ret, frame_org = video_capture.read()
                height, width, channels = frame_org.shape
                frame = frame_org[0:height, 0:(int(width/2))]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result_rgb = detect_objects(frame_rgb, sess, detection_graph)

                result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

                cv2.imshow('Video', result_bgr)

                if cv2.waitKey(1) & 0xFF == ord('q'): ##Ven___Added to break the loop
                    break

    video_capture.release()
cv2.destroyAllWindows()

