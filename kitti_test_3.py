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

TEST_DIR='/home/lenovo/tensorflow3/test/ssd_mobilenet_v1_coco_2018_01_28/testing/image_3/'
RESULT_DIR='/home/lenovo/tensorflow3/test/ssd_mobilenet_v1_coco_2018_01_28/result/'

# Download pre-train SSD-MobileNet model from
# http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
# What model to download.
MODEL_NAME = 'object_detection/ssd_kitti'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('object_detection', 'data', 'kitti_map.pbtxt')

NUM_CLASSES = 3


    
def detect_objects(image_np, sess, detection_graph,category_index):
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
    out = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
   
    
    
    (boxes, scores, classes, num_detections) = out
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
       np.squeeze(boxes),
       np.squeeze(classes).astype(np.int32),
       np.squeeze(scores),
       category_index,
       use_normalized_coordinates=True,
       line_thickness=8)
    centerx = 0
    centery = 0
    key = ''   
    rows = image_np.shape[0]
    cols = image_np.shape[1]
    num_detections = int(num_detections)

    for i in range(num_detections):
         classId = (out[2][0][i])
         score = float(out[1][0][i])
         bbox = [float(v) for v in out[0][0][i]]
         if score > 0.5:
             x = bbox[1] * cols
             y = bbox[0] * rows
             right = bbox[3] * cols
             bottom = bbox[2] * rows
             #cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=1)
             cv2.circle(image_np, ( int((x+right)/2), int((y + bottom)/2) ), 8, (255, 255, 255), thickness=1)
             centerx = int((x+right)/2)
             centery = int((y + bottom)/2)
                       
            
             # Get and print distance value in mm at the center of the object
             # We measure the distance camera - object using Euclidean distance

             #err, point_cloud_value = point_cloud.get_value(centerx, centery)
             
             
             #distance = depth[centery][centerx]
             #distance =  math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
             #                   point_cloud_value[1] * point_cloud_value[1] +
             #                    point_cloud_value[2] * point_cloud_value[2])
             #error = math.fabs((distance - distance_pc))*100/distance
             #print(depth.shape)
             #print(str(distance) + ';'+ str(distance_pc) +';'+str(error))
             
             #if not np.isnan(distance) and not np.isinf(distance):
             #    distance = str(distance) #round() for int
             ##    #print("object {0} Distance to Camera at ({1}, {2}): {3} mm\n".format(i,x, y, distance))
             classes = np.squeeze(classes).astype(np.int32)
             if classes[i] in category_index.keys():
                 class_name = category_index[classes[i]]['name']
             else:
                 class_name = 'N/A'
             display_str = str(class_name)
             cv2.putText(image_np, "{0}".format(display_str),(int(centerx), int(centery)), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 255, 255),thickness=2,lineType=2)
                            
                 # Increment the loop
                            
            # else:
            #     print("Can't estimate distance at this position, move the camera\n")
            # sys.stdout.flush()

    t2 = cv2.getTickCount()
    print((t2 - t1)/cv2.getTickFrequency())
    return image_np


if __name__ == '__main__':
    # This is needed since the notebook is stored in the object_detection folder.

    #video_capture = cv2.VideoCapture(0)
    #if not video_capture.isOpened():
    #    print('No video camera found')
    #    exit()
  ##_____________________Ven___Displaying the fps________________________##
        
         # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
     
    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.
     
    #if int(major_ver)  < 3 :
    #    fps = video_capture.get(cv2.cv.CV_CAP_PROP_FPS)
    #    print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
   # else :
     #   fps = video_capture.get(cv2.CAP_PROP_FPS)
     #   print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
     
 
    # Number of frames to capture
    #num_frames = 120;
     
     
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

            for filename in os.listdir(TEST_DIR):
                img_name = os.path.join(TEST_DIR,filename)
                frame = cv2.imread(img_name)
                #frame = cv2.imread(TEST_DIR+'00'+str(i)+'.png')
                #height, width, channels = frame_org.shape
                #frame = frame_org[0:height, 0:(int(width/2))]
                #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result_rgb = detect_objects(frame, sess, detection_graph,category_index)
                print(img_name)
                #result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

                cv2.imshow('image', result_rgb)
                cv2.imwrite(RESULT_DIR+filename,result_rgb)

                #if cv2.waitKey(1) & 0xFF == ord('q'): ##Ven___Added to break the loop
                #    break

    #video_capture.release()
cv2.destroyAllWindows()

