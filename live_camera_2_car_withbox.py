import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import math

import pyzed.camera as zcam
import pyzed.types as tp
import pyzed.core as core
import pyzed.defines as sl

from collections import defaultdict
from io import StringIO
from PIL import Image

import cv2
import time


from object_detection.utils import ops as utils_ops

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util




print("OpenCV version :  {0}".format(cv2.__version__))
num_frames = 120

MODEL_NAME = 'object_detection/ssd_kitti'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('object_detection', 'data', 'kitti_map.pbtxt')

NUM_CLASSES = 90

camera_settings = sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_BRIGHTNESS
str_camera_settings = "BRIGHTNESS"

#to disable GPU
config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

#to allocate subset of available gpu memory as needed by process
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True


def detect_objects(image_np, sess, detection_graph,image,depth,point_cloud,category_index):
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
         if score > 0.4:
             x = bbox[1] * cols
             y = bbox[0] * rows
             right = bbox[3] * cols
             bottom = bbox[2] * rows
             #cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=1)
             cv2.circle(image_np, ( int((x+right)/2), int((y + bottom)/2) ), 8, (125, 155, 21), thickness=1)
             centerx = int((x+right)/2)
             centery = int((y + bottom)/2)
                       
            
             # Get and print distance value in mm at the center of the object
             # We measure the distance camera - object using Euclidean distance

             err, point_cloud_value = point_cloud.get_value(centerx, centery)

             distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                  point_cloud_value[1] * point_cloud_value[1] +
                                  point_cloud_value[2] * point_cloud_value[2])

             if not np.isnan(distance) and not np.isinf(distance):
                 distance = round(distance)
                 #print("object {0} Distance to Camera at ({1}, {2}): {3} mm\n".format(i,x, y, distance))
                 classes = np.squeeze(classes).astype(np.int32)
                 if classes[i] in category_index.keys():
                     class_name = category_index[classes[i]]['name']
                 else:
                     class_name = 'N/A'
                 display_str = str(class_name)
                 cv2.putText(image_np, "{0}:{1} mm".format(display_str,distance),(int(centerx), int(centery)), cv2.FONT_HERSHEY_SIMPLEX, 1, (125, 155, 21),thickness=2,lineType=2)
                            
                 # Increment the loop
                            
             else:
                 print("Can't estimate distance at this position, move the camera\n")
             sys.stdout.flush()

    t2 = cv2.getTickCount()
    print(cv2.getTickFrequency()/(t2 - t1))
    return image_np


if __name__ == '__main__':

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)

    category_index = label_map_util.create_category_index(categories)
    
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config=config) as sess:
            print("Running...")
            init = zcam.PyInitParameters()
            init.camera_resolution = sl.PyRESOLUTION.PyRESOLUTION_VGA    #'PyRESOLUTION_HD1080', 'PyRESOLUTION_HD2K', 'PyRESOLUTION_HD720', 'PyRESOLUTION_LAST', 'PyRESOLUTION_VGA'
            init.depth_mode = sl.PyDEPTH_MODE.PyDEPTH_MODE_PERFORMANCE  # Use PERFORMANCE depth mode
            init.coordinate_units = sl.PyUNIT.PyUNIT_MILLIMETER  # Use milliliter units (for depth measurements)
            init.camera_buffer_count_linux = 1
            init.camera_fps = 100
            cam = zcam.PyZEDCamera()
            if not cam.is_opened():
                print("Opening ZED Camera...")
            status = cam.open(init)
            if status != tp.PyERROR_CODE.PySUCCESS:
                print(repr(status))
                exit()

            runtime = zcam.PyRuntimeParameters()
            runtime.sensing_mode = sl.PySENSING_MODE.PySENSING_MODE_STANDARD  # Use STANDARD sensing mode

            mat = core.PyMat()   
            image = core.PyMat()
            depth = core.PyMat()
            point_cloud = core.PyMat()
           
            while True:

                err = cam.grab(runtime)  
              
                if err == tp.PyERROR_CODE.PySUCCESS:

                    cam.retrieve_image(image, sl.PyVIEW.PyVIEW_LEFT)
                    cam.retrieve_measure(depth, sl.PyMEASURE.PyMEASURE_DEPTH)
                    cam.retrieve_measure(point_cloud, sl.PyMEASURE.PyMEASURE_XYZRGBA)
                    # Read and preprocess an image.                
                    img = image.get_data()
                    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    result_rgb = detect_objects(frame_rgb, sess, detection_graph,image,depth,point_cloud,category_index)
                    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)                            
                    cv2.imshow('Video', result_bgr)
    
                if cv2.waitKey(1) & 0xFF == ord('q'): ##Ven___Added to break the loop
                    break
            cam.close()
#video_capture.release()
print("end")
cv2.destroyAllWindows()

