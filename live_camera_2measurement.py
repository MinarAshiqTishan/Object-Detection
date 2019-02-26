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
import pptk

from object_detection.utils import ops as utils_ops

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util


#prefetch vhd files in windows hold the instruction in vhdl for file execution, fetch cycle opcodes,

print("OpenCV version :  {0}".format(cv2.__version__))
num_frames = 120

MODEL_NAME = 'object_detection/ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('object_detection', 'data', 'mscoco_label_map.pbtxt')

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
         if score > 0.3:
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
             
             
             #distance = depth[centery][centerx]
             distance =  math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                point_cloud_value[1] * point_cloud_value[1] +
                                 point_cloud_value[2] * point_cloud_value[2])
             #error = math.fabs((distance - distance_pc))*100/distance
             #print(depth.shape)
             #print(str(distance) + ';'+ str(distance_pc) +';'+str(error))
             
             if not np.isnan(distance) and not np.isinf(distance):
                 distance = str(distance) #round() for int
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
    print((t2 - t1)/cv2.getTickFrequency())
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
    print('depthmapZ;pointcloudZ;error_percentage')
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config=config) as sess:
            print("Running...")
            init = zcam.PyInitParameters()
            init.camera_resolution = sl.PyRESOLUTION.PyRESOLUTION_VGA   #'PyRESOLUTION_HD1080', 'PyRESOLUTION_HD2K', 'PyRESOLUTION_HD720', 'PyRESOLUTION_LAST', 'PyRESOLUTION_VGA'
            init.depth_mode = sl.PyDEPTH_MODE.PyDEPTH_MODE_ULTRA  # Use ULTRA depth mode
            init.coordinate_units = sl.PyUNIT.PyUNIT_MILLIMETER  # Use milliliter units (for depth measurements)
            init.camera_buffer_count_linux = 1
            
            cam = zcam.PyZEDCamera()
            if not cam.is_opened():
                print("Opening ZED Camera...")
            status = cam.open(init)
            if status != tp.PyERROR_CODE.PySUCCESS:
                print(repr(status))
                exit()

            runtime = zcam.PyRuntimeParameters()
            runtime.sensing_mode = sl.PySENSING_MODE.PySENSING_MODE_FILL  # Use STANDARD sensing mode
            cam_params = cam.get_camera_information()
            print(cam_params.calibration_parameters.left_cam.fx)
            print(cam_params.calibration_parameters.left_cam.cx)
            print(cam_params.calibration_parameters.left_cam.cy)
            print(cam_params.calibration_parameters.left_cam.d_fov)
            print(cam_params.calibration_parameters.left_cam.disto)
            print(cam_params.calibration_parameters.left_cam.fy)
            print(cam_params.calibration_parameters.left_cam.h_fov)
            print(cam_params.calibration_parameters.left_cam.image_size)
            print(cam_params.calibration_parameters.left_cam.v_fov)
            #682.6981201171875
            #648.062255859375
            #340.35528564453125
            #94.17412567138672
            #[0.0, 0.0, 0.0, 0.0, 0.0]
            #682.6981201171875
            #86.30469512939453
            #{'width': 1280, 'height': 720} #720
            #55.60874557495117
            #{'width': 672, 'height': 376} #vga
            #{'width': 1920, 'height': 1080} #1080

            #fx = 698.581 #720p
            fx = 349.29 #vga
            #fx = 1397.16 #1080
            baseline = 63

            mat = core.PyMat()   
            image = core.PyMat()
            depth = core.PyMat()
            point_cloud = core.PyMat()
            confidence = core.PyMat()
            depth_for_display = core.PyMat()
            disparity = core.PyMat()
           
            while True:

                err = cam.grab(runtime)  
              
                if err == tp.PyERROR_CODE.PySUCCESS:

                    cam.retrieve_image(image, sl.PyVIEW.PyVIEW_LEFT)
                    #cam.retrieve_measure(depth, sl.PyMEASURE.PyMEASURE_DEPTH)
                    #cam.retrieve_measure(disparity, sl.PyMEASURE.PyMEASURE_DISPARITY)
                    cam.retrieve_measure(point_cloud, sl.PyMEASURE.PyMEASURE_XYZRGBA)
                    #cam.retrieve_image(confidence, sl.PyVIEW.PyVIEW_CONFIDENCE) #for viewing purpose only, for measurement, change VIEW to MEASURE
                    #cam.retrieve_image(depth_for_display, sl.PyVIEW.PyVIEW_DEPTH)  
                    ######calculate z value of each pixel in disparity map
                    #d = ((disparity.get_data()).squeeze()).astype(np.float32)
                    #dr = cv2.resize(d,(672,376)) 
                    #Z = (fx*baseline)/(disparity.get_data()*-1) 
                    #this depth Z is calculated using the formula -fx_l*baseline/disparity
                    #the measure_depth has uses the same formula to measure depth from disparity
                    # the view_depth uses -1*disparity.get_data()/100 this to view depth
                    # Read and preprocess an image.  

                    #depth matrix
                    #depth_data = cv2.resize(( (depth.get_data()).squeeze()).astype(np.float32),(1280,720))  
                    #depth_data = depth.get_data()
                    img = image.get_data()
                    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    result_rgb = detect_objects(frame_rgb, sess, detection_graph,image,depth,point_cloud,category_index)
                    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)                            
                    cv2.imshow('Video', result_bgr)
                    #cv2.imshow('confidence', confidence.get_data())
                    #cv2.imshow('depth from sdk', depth_for_display.get_data())
                    #cv2.imshow('depth for display  calculated from disparity', -1*disparity.get_data()/100)
                    #cv2.imshow('depth calculated from disparity', Z/1280)
                                       
                      
                    #print(depth.get_data().shape)
                   
                if cv2.waitKey(1) & 0xFF == ord('q'): ##Ven___Added to break the loop
                    break
                
            cam.close()
#video_capture.release()
print("end")
cv2.destroyAllWindows()

