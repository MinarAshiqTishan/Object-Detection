########################################################################
#
# Copyright (c) 2017, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
    Live camera sample showing the camera information and video in real time and allows to control the different
    settings.
"""

import cv2
import pyzed.camera as zcam
import pyzed.types as tp
import pyzed.core as core
import pyzed.defines as sl
import math
import numpy as np
import sys
import tensorflow as tf

from object_detection.utils import ops as utils_ops

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

import os
PATH_TO_CKPT = '/frozen_inference_graph.pb'
PATH_TO_LABELS = 'mscoco_label_map.pbtxt'

camera_settings = sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_BRIGHTNESS
str_camera_settings = "BRIGHTNESS"
step_camera_settings = 1

NUM_CLASSES = 80



def print_camera_information(cam):
    print("Resolution: {0}, {1}.".format(round(cam.get_resolution().width, 2), cam.get_resolution().height))
    print("Camera FPS: {0}.".format(cam.get_camera_fps()))
    print("Firmware: {0}.".format(cam.get_camera_information().firmware_version))
    print("Serial number: {0}.\n".format(cam.get_camera_information().serial_number))


def print_help():
    print("Help for camera setting controls")
    print("  Increase camera settings value:     +")
    print("  Decrease camera settings value:     -")
    print("  Switch camera settings:             s")
    print("  Reset all parameters:               r")
    print("  Record a video:                     z")
    print("  Quit:                               q\n")


def settings(key, cam, runtime, mat):
    if key == 115:  # for 's' key
        switch_camera_settings()
    elif key == 43:  # for '+' key
        current_value = cam.get_camera_settings(camera_settings)
        cam.set_camera_settings(camera_settings, current_value + step_camera_settings)
        print(str_camera_settings + ": " + str(current_value + step_camera_settings))
    elif key == 45:  # for '-' key
        current_value = cam.get_camera_settings(camera_settings)
        if current_value >= 1:
            cam.set_camera_settings(camera_settings, current_value - step_camera_settings)
            print(str_camera_settings + ": " + str(current_value - step_camera_settings))
    elif key == 114:  # for 'r' key
        cam.set_camera_settings(sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_BRIGHTNESS, -1, True)
        cam.set_camera_settings(sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_CONTRAST, -1, True)
        cam.set_camera_settings(sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_HUE, -1, True)
        cam.set_camera_settings(sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_SATURATION, -1, True)
        cam.set_camera_settings(sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_GAIN, -1, True)
        cam.set_camera_settings(sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_EXPOSURE, -1, True)
        cam.set_camera_settings(sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_WHITEBALANCE, -1, True)
        print("Camera settings: reset")
    elif key == 122:  # for 'z' key
        record(cam, runtime, mat)


def switch_camera_settings():
    global camera_settings
    global str_camera_settings
    if camera_settings == sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_BRIGHTNESS:
        camera_settings = sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_CONTRAST
        str_camera_settings = "Contrast"
        print("Camera settings: CONTRAST")
    elif camera_settings == sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_CONTRAST:
        camera_settings = sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_HUE
        str_camera_settings = "Hue"
        print("Camera settings: HUE")
    elif camera_settings == sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_HUE:
        camera_settings = sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_SATURATION
        str_camera_settings = "Saturation"
        print("Camera settings: SATURATION")
    elif camera_settings == sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_SATURATION:
        camera_settings = sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_GAIN
        str_camera_settings = "Gain"
        print("Camera settings: GAIN")
    elif camera_settings == sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_GAIN:
        camera_settings = sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_EXPOSURE
        str_camera_settings = "Exposure"
        print("Camera settings: EXPOSURE")
    elif camera_settings == sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_EXPOSURE:
        camera_settings = sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_WHITEBALANCE
        str_camera_settings = "White Balance"
        print("Camera settings: WHITEBALANCE")
    elif camera_settings == sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_WHITEBALANCE:
        camera_settings = sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_BRIGHTNESS
        str_camera_settings = "Brightness"
        print("Camera settings: BRIGHTNESS")


def record(cam, runtime, mat):
    vid = tp.PyERROR_CODE.PyERROR_CODE_FAILURE
    out = False
    while vid != tp.PyERROR_CODE.PySUCCESS and not out:
        filepath = input("Enter filepath name: ")
        vid = cam.enable_recording(filepath)
        print(repr(vid))
        if vid == tp.PyERROR_CODE.PySUCCESS:
            print("Recording started...")
            out = True
            print("Hit spacebar to stop recording: ")
            key = False
            while key != 32:  # for spacebar
                err = cam.grab(runtime)
                if err == tp.PyERROR_CODE.PySUCCESS:
                    cam.retrieve_image(mat)
                    cv2.imshow("ZED", mat.get_data())
                    key = cv2.waitKey(5)
                    cam.record()
        else:
            print("Help: you must enter the filepath + filename + SVO extension.")
            print("Recording not started.")
    cam.disable_recording()
    print("Recording finished.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)

    category_index = label_map_util.create_category_index(categories)
    with tf.gfile.FastGFile('frozen_inference_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    

    print("Running...")
    init = zcam.PyInitParameters()
    init.camera_resolution = sl.PyRESOLUTION.PyRESOLUTION_VGA
    init.depth_mode = sl.PyDEPTH_MODE.PyDEPTH_MODE_PERFORMANCE  # Use PERFORMANCE depth mode
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
    runtime.sensing_mode = sl.PySENSING_MODE.PySENSING_MODE_STANDARD  # Use STANDARD sensing mode
    mat = core.PyMat()
	
    
    image = core.PyMat()
    depth = core.PyMat()
    point_cloud = core.PyMat()
    
    print_camera_information(cam)
    print_help()
    centerx = 0
    centery = 0
    key = ''
    while key != 113:  # for 'q' key
        err = cam.grab(runtime)
        if err == tp.PyERROR_CODE.PySUCCESS:
            cam.retrieve_image(image, sl.PyVIEW.PyVIEW_LEFT)
            cam.retrieve_measure(depth, sl.PyMEASURE.PyMEASURE_DEPTH)
            cam.retrieve_measure(point_cloud, sl.PyMEASURE.PyMEASURE_XYZRGBA)
            # Read and preprocess an image.                
            img = image.get_data()
            rows = img.shape[0]
            cols = img.shape[1]
            inp = cv2.resize(img, (300, 300))
            inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
            
            with tf.Session() as sess:
                # Restore session
                sess.graph.as_default()
                tf.import_graph_def(graph_def, name='')

                # Run the model
                out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                                sess.graph.get_tensor_by_name('detection_scores:0'),
                                sess.graph.get_tensor_by_name('detection_boxes:0'),
                                sess.graph.get_tensor_by_name('detection_classes:0')],
                                feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
	
                num_detections = int(out[0][0])
                (nums, scores,boxes, classes) = out
                
                for i in range(num_detections):
                    classId = (out[3][0][i])
                    score = float(out[1][0][i])
                    bbox = [float(v) for v in out[2][0][i]]
                    if score > 0.3:
                        x = bbox[1] * cols
                        y = bbox[0] * rows
                        right = bbox[3] * cols
                        bottom = bbox[2] * rows
                        #cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=1)
                        cv2.circle(img, ( int((x+right)/2), int((y + bottom)/2) ), 8, (125, 155, 21), thickness=1)
                        #vis_util.visualize_boxes_and_labels_on_image_array(
                        #    img,
                        #    np.squeeze(boxes),
                        #    np.squeeze(classes).astype(np.int32),
                        #    np.squeeze(scores),
                        #    category_index,
                        #    use_normalized_coordinates=True,
                        #    line_thickness=8)
                        centerx = int((x+right)/2)
                        centery = int((y + bottom)/2)
                       
            
                        # Get and print distance value in mm at the center of the image
                        # We measure the distance camera - object using Euclidean distance
                        #x = round(image.get_width() / 2)
                        #y = round(image.get_height() / 2)
                        err, point_cloud_value = point_cloud.get_value(centerx, centery)

                        distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                             point_cloud_value[1] * point_cloud_value[1] +
                                             point_cloud_value[2] * point_cloud_value[2])

                        if not np.isnan(distance) and not np.isinf(distance):
                            distance = round(distance)
                            print("object {0} Distance to Camera at ({1}, {2}): {3} mm\n".format(i,x, y, distance))
                            classes = np.squeeze(classes).astype(np.int32)
                            if classes[i] in category_index.keys():
                                class_name = category_index[classes[i]]['name']
                            else:
                                class_name = 'N/A'
                            display_str = str(class_name)
                            cv2.putText(img, "{0}:{1} mm".format(display_str,distance),(int(centerx), int(centery)), cv2.FONT_HERSHEY_SIMPLEX, 1, (125, 155, 21),thickness=2,lineType=2)
                            
                            # Increment the loop
                            
                        else:
                            print("Can't estimate distance at this position, move the camera\n")
                        sys.stdout.flush()
            sess.close() 
            cv2.imshow("ZED", img)
            key = cv2.waitKey(5)
            settings(key, cam, runtime, image)
        else:
            key = cv2.waitKey(5)
    cv2.destroyAllWindows()
    

    cam.close()
    print("\nFINISH")
