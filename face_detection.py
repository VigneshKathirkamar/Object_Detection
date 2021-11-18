# Draws bounding box manually using co-ordinates instead of using tensorflow object detection module

import os
import numpy as np
import tensorflow as tf
import cv2

PATH_TO_CKPT = '/home/vignesh/virtual_environments/tf_114_cpu/ext_data/face_detection/face_detecton.pb'

def face_detection():

    # Load Tensorflow model
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef() 
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.compat.v1.Session(graph=detection_graph) 

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

    # Actual detection.
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    video_capture = cv2.VideoCapture('/home/vignesh/virtual_environments/tf_114_cpu/ext_data/face_detection/test_video_1.mp4')
    #video_capture = cv2.VideoCapture(0)
    # width = int(video_capture.get(3))
    # print('width is',width)
    # height = int(video_capture.get(4))
    # print('height is',height)

    
    #out = cv2.VideoWriter('/home/vignesh/delete_classify_video.mp4',cv2.VideoWriter_fourcc(*'MJPG'), 30, (width,height))

    nb_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))  #get the total number of frames in a video
    
    frame_count=0
    # Here output the category as string and score to terminal
    corrupt_frame = 0
    while frame_count<=nb_frames: #while loop gets terminated at the last frame of the video
        ret, frame = video_capture.read() #'ret' is a boolean value, 'frame' reads and stores the frame of the video
        width,height,_ = frame.shape
        frame_count+=1
        print('Frame number:',frame_count,'/',nb_frames) #prints the current frame number on terminal
        if ret == False:
            print('corrupted frame')
            corrupt_frame+=1
            
        else:
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            boxes_set =[] # to store the detection boxes co-ordinates
            #frame = cv2.resize(frame,(300,300))
            expanded_frame = np.expand_dims(frame, axis=0)
            (boxes, scores, classes, num_c) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: expanded_frame})
        
            print('number of classes in the frame',num_c[0])
            if num_c[0] == 0: #to add frames which has zero detection in it to the output video
                print('no detections')
            else:
                print('number of classes in the frame {0} is {1}'.format(frame_count,num_c[0]))        
                width,height,_ = frame.shape
                print(height,width)
                conf_threshold = 0.5
                nms_threshold = 0.3
                print('scores are',scores[0])
                confidences = [float(j) for j in scores[0]]
                boxes_set = boxes[0]
                boxes_set = [list(float(b) for b in items) for items in boxes_set]
                
                # Applying NMS to the boudning boxes
                indices = cv2.dnn.NMSBoxes(boxes_set, confidences, conf_threshold, nms_threshold)

                if len(indices)!=0:

                    for i in indices:
                        #i = i[0]
                        box = boxes_set[i]
                        xmin = box[0]
                        ymin = box[1]
                        xmax = box[2]
                        ymax = box[3]
                        font = cv2.FONT_HERSHEY_SIMPLEX 
                        fontScale=1
                        color=(255,0,255)
                        thickness=2
                        lineType=cv2.LINE_AA
                        cv2.putText(frame, str(PATH_TO_CKPT)[-50:], (10,20), font, fontScale, color, thickness, lineType)
                    
                    for k in range(0,int(num_c[0])): # iterating through the classes
                        if scores[0][k]>=0.5: #if the scores of the classes are greater than 0.5
                            cv2.putText(frame,str(round(scores[0][i]*100,2))+'% class' +str(classes[0][i]),(int(ymin*height),int(xmin*width)),font,fontScale,color,thickness, lineType)
                            cv2.rectangle(frame,(int(ymin*height),int(xmin*width)),(int(ymax*height),int(xmax*width)),color,thickness)

                else:
                    print('empty index')
                cv2.imshow('Detection', frame)
        
        if cv2.waitKey(1) == ord('q'):
            print('user stopped')
            break  
    cv2.destroyAllWindows()

if __name__ == '__main__':
    face_detection()
