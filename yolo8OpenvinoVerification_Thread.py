"""
    Ultralytics yolo8 verification thread.
    Basically an intermediate queue is created and filled with zoomed detections
    from the TPU, CPU, etc. AI and then yolo8 is run on the zoomed in
    image and if person is detected the image is sent to the results queue.
    
    On 19-12900K desktop with Nvidia RTX3070:
    python3 AI.py -tpu -y8v -d -cam 6onvif.txt -rtsp 19cams.rtsp
    Yielded ~75 fps with 25 cameras for an ~80660 second test run.
    There were 6,064,648 frames processed by the TPU with 3595 persons detected.
    The yolo8 verification accepted 3024 and rejected 571.
    
    My review suggests almost all the rejections were false negatives, but a fair price to pay
    for the near complete rejection of false positives from my bogus detection collection.
    
    This setup and inference code closely follows the sample code at:
    https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/yolov8-optimization/yolov8-object-detection.ipynb
    
"""

import datetime
from imutils.video import FPS
import cv2
from ultralytics import YOLO
import openvino as ov
import torch
from pathlib import Path


global __Thread__
__Thread__ = False

global __verifyConf__
__verifyConf__ = 0.75

global __y8modelSTR__
__y8modelSTR__ = 'yolov8l'

global model

global __CONVERTING__
__CONVERTING__ = True


def yolo8ov_thread(resultsQ, yoloQ):
    global __Thread__
    global __verifyConf__
    global model
    global __y8modelSTR__
    global __CONVERTING__
    
    print("Starting Yolo v8 verification thread...\n")
    if yoloQ is None:
        print(    "ERROR! no yolo Queue!")
        return -1
        
    # intialize and setup the model
    # Load the YOLOv8 model
    models_dir = Path("./yolo8")
    models_dir.mkdir(exist_ok=True)

    det_model = YOLO(models_dir / f"{__y8modelSTR__}.pt")
    label_map = det_model.model.names
    ###res = det_model('TestDetection.jpg', conf=__verifyConf__-0.001, verbose=True)    #Useful verify iGPU is being used and get rough idea of inference time.
    res = det_model('TestDetection.jpg', conf=__verifyConf__-0.001, verbose=False)    #Dummy inference to initialize object, better way?  But this works!
    # object detection model  export to OpenVINO format
    det_model_path = models_dir / f"{__y8modelSTR__}_openvino_model/{__y8modelSTR__}.xml"
    if not det_model_path.exists():
        print('\n[INFO] Exporting yolo model to OpenVINO format...')
        det_model.export(format="openvino", dynamic=True, half=True)
    __CONVERTING__ = False
    print('\n[INFO] Using OpenVINO: ' + ov.__version__)
    core = ov.Core()
    ov_config = {}
    det_ov_model = core.read_model(det_model_path)
    det_ov_model.reshape({0: [1, 3, 640, 640]})
    ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
    det_compiled_model = core.compile_model(det_ov_model, "GPU", ov_config)
    
    ## I do not understand the point of this, does it make the ultralytics framework use openvino inference?
    # Aparantly yes, semms the YOLO "object" loads the image from the specified filepath and sets everything up
    def infer(*args):   
        result = det_compiled_model(args)
        return torch.from_numpy(result[0])
        
    det_model.predictor.inference = infer
    det_model.predictor.model.pt = False
    
    yoloVerified=0
    yoloRejected=0
    yoloWaited = 0
    dcnt=0
    ecnt=0
    ncnt=0
    print("Yolo v8 verification thread is running...")
    __Thread__ = True
    while __Thread__ is True:
        try:
            # ssd_frame is full camera resolution with SSD detection box overlaid
            # yolo_frame is "zoomed in" on the SSD detection box and resized to 640x640 for yolo8 inference
            #yoloQ.put((image, cam, personDetected, imageDT, aiStr, boxPoints, yolo_frame), True, 1.0)
            ssd_frame, cam, personDetected, imageDT, ai, boxPoints, yolo_frame = yoloQ.get(True, 1.0)
        except:
            yoloWaited+=1
            continue

        try:
            box_points=(0,0, 0,0, 0,0, 0,0)  # startX, startY, endX, endY, Xcenter, Ycenter, Xlength, Ylength
            personDetected = False
            detectConfidence = 0.0
            # call code to do an inference
            results = det_model(yolo_frame)  # nice!  Ultralytics overloaded method automatically handles jpg vs. image buffer
            # Visualize the results on the image
            annotated_image = results[0].plot(line_width=1, labels=False)
            for result in results:
                boxes=result.boxes
                for i in range(len(boxes.data)):
                    if int(boxes.data[i][5].item()) == 0 and boxes.data[i][4].item() > __verifyConf__:
                        personDetected = True
                        detectConfidence = boxes.data[i][4].item()
                        startX = int(boxes.data[i][0].item())
                        startY = int(boxes.data[i][1].item())
                        endX = int(boxes.data[i][2].item())
                        endY = int(boxes.data[i][3].item())
                        xlen=endX-startX
                        ylen=endY-startY
                        xcen=int((startX+endX)/2)
                        ycen=int((startY+endY)/2)
                        box_points=(startX,startY, endX,endY, xcen,ycen, xlen,ylen)
                        break
            
            if personDetected is True:   # yolov8 has verified the MobilenetSSDv2 person detection
                ## annotated_image is the yolo_frame with the yolo detection boxes overlaid (from the box_points).
                ## boxPoints are from the SSD inference.
                # draw the verification confidence onto the ssd_frame
                yoloVerified+=1
                text = "Yolo8ov: {:.1f}%".format(detectConfidence * 100)   # show verification confidence on detection image
                cv2.putText(ssd_frame, text, (2, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 200), 2)
                if resultsQ.full():
                    [_,_,_,_,_,_,_]=resultsQ.get(False)  # remove oldest result 
                    dcnt+=1                       
                resultsQ.put((ssd_frame, cam, True, imageDT, ai, boxPoints, annotated_image.copy()), True, 1.0)
                ###print(detections, boxpoints)    # lets take a look at what we are getting
            else:
                yoloRejected+=1
                if resultsQ.full():
                    [_,_,_,_,_,_,_]=resultsQ.get(False)  # remove oldest result 
                    ncnt+=1                       
                resultsQ.put((ssd_frame, cam, False, imageDT, ai, (-2,0, 0,0, 0,0, 0,0), annotated_image.copy()), True, 1.0)
        except Exception as e:
            ecnt+=1
            print('[Exception] yolo_thread'+ str(cam) + ': ' + str(e))
            continue
    print("Yolo v8 frames Verified: {}, Rejected: {},  Waited: {} seconds.".format(str(yoloVerified), str(yoloRejected), str(yoloWaited)))
    print("    Verified dropped: " + str(dcnt) + ", results dropped: " + str(ncnt) + ", results.put() exceptions: " + str(ecnt))


