#! /usr/bin/python3
'''
    28JUL2024wbk -- OpenVINO_SSD_Thread.py
    Run MobilenetSSD_v2 inferences on CPU using OpenVINO 2024
    For use with AI2.py
    
    Setup and inference code largely lifted from the openvino python example:
    hello_reshape_ssd.py
    That was installed by the apt install of openvino 2024.2, the apt install is broken
    so I had to do a pip install to run the code :(
'''

import numpy as np
import cv2
import datetime
import logging as log
import sys
import os
from imutils.video import FPS
from pathlib import Path
import openvino as ov


global __Thread__
__Thread__ = False

global __VERIY_DIMS__
__VERIFY_DIMS__ = (300,300)

global __Color__
__Color__ = (0, 200, 200)

global __CONVERTING__
__CONVERTING__ = False

global QUIT
QUIT = False

## *** OpenVINO 2024 CPU SSD AI Thread ***
#******************************************************************************************************************
#******************************************************************************************************************
def AI_thread(resultsQ, inframe, cameraLock, nextCamera, Ncameras,
                PREPROCESS_DIMS, confidence, verifyConf, dnnTarget, blobThreshold, yoloQ):
    global __Thread__
    global __VERIY_DIMS__
    global __Color__
    global __CONVERTING__
    global QUIT
    
    fcnt=0
    waits=0
    dcnt=0
    ncnt=0
    ecnt=0
    detect=0
    noDetect=0
    DNN_verify_fail=0
    
    models_dir = Path("./mobilenet_ssd_v2")
    models_dir.mkdir(exist_ok=True)

    if os.path.exists('mobilenet_ssd_v2/MobilenetSSDv2cocoIR10.xml'):
        MO_2021 = True
        model_path = 'mobilenet_ssd_v2/MobilenetSSDv2cocoIR10.xml'   # my IR10 conversion done with openvino 2021.3
        aiStr = dnnTarget
    else:
        aiStr = 'ovCPU'
        MO_2021 = False
        if os.path.exists('mobilenet_ssd_v2/ssd_mobilenet_v2_coco_2018_03_29.xml'): # ov converted and saved model from 2018 
            model_path = 'mobilenet_ssd_v2/ssd_mobilenet_v2_coco_2018_03_29.xml'
        else:
            if os.path.exists('../ssd_mobilenet_v2_coco_2018_03_29'):
                __CONVERTING__ = True
                print('[INFO] Converting downloaded ssd_mobilenet_v2_coco_2018_03_29 model, be patient ...')
                model = ov.convert_model('../ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb')
                print('[INFO] Saving converted mode, so this step can be skipped on the next program run.')
                ov.save_model(model,'mobilenet_ssd_v2/ssd_mobilenet_v2_coco_2018_03_29.xml')
                model_path = 'mobilenet_ssd_v2/ssd_mobilenet_v2_coco_2018_03_29.xml'
                __CONVERTING__ = False
            else:
                print('[ERROR] ssd_mobilenet_v2_coco_2018_03_29 has not been found!')
                print('     Download it to one level above AI2 directory with:')
                print('cd ..')
                print('wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz')
                print('tar -zxf ssd_mobilenet_v2_coco_2018_03_29.tar.gz')
                print(' Exiting...')
                __CONVERTING__ = False
                QUIT = True
    
    device_name = 'CPU'
    __VERIFY_DIMS__ = PREPROCESS_DIMS
    
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    print("[INFO] OpenVINO CPU MobilenetSSD  AI thread using " + aiStr + " is starting...")
    if yoloQ is not None:
        print("    OpenVINO CPU MobilenetSSD AI thread is using yolo verification.")
    
    ## basically lifted from hello_reshape_ssd.py sample code installed with apt install of openvino 2024
    # --------------------------- Step 1. Initialize OpenVINO Runtime Core ------------------------------------------------
    log.info('Creating OpenVINO Runtime Core')
    core = ov.Core()
    print('[INFO] Using OpenVINO: ' + ov.__version__)
    devices = core.available_devices
    log.info('Available devices:')
    for device in devices:
        deviceName = core.get_property(device, "FULL_DEVICE_NAME")
        print(f"   {device}: {deviceName}")
        
    # --------------------------- Step 2. Read a model --------------------------------------------------------------------
    log.info(f'Reading the model: {model_path}')
    # (.xml and .bin files) or (.onnx file)
    model = core.read_model(model_path)
    ##print(len(model.outputs), model.outputs)
    ##print('')      

    if len(model.inputs) != 1:
        log.error('Supports only single input topologies.')
        QUIT = True
        return -1   # I don't think this error handling is very clean, but shouldn't happen
    '''
    if len(model.outputs) != 1:
        log.error('Supports only single output topologies')
        print(len(model.outputs), model.outputs)
        print('')      
        return -1
    '''
    
    # --------------------------- Step 3. Set up input --------------------------------------------------------------------
    ## create image to set model size
    '''
        This was very confusing, sample code says:
        'Reshaping the model to the height and width of the input image'
        which makes no sence to me.  If I feed in larger images it sort of works
        but boxes are wrong and detections are poor. I know my MobilenetSSD_v2
        model was for images sized 300x300 so I create a dummy image of this size
        and use it to "reshape" the model.
    '''
    imageM = np.zeros(( 300, 300, 3), np.uint8)
    imageM[:,:] = (127,127,127)
    input_tensor = np.expand_dims(imageM, 0)    # Add N dimension
    n, h, w, c = input_tensor.shape # we'll need h,w later
    '''
    log.info('Reshaping the model to the height and width of the input image')
    model.reshape({model.input().get_any_name(): ov.PartialShape((n, c, h, w))})
    #print(n, c, w, h)
    '''
    if MO_2021:
        # --------------------------- Step 4. Apply preprocessing -------------------------------------------------------------
        ## I've made zero effort to understand this, but it seems to work!
        ppp = ov.preprocess.PrePostProcessor(model)
        # 1) Set input tensor information:
        # - input() provides information about a single model input
        # - precision of tensor is supposed to be 'u8'
        # - layout of data is 'NHWC'
        ppp.input().tensor() \
            .set_element_type(ov.Type.u8) \
            .set_layout(ov.Layout('NHWC'))  # noqa: N400
        # 2) Here we suppose model has 'NCHW' layout for input
        ppp.input().model().set_layout(ov.Layout('NCHW'))
        # 3) Set output tensor information:
        # - precision of tensor is supposed to be 'f32'
        ##ppp.output().tensor().set_element_type(ov.Type.f32)
        # 4) Apply preprocessing modifing the original 'model'
        model = ppp.build()
        # ---------------------------Step 4. Loading model to the device-------------------------------------------------------
    
    log.info('Compiling model to the ' + device_name + ' plugin')
    compiled_model = core.compile_model(model, device_name)


    __Thread__ = True
    print("[INFO] OpenVINO CPU MobilenetSSD AI thread using " + aiStr + " is running...")
    cfps = FPS().start()
    while __Thread__ and not QUIT:
        cameraLock.acquire()
        cq=nextCamera
        nextCamera = (nextCamera+1)%Ncameras
        cameraLock.release()
        # get a frame
        try:
            (image, cam, imageDT) = inframe[cq].get(True,0.100)
        except:
            image = None
            waits+=1
            continue
        if image is None:
            continue
        orig_image=image.copy()   # for zoomed in verification run
        (H,W)=image.shape[:2]
        imageM = cv2.resize(image, (300,300))
        input_tensor = np.expand_dims(imageM, 0)
        results = compiled_model.infer_new_request({0: input_tensor})
        '''
        print('\nModel Output:')  # dump what the model returns
        print(results)
        print(' ')
        '''
        if MO_2021:
            predictions = next(iter(results.values()))
            detections = predictions.reshape(-1, 7)

        cfps.update()    # update the FPS counter
        # loop over the detections, pretty much straight from the PyImageSearch sample code.
        personIdx=1     # openvino is one based index into coco_labels.txt
        personDetected = False
        DNNdetect=False
        ndetected=0
        fcnt+=1
        boxPoints=(0,0, 0,0, 0,0, 0,0)  # startX, startY, endX, endY, Xcenter, Ycenter, Xlength, Ylength
        
        if MO_2021:
            for detection in detections:
                conf = detection[2]  # extract the confidence (i.e., probability)
                idx = int(detection[1])   # extract the index of the class label, 1 based, not zero based!
                # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
                if conf > confidence and idx == personIdx :
                    # then compute the (x, y)-coordinates of the bounding box for the object
                    startX = int(detection[3] * W)
                    startY = int(detection[4] * H)
                    endX = int(detection[5] * W)
                    endY = int(detection[6] * H)
                    xlen=endX-startX
                    ylen=endY-startY
                    xcen=int((startX+endX)/2)
                    ycen=int((startY+endY)/2)
                    boxPoints=(startX,startY, endX,endY, xcen,ycen, xlen,ylen)
                    # adhoc "fix" for out of focus blobs close to the camera
                    # out of focus blobs sometimes falsely detect -- insects walking on camera, etc.
                    # TODO: make blobThreshold be camera specific?  So far doesn't seem necessary.
                    if float(xlen*ylen)/(W*H) > blobThreshold:     # detection filling too much of the frame is bogus
                        continue
                    personDetected = True
                    ndetected+=1
                    break    # the one with highest confidence is enough
        else:   # 2024.3 model conversion
            num_detected=int(results[3][0]) # get number of objects detected, always seems to be 1 with 2024.3 converted model
            for i in range(num_detected):
                conf = results[2][0][i]
                idx = int(results[1][0][i])
                if conf > confidence and idx == personIdx:
                    startX = int(results[0][0][i][1] * W)   # box points
                    startY = int(results[0][0][i][0] * H)
                    endX = int(results[0][0][i][3] * W)
                    endY = int(results[0][0][i][2] * H)
                    xlen=endX-startX
                    ylen=endY-startY
                    xcen=int((startX+endX)/2)
                    ycen=int((startY+endY)/2)
                    boxPoints=(startX,startY, endX,endY, xcen,ycen, xlen,ylen)
                    if float(xlen*ylen)/(W*H) > blobThreshold:     # detection filling too much of the frame is bogus
                        continue
                    personDetected = True
                    ndetected+=1
                    break
        
        if personDetected:   
            # In my real world use I have some static false detections, mostly under IR or mixed lighting -- hanging plants etc.
            # I could put camera specific adhoc filters here based on (xlen,ylen,xcenter,ycenter)
            # TODO: come up with better way to do it, probably return (xlen,ylen,xcenter,ycenter) and filter at saving or Notify step.
            #       Now seems best to do it in the node-red notification step, don't need to stop AI to make changes!
            # display and label the prediction
            label = "{:.1f}%  C:{},{}  W:{} H:{}  UL:{},{}  LR:{},{}  {}".format(conf * 100,
                     str(xcen), str(ycen), str(xlen), str(ylen), str(startX), str(startY), str(endX), str(endY), aiStr)
            cv2.putText(image, label, (2, (H-5)-(ndetected*28)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, __Color__, 2, cv2.LINE_AA)
            cv2.rectangle(image, (startX, startY), (endX, endY), __Color__, 2)
            initialConf=conf
            # do zoom and redetect to verify, rejects lots of plants as people false detection.
            personDetected = False  # repeat on zoomed detection box
            DNNdetect = True    # flag we had initial DNN detection
            try:
                ## removing this box expansion really hurt the verification sensitivity
                # zoom in on detection box and run second inference for verification.
                blen=max(xlen,ylen)
                if blen < PREPROCESS_DIMS[0]:
                    blen = PREPROCESS_DIMS[0]   # expand crop pixels so resize always makes smaller image
                adj=int(1.3*blen/2) # enlarge detection box by 30% and make crop be square about box center
                CstartX=max(xcen-adj,0)
                CendX=min(xcen+adj,W-1)
                CstartY=max(ycen-adj,0)
                CendY=min(ycen+adj,H-1)
                zimg = cv2.resize(orig_image[CstartY:CendY, CstartX:CendX], PREPROCESS_DIMS, interpolation = cv2.INTER_AREA)
                (h, w) = zimg.shape[:2]  # this will be PREPROCESS_DIMS
                if (h,w) != PREPROCESS_DIMS:    ## do I need this?
                    print(" OpenVINO CPU verification, Bad resize!  h:{}  w:{}".format(h, w))
                    continue
            except Exception as e:
                print(aiStr + " crop Exception: " + str(e))
                ##print(aiStr + " crop region ERROR: ", startY, endY, startX, endX)
                continue
            input_tensor = np.expand_dims(zimg, 0)
            results = compiled_model.infer_new_request({0: input_tensor})
            
            if MO_2021:
                predictions = next(iter(results.values()))
                detections = predictions.reshape(-1, 7)
                for detection in detections:
                    conf = detection[2]  # extract the confidence (i.e., probability)
                    idx = int(detection[1])   # extract the index of the class label, 1 based, not zero based!
                    # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
                    if conf > verifyConf and idx == personIdx :
                        text = "Verify: {:.1f}%".format(conf * 100)   # show verification confidence
                        cv2.putText(image, text, (2, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, __Color__, 2)
                        personDetected = True
                        break   # one is good enough
            else:
                num_detected=int(results[3][0]) # get number of objects detected, always seems to be 1 with 2024.3 converted model
                for i in range(num_detected):
                    conf = results[2][0][i]
                    idx = int(results[1][0][i])
                    if conf > confidence and idx == personIdx:
                        text = "Verify: {:.1f}%".format(conf * 100)   # show verification confidence
                        cv2.putText(image, text, (2, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, __Color__, 2)
                        personDetected = True
                        break   # one is good enough
            
            cfps.update()    # update the FPS counter
        try:
            # Queue results
            if yoloQ is not None:
                # pass to yolo  for verification, or pass as zoomed image for alerts
                if personDetected: # OpenVINO detection
                    detect+=1
                    if blen < __VERIFY_DIMS__[0]:
                        adj=int(1.1*__VERIFY_DIMS__[0]/2) 
                        CstartX=max(xcen-adj,0)
                        CendX=min(xcen+adj,W-1)
                        CstartY=max(ycen-adj,0)
                        CendY=min(ycen+adj,H-1)
                    person_crop = orig_image[CstartY:CendY, CstartX:CendX]
                    if yoloQ.full():
                        [_,_,_,_,_,_,_]=yoloQ.get(False)  # remove oldest result 
                        dcnt+=1                                               
                    yoloQ.put((image, cam, personDetected, imageDT, aiStr, boxPoints, person_crop.copy() ), True, 1.0)    # try not to drop frames with detections
                    ###print("yoloQ.put() " + str(detect))
                else:
                    noDetect+=1
                    if resultsQ.full():
                        [_,_,_,_,_,_,_]=resultsQ.get(False)  # remove oldest result 
                        ncnt+=1                       
                    if DNNdetect: # DNN verification failed
                        DNN_verify_fail+=1
                        resultsQ.put((image.copy(), cam, False, imageDT, aiStr, (-1,0, 0,0, 0,0, 0,0), zimg.copy() ), True, 1.00) # -1 flags this AI verify fail
                    else:
                        resultsQ.put((image.copy(), cam, False, imageDT, aiStr, (0,0, 0,0, 0,0, 0,0), None), True, 0.200) # 0 boxpoints flag no detection
            else:   # No yolo verification
                if personDetected:
                    detect+=1
                    if resultsQ.full():
                        [_,_,_,_,_,_,_]=resultsQ.get(False)  # remove oldest result 
                        dcnt+=1                       
                    person_crop = image[CstartY:CendY, CstartX:CendX] # since no yolo verification, show original detection in zoomed version
                    resultsQ.put((image.copy(), cam, personDetected, imageDT, aiStr, boxPoints, person_crop.copy() ), True, 1.0) 
                else:
                    noDetect+=1
                    if resultsQ.full():
                        [_,_,_,_,_,_,_]=resultsQ.get(False)  # remove oldest result 
                        ncnt+=1   
                    if DNNdetect: # DNN verification failed
                        DNN_verify_fail+=1
                        resultsQ.put((image.copy(), cam, False, imageDT, aiStr, (-1,0, 0,0, 0,0, 0,0), zimg.copy() ), True, 1.00)
                    else:
                        resultsQ.put((image.copy(), cam, False, imageDT, aiStr, (0,0, 0,0, 0,0, 0,0), None), True, 0.200)
        except Exception as e:
            # presumably outptut queue was full, main thread too slow.
            ecnt+=1
            #print("OpenVINO CPU MobilenetSSD output queue write Exception: " + str(e))
            continue
    # Thread exits
    cfps.stop()    # stop the FPS counter timer
    print("OpenVINO CPU MobilenetSSD AI thread " + aiStr + ", waited: " + str(waits) + " dropped: " + str(ecnt+dcnt+ncnt) + " of "
         + str(fcnt) + " images.  AI: {:.2f} inferences/sec".format(cfps.fps()))
    print("    " + aiStr + " Persons Detected: " + str(detect) + ",  Frames with no person: " + str(noDetect))
    print("    " + aiStr + " " + str(DNN_verify_fail) + " detections failed zoom-in verification.")
    print("    " + aiStr + " Detections dropped: " + str(dcnt) + ", results dropped: " + str(ncnt) + ", resultsQ.put() exceptions: " + str(ecnt))



