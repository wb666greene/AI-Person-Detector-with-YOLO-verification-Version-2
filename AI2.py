#! /usr/bin/python3

# Typical command lines for a full test on 22.04 with openvino yolo8 verification:
"""
28JUL2024wbk -- AI2.py
    add support for OpenVINO 2024
    only support MobilenetSSD_V2 for initial detecting, on openvino CPU or Coral TPU
    remove yolo4 support
    support yolo8 verification step only on CUDA and OpenVINO GPU
    remove support for MQTT cams

Example command line:
  wally@Wahine:~$ conda activate yolo8  or source yolo8/bin/activate depending on which virtual environment
  (yolo8) wally@Wahine:~$ cd AI2
  Coral TPU initial detection and CUDA yolo8 verification:
  (yolo8) wally@Wahine:~/AI2$    python AI2.py -y8v -nTPU 1 -d 1 -cam 6onvif.txt -rtsp 19cams.rtsp

OpenVINO CPU initial detection and GPU yolo8 verification:
(yolo8) wally@Wahine:~/AI2$    python AI2.py -y8ovv -nt 1 -d 1 -cam 6onvif.txt -rtsp 19cams.rtsp

It can be installed to run on 22.04 without virtual environment to simplify installation,
but a VENV virtual environment is recommended, I've had issues starting Conda envs with node-red.

appending 2> /dev/null to the start command line helps if you are getting "Invalid UE golomb code" opencv warnings.
"""


# import the necessary packages
import sys
import signal
from imutils.video import FPS
import argparse
import numpy as np
import cv2
import paho.mqtt.client as mqtt
import os
import time
import datetime
import requests

# threading stuff
from queue import Queue
from threading import Lock, Thread
# for saving PTZ view maps
import pickle


# *** System Globals
# these are write once in main() and read-only everywhere else, thus don't need syncronization
global QUIT
QUIT=False  # True exits main loop and all threads
global Nrtsp
global Nonvif
global Ncameras
global __CamName__
global AlarmMode    # would be Notify, Audio, or Idle, Idle mode still saves detections
global UImode
global CameraToView
global subscribeTopic
subscribeTopic = "Alarm/#"  # topic controller publishes to to set AI operational modes
global inframeQ
global resultsQ
# this variable to distribute queued data to the AI threads needs syncronization
global nextCamera
nextCamera = 0      # next camera queue for AI threads to use to grab a frame
cameraLock = Lock()
# globals for thread control

global __onvifThread__
global __rtspThread__
global __fisheyeThread__

global GRID_SIZE
global CLIP_LIMIT
global CLAHE

global __DEBUG__
__DEBUG__ = False

# *** constants for MobileNet-SSD_V2  AI model
# frame dimensions should be sqaure for MobileNet-SSD_v2
PREPROCESS_DIMS = (300, 300)


if 1:
    # *** get command line parameters
    # construct the argument parser and parse the arguments for this module
    ap = argparse.ArgumentParser()

    # specify use of Coral TPU stick
    ap.add_argument("-tpu", "--TPU", action="store_true", help="Use Coral TPU device instead of CPU for SSD thread")

    # enable zoom and verify using yolo inference (requires Nvidia cuda capable video card and working CUDA installation.
    ap.add_argument("-y8v", "--yolo8_verify", action="store_true", help="Verify detection with a CUDA yolov8 inference on zoomed region")
    ap.add_argument("-y8ovv", "--yolo8ov_verify", action="store_true", help="Verify detection with openvino GPU yolov8 inference on zoomed region")
    ap.add_argument("-y8tpu", "--yolo8tpu_verify", action="store_true", help="Verify detection with a CUDA yolov8 inference on zoomed region")
    
    # parameters that might be installation dependent
    ap.add_argument("-c", "--confidence", type=float, default=0.70, help="Detection confidence threshold")
    ap.add_argument("-vc", "--verifyConfidence", type=float, default=0.80, help="Detection confidence for verification")
    ap.add_argument("-yvc", "--yoloVerifyConfidence", type=float, default=0.75, help="Detection confidence for yolp verification")
    ap.add_argument("-blob", "--blobFilter", type=float, default=0.33, help="Reject detections that are more than this fraction of the frame")

    # yolo8 verification
    ap.add_argument("-yvq", "--YoloVQ", type=int, default=10, help="Depth of YOLO verification queue, should be about YOLO framerate, default=10")
    ap.add_argument("-rq", "--resultsQ", type=int, default=10, help="Minimum Depth of results queue, default=10")

    # specify text file with list of URLs for camera rtsp streams
    ap.add_argument("-rtsp", "--rtspURLs", default="cameraURL.rtsp", help="Path to file containing rtsp camera stream URLs")

    # specify text file with list of URLs cameras http "Onvif" snapshot jpg images
    ap.add_argument("-cam", "--cameraURLs", default="cameraURL.txt", help="Path to file containing http camera jpeg image URLs")

    # display mode, mostly for test/debug and setup, general plan would be to run "headless"
    ap.add_argument("-d", "--display", action="store_true", help="Display live images on host screen")

    # specify MQTT broker
    ap.add_argument("-mqtt", "--mqttBroker", default="localhost", help="Name or IP of MQTT Broker")

    # specify display width and height
    ap.add_argument("-dw", "--displayWidth", type=int, default=1920, help="Host display Width in pixels, default=1920")
    ap.add_argument("-dh", "--displayHeight", type=int, default=1080, help="Host display Height in pixels, default=1080")

    # specify host display width and height of camera image
    ap.add_argument("-iw", "--imwinWidth", type=int, default=608, help="Camera host display window Width in pixels, default=608")
    ap.add_argument("-ih", "--imwinHeight", type=int, default=342, help="Camera host display window Height in pixels, default=342")

    # These are too help the auto tiling algorithm, but not realiabe with window manager and CV2 version difference
    ap.add_argument("-Ytop", "--Ytop", type=int, default=0, help="Y in pixels to move all windows down for tiling, default=0")
    ap.add_argument("-Xleft", "--Xleft", type=int, default=0, help="X in pixels to move all windows left for tiling, default=0")
    ap.add_argument("-Yoff", "--Yoffset", type=int, default=36, help="Y offset to account for window decorations, default=38")
    ap.add_argument("-Xoff", "--Xoffset", type=int, default=0, help="X offset to account for window decorations, default=0")
    
    # show zoom image of detections even if -d parameter is 0
    ap.add_argument("-z", "--DisplayZoom", action="store_true", help="Always display zoomed image of detection.")
    ap.add_argument("-y", "--DisplayYolo", action="store_true", help="Always Yolo detect/reject.")
    
    # Disable local save of detections on AI host -nls is same as -nsz and -nsf options
    ap.add_argument("-nls", "--NoLocalSave", action="store_true", help="No saving of detection images on local AI host")
    # don't save zoomed image locally
    ap.add_argument("-nsz", "--NoSaveZoom", action="store_true", help="Don't locally save zoomed detection image")
    # don't save full images locally  
    ap.add_argument("-nsf", "--NoSaveFull", action="store_true", help="Don't locally save full detection frame.")

    # send full frame image of detections to node-red instead of zoomed in on detection
    ap.add_argument("-nrf", "--nodeRedFull", action="store_true", help="Full frame detection images to node-read instead of zoom images")

    # specify file path of location to same detection images on the localhost
    ap.add_argument("-sp", "--savePath", default="", help="Path to location for saving detection images, default ../detect")
    # save all processed images, fills disk quickly, really slows things down, but useful for test/debug

    ## CLAHE parameters
    ap.add_argument("-cl", "--ClipLimit", type=float, default=4.5, help="CLAHE clipLimit parameter, default=4.5")
    ap.add_argument("-gs", "--GridSize", type=int, default=5, help="CLAHE tileGridSize parameter, default=5")
    ap.add_argument("-clahe", "--CLAHE", action="store_true", help="Enable CLAHE contrast enhancement on zoomed detection")

    # debug visulize verification rejections
    ap.add_argument("-dbg", "--debug", action="store_true", help="Enable debug display of verification failures")

    args = vars(ap.parse_args())



# mark start of this code in log file
print("$$$**************************************************************$$$")
currentDT = datetime.datetime.now()
print("*** " + currentDT.strftime(" %Y-%m-%d %H:%M:%S") + "  ***")
print("[INFO] using openCV-" + cv2.__version__)


# *** Function definitions
#**********************************************************************************************************************
#**********************************************************************************************************************
#**********************************************************************************************************************

# Boilerplate code to setup signal handler for graceful shutdown on Linux
def sigint_handler(signal, frame):
        global QUIT
        currentDT = datetime.datetime.now()
        print('caught SIGINT, normal exit. -- ' + currentDT.strftime("%Y-%m-%d  %H:%M:%S"))
        QUIT=True

def sighup_handler(signal, frame):
        global QUIT
        currentDT = datetime.datetime.now()
        print('caught SIGHUP! ** ' + currentDT.strftime("%Y-%m-%d  %H:%M:%S"))
        QUIT=True

def sigquit_handler(signal, frame):
        global QUIT
        currentDT = datetime.datetime.now()
        print('caught SIGQUIT! *** ' + currentDT.strftime("%Y-%m-%d  %H:%M:%S"))
        QUIT=True

def sigterm_handler(signal, frame):
        global QUIT
        currentDT = datetime.datetime.now()
        print('caught SIGTERM, normal exit. ' + currentDT.strftime("%Y-%m-%d  %H:%M:%S"))
        QUIT=True

signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGHUP, sighup_handler)
signal.signal(signal.SIGQUIT, sigquit_handler)
signal.signal(signal.SIGTERM, sigterm_handler)



#**********************************************************************************************************************
## MQTT callback functions
##
# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    global subscribeTopic
    #print("Connected with result code "+str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.  -- straight from Paho-Mqtt docs!
    client.subscribe(subscribeTopic)



# The callback for when a PUBLISH message is received from the server, aka message from SUBSCRIBE topic.
def on_message(client, userdata, msg):
    global AlarmMode    # would be Notify, Audio, or Idle, Idle mode doesn't save detections
    global UImode
    global CameraToView
    global QUIT
    if str(msg.topic) == "Alarm/MODE":          # Idle will not save detections, Audio & Notify are the same here
        currentDT = datetime.datetime.now()     # logfile entry
        AlarmMode = str(msg.payload.decode('utf-8'))
        print(str(msg.topic)+":  " + AlarmMode + currentDT.strftime(" ... %Y-%m-%d %H:%M:%S"))
        return
    # UImode: 0->no Dasboard display, 1->live image from selected cameram 2->detections from selected camera, 3->detection from any camera
    if str(msg.topic) == "Alarm/UImode":    # dashboard control Disable, Detections, Live exposes apparent node-red websocket bugs
        currentDT = datetime.datetime.now() # especially if browser is not on localhost, use sparingly, useful for camera setup.
        print(str(msg.topic)+": " + str(int(msg.payload)) + currentDT.strftime("   ... %Y-%m-%d %H:%M:%S"))
        UImode = int(msg.payload)
        return
    if str(msg.topic) == "Alarm/ViewCamera":    # dashboard control to select image to view
        currentDT = datetime.datetime.now()
        print(str(msg.topic)+": " + str(int(msg.payload)) + currentDT.strftime("   ... %Y-%m-%d %H:%M:%S"))
        CameraToView = int(msg.payload)
        return
    if str(msg.topic) == "Alarm/QUIT":    # dashboard message to exit program, signals seem unreliable on recent Ubuntu update! (~7AUG2024)
        currentDT = datetime.datetime.now()
        print(str(msg.topic)+": "  + currentDT.strftime("   ... %Y-%m-%d %H:%M:%S"))
        QUIT = True
        return


def on_publish(client, userdata, mid):
    #print("mid: " + str(mid))      # don't think I need to care about this for now, print for initial tests
    pass


def on_disconnect(client, userdata, rc):
    if rc != 0:
        currentDT = datetime.datetime.now()
        print("Unexpected MQTT disconnection!" + currentDT.strftime(" ... %Y-%m-%d %H:%M:%S  "), client)
    pass




# *** main()
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def main():
    global QUIT
    global AlarmMode    # would be Notify, Audio, or Idle, Idle mode doesn't save detections
    AlarmMode="Audio"   # will be Email, Audio, or Idle  via MQTT controller from alarmboneServer
    global CameraToView
    CameraToView=0
    global UImode
    UImode=0    # controls if MQTT buffers of processed images from selected camera are sent as topic: ImageBuffer
    global subscribeTopic
    global Nonvif
    global Nrtsp
    global inframeQ
    global resultsQ
    global Ncameras
    global __CamName__
    global CamName
##    global __PYCORAL__
    # globals for thread control, maybe QUITf() was cleaner, but I can stage the stopping for better [INFO} reporting on exit

    global __rtspThread__
    global __fisheyeThread__
    # command line "store true" flags

    global GRID_SIZE
    global CLIP_LIMIT
    global CLAHE

    global __DEBUG__


    # set variables from command line auguments or defaults
    nCPUthreads = True
    nCoral = False
    dispMode = False
    nCoral = args["TPU"]
    if nCoral is True:
        nCPUthreads = False
    confidence = args["confidence"]
    verifyConf = args["verifyConfidence"]
    yoloVerifyConf = args["yoloVerifyConfidence"]
    blobThreshold = args["blobFilter"]
    dispMode = args["display"]
    CAMERAS = args["cameraURLs"]
    RTSP = args["rtspURLs"]
    MQTTserver = args["mqttBroker"]     # this is for command and control messages, and detection messages
    displayWidth = args["displayWidth"]
    displayHeight = args["displayHeight"]
    imwinWidth = args["imwinWidth"]
    imwinHeight = args["imwinHeight"]
    Ytop = args["Ytop"]
    Xleft = args["Xleft"]
    Yborder = args["Yoffset"]
    Xborder = args["Xoffset"]
    savePath = args["savePath"]
    NoLocalSave = args["NoLocalSave"]
    NoSaveZoom = args["NoSaveZoom"]
    NoSaveFull = args["NoSaveFull"]
    nodeRedFull= args["nodeRedFull"]
    __DEBUG__ = args["debug"]
    show_zoom = args["DisplayZoom"]
    show_yolo = args["DisplayYolo"]
    if show_zoom and show_yolo:
        print("[INFO] Doesn't make sense to use both -z and -y, detection_zoom and yolo_verify are the same when Person Detected.")
    # *** connect to MQTT broker for control/status messages
    print("\n[INFO] connecting to MQTT " + MQTTserver + " broker...")
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_publish = on_publish
    client.on_disconnect = on_disconnect
    client.will_set("AI/Status", "Python AI2 has died!", 2, True)  # let everyone know we have died, perhaps node-red can restart it
    client.connect(MQTTserver, 1883, 60)
    client.loop_start()

    client.publish("AI/Status", "AI2 Python Code Has Started.", 2, True)
    # *** setup path to save AI detection images
    if savePath == "":
        home, _ = os.path.split(os.getcwd())
        detectPath = home + "/detect"
        if os.path.exists(detectPath) == False:
            os.mkdir(detectPath)
    else:
        detectPath=savePath
        if os.path.exists(detectPath) == False:
            print(" Path to location to save detection images must exist!  Exiting ...")
            client.publish("AI/Status", "Path to location to save detection images must exist!  Exiting ...", 2, True)
            quit()
    yolo8_verify=args["yolo8_verify"]
    OVyolo8_verify=args["yolo8ov_verify"]
    TPUyolo8_verify=args["yolo8tpu_verify"]
    yoloVQdepth=args["YoloVQ"]
    resultsQdepth=args["resultsQ"]


    if yolo8_verify and OVyolo8_verify:
        OVyolo8_verify = False
        print("[WARN] Only one of -y8ovv or -y8v can be used. Forcing -y8v CUDA")

    # init CLAHE
    CLAHE = args["CLAHE"]
    if CLAHE:
        GRID_SIZE = (args["GridSize"],args["GridSize"])
        CLIP_LIMIT = args["ClipLimit"]
        clahe = cv2.createCLAHE(CLIP_LIMIT,GRID_SIZE)


    # starting AI threads can take a long time, send image and message to dasboard to indicate progress
    img = np.zeros(( imwinHeight, imwinWidth, 3), np.uint8)
    img[:,:] = (192,127,127)
    retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    client.publish("ImageBuffer/!Starting AI threads, this can take awhile!.", bytearray(img_as_jpg), 0, False)


    # *** get Onvif camera URLs
    # cameraURL.txt file can be created by first running the nodejs program (requires node-onvif be installed):
    # nodejs onvif_discover.js
    #
    # This code does not really use any Onvif features, Onvif compatability is useful to "automate" getting  URLs used to grab snapshots.
    # Any camera that returns a jpeg image from a web request to a static URL should work.
    CamName=list()  # dynamically built list of camera names read from file or created as Cam0, Cam1, ... CamN
    try:
        #CameraURL=[line.rstrip() for line in open(CAMERAS)]    # force file not found
        #Nonvif=len(CameraURL)
        l=[line.split() for line in open(CAMERAS)]
        CameraURL=list()
        client.publish("AI/Status", "Loading Onvif Camera URLS.", 2, True)
        time.sleep(1.0)     # give user a chance to see the feedback in node-red
        for i in range(len(l)):
            CameraURL.append(l[i][0])
            if len(l[i]) > 1:
                CamName.append(l[i][1])
            else:
                CamName.append("Cam" + str(i))
        Nonvif=len(CameraURL)
        print("\n[INFO] " + str(Nonvif) + " http Onvif snapshot threads will be created.")
    except Exception as e:
        # No Onvif cameras
        #print(e)
        print("[INFO] No " + str(CAMERAS) + " file.  No Onvif snapshot threads will be created.")
        Nonvif=0
    Ncameras=Nonvif
    #print(CamName)


    # *** get rtsp URLs
    try:
        #rtspURL=[line.rstrip() for line in open(RTSP)]
        #Nrtsp=len(rtspURL)
        rtspURL=list()
        l=[line.split() for line in open(RTSP)]
        client.publish("AI/Status", "Loading RTSP Camera URLS.", 2, True)
        for i in range(len(l)):
            rtspURL.append(l[i][0])
            if len(l[i]) > 1:
                CamName.append(l[i][1])
            else:
                CamName.append("Cam" + str(i+Ncameras))
        Nrtsp=len(rtspURL)
        print("\n[INFO] " + str(Nrtsp) + " rtsp stream threads will be created.")
    except:
        # no rtsp cameras
        print("[INFO] No " + str(RTSP) + " file.  No rtsp stream threads will be created.")
        Nrtsp=0
    Ncameras+=Nrtsp


    # define fisheye cameras and virtual PTZ views
    # fisheye.rtsp is expected to be created with the interactive fisheye_window C++ utility program
    try:
        l=[line.rstrip() for line in open('fisheye.rtsp')]
        FErtspURL=list()
        PTZparam=list()
        j=-1
        client.publish("AI/Status", "Loading fisheye Camera URLS.", 2, True)
        for i in range(len(l)):
            if not l[i]: continue
            if l[i].startswith('rtsp'):
                FErtspURL.append(l[i])
                j+=1
                PTZparam.append([])
            else:
                PTZparam[j].append(l[i].strip().split(' '))

        print("\n[INFO] Setting up PTZ virtual cameras views from fisheye camera ...")
        #print(FErtspURL)
        #print(PTZparam)
        Nfisheye=len(FErtspURL)     # modified rtsp thread will send PTZ views to seperate queues, this is number of fisheye threads
        NfeCam=0                    # total number of queues to be created for virtual PTZ cameras
        for i in range(Nfisheye):
            if len(PTZparam[i])<2 or len(PTZparam[i][0])<2 or len(PTZparam[i][1])!=6:
                # this is where Python's features make code simple but obtuse!
                # setting up this data structure in C/C++ gives me cooties with the variable number of possible PTZ views per camera!
                print('[ERROR] PTZparam[' + str(i) + '] must contain [srcW, srcH],[dstW,detH,  alpha,beta,theta,zoom] entries, Exiting ...')
                quit()
            NfeCam += len(PTZparam[i])-1 # the first entry is camera resolution, not a PTZ view
        # I'm not bothering with naming fisheye camera views, just create sequential names
        for i in range(NfeCam):
            CamName.append("FEview" + str(i))
    except:
        # no fisheye cameras
        print("[INFO] No fisheye.rtsp file.  No fisheye camera rtsp stream threads will be created.")
        NfeCam=0
        Nfisheye=0
    FishEyeOffset=Ncameras
    Ncameras+=NfeCam # add fisheye virtual PTZ views to cameras count


    if Ncameras == 0:
        print("[INFO] No Cameras or rtsp Streams specified!  Exiting...")
        client.publish("ImageBuffer/!No Camera URLs specified, exiting..!.", bytearray(img_as_jpg), 0, False)
        quit()


    # *** allocate queues
    print("[INFO] allocating camera and stream image queues...")
    client.publish("AI/Status", "Allocating camera and stream image queues.", 2, True)
    # we simply make one queue for each camera, rtsp stream, and MQTTcamera
    QDEPTH = 3      # Make queue depth be three, sometimes get two frames less then 20 mS appart with
                    # "read queue if full and then write frame to queue" in camera input thread
##    QDEPTH = 2      # bump up for trial of "read queue if full and then write to queue" in camera input thread
##    QDEPTH = 1      # small values improve latency
    resultsQ = Queue(max(resultsQdepth,Ncameras))
    inframeQ = list()
    for i in range(Ncameras):
        inframeQ.append(Queue(QDEPTH))

    if yolo8_verify or OVyolo8_verify or TPUyolo8_verify:
        ###yoloQ = Queue(max(10,Ncameras))  # this can lead to very long latencies if the AI thread is much faster than the yolo verification thread.
        yoloQ = Queue(yoloVQdepth)   # This should be approx the lessor of the AI thread frame rate or yolo verification frame rate
    else:
        yoloQ = None

    # build grey image for mqtt windows
    img = np.zeros(( imwinHeight, imwinWidth, 3), np.uint8)
    img[:,:] = (127,127,127)
    retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

    # built json string to dynamically set camera names.
    ## [{"Driveway":"0"},{"Garage":"1"},{"Porch":"2"}]
    cams_json = '[{"'
    for i in range (Ncameras-1):
        cams_json = cams_json + CamName[i] + '":"' + str(i) + '"},{"'
    cams_json = cams_json + CamName[Ncameras-1] + '":"' + str(Ncameras-1) + '"}]'
    ##print(cams_json)
    client.publish("dynamic", cams_json, 0, False)
    
    
    # *** setup display windows if necessary
    # mostly for initial setup and testing, not worth a lot of effort at the moment
    if dispMode:
        client.publish("AI/Status", "Creating live display windows.", 2, True)
        if Nonvif > 0:
            print("[INFO] setting up Onvif camera image windows ...")
            for i in range(Nonvif):
                name=str("Live_" + CamName[i])
                cv2.namedWindow(name, flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
                cv2.imshow(name, img)
                cv2.waitKey(1)
        if Nrtsp > 0:
            print("[INFO] setting up rtsp camera image windows ...")
            for i in range(Nrtsp):
                name=str("Live_" + CamName[i+Nonvif])
                cv2.namedWindow(name, flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
                cv2.imshow(name, img)
                cv2.waitKey(1)
        if NfeCam > 0:
            print("[INFO] setting up  FishEye camera PTZ windows ...")
            for i in range(NfeCam):
                name=str("Live_" + CamName[i+FishEyeOffset])
                cv2.namedWindow(name, flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
                cv2.imshow(name, img)
                cv2.waitKey(1)
        # setup yolov4 verification windows
        if (OVyolo8_verify or yolo8_verify or TPUyolo8_verify) and show_yolo:
            print("[INFO] setting up YOLO verification/reject image windows ...")
            cv2.namedWindow("yolo_verify", flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
            cv2.imshow("yolo_verify", img)
            cv2.waitKey(1)
            cv2.namedWindow("yolo_reject",flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
            cv2.imshow("yolo_reject", img)
            cv2.waitKey(1)
        if show_zoom:
            print("[INFO] setting detection zoom image window ...")
            cv2.namedWindow("detection_zoom", flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
            cv2.imshow("detection_zoom", img)
            cv2.waitKey(1)

        # *** attempt to move windows into tiled grid
        ''' These are set by arguments, these should be the defaults.
        # attempt to compensate for openCV window "decorations" varies too much with system to really work
        Ytop=0
        Xleft=0
        Xborder=0
        Yborder=38
        '''
        Xshift=imwinWidth+Xleft
        Yshift=imwinHeight+Ytop
        Ncols=int(displayWidth/imwinWidth)
        Nrows=int(displayHeight/imwinHeight)
        print("[INFO] Attempting to tile live camera display windows.")
        print(" Rows, Columns: ",Nrows,Ncols)
        for i in range(Ncameras):
            name=str("Live_" + CamName[i])
            row=int(i/Ncols)
            col=i%Ncols
            cv2.moveWindow(name, Xborder+col*Xshift, Yborder+row*Yshift)
            print("Row, Column, x, y: ",row, col, Yborder+row*Yshift, Xborder+col*Xshift)
            cv2.waitKey(1)
    else:
        if show_zoom:
            print("[INFO] setting detection zoom image window ...")
            cv2.namedWindow("detection_zoom", flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
            cv2.imshow("detection_zoom", img)
            cv2.waitKey(1)


    if nCoral is False and nCPUthreads is False:
        client.publish("AI/Status", "[INFO] No Coral TPU device specified, forcing CPU thread.", 2, True)
        print("\n[INFO] No Coral TPU device specified,  forcing one CPU AI thread.")
        nCPUthreads=True   # we always can force one CPU thread, but ~1.8 seconds/frame on Pi3B+

    # these need to be loaded before an AI thread launches them
    '''
    # From the Ultralytics website, tradeoff for the different models, clearly "x" is "best"
    # but its too much for a GTX950 and "m" running on openvino i3 iGPU seemed just as good
    # in a parallel run using the same rtsp streams, too many drops with "x" model on GTX950.
    # Realistically GTX950 is about as low as we can go, I may make this selection a command
    # line parameter eventually, but "l" needs twice the Flops for 2.7 gain in mAPval.
    Model	    Pixels  mAPval  CPU(mS)  A100(nS)   #parms(M)   Flops(B)
    YOLOv8n	    640	    37.3	80.4	 0.99	    3.2	        8.7
    YOLOv8s	    640	    44.9	128.4	 1.20	    11.2	    28.6
    YOLOv8m	    640	    50.2	234.7	 1.83	    25.9	    78.9
    YOLOv8l	    640	    52.9	375.2	 2.39	    43.7	    165.2
    YOLOv8x	    640 	53.9	479.1	 3.53	    68.2	    257.8
    '''
    if yolo8_verify:
        # import Ultralytics yolo8
        client.publish("AI/Status", "Loading Ultralytics CUDA yolo8.", 2, True)
        import yolo8_verification_Thread
        # using yolov8m.pt for now m seems the best speed-accuracy tradeoff
        yolo8_verification_Thread.__y8modelSTR__ = 'yolo8/yolov8m.pt'
        yolo8_verification_Thread.__verifyConf__ = yoloVerifyConf
    if TPUyolo8_verify:
        # import Ultralytics yolo8 thread, flag to use TPU instead of CUDA, small model 512x512
        client.publish("AI/Status", "Loading Ultralytics TPU yolo8.", 2, True)
        import yolo8_verification_Thread
        yolo8_verification_Thread.__verifyConf__ = yoloVerifyConf
        yolo8_verification_Thread.__useTPU__ = True
    if OVyolo8_verify:
        client.publish("AI/Status", "Loading OpenVINO yolo8.", 2, True)
        import yolo8OpenvinoVerification_Thread
        yolo8OpenvinoVerification_Thread.__y8modelSTR__ = 'yolov8m'
        yolo8OpenvinoVerification_Thread.__verifyConf__ = yoloVerifyConf
            
    # *** setup and start Coral AI threads
    if nCoral is True:
        print("\n[INFO] starting Coral TPU AI Thread ...")
        client.publish("AI/Status", "Starting Coral TPU thread.", 2, True)
        import Coral_TPU_Thread
        # *** start Coral TPU threads
        Ct = list() ## not necessary only supporting a single TPU for now.
        print("   ... loading model...")
        if yolo8_verify or OVyolo8_verify:
            Coral_TPU_Thread.__VERIFY_DIMS__ = (640,640)
        if TPUyolo8_verify:
            Coral_TPU_Thread.__VERIFY_DIMS__ = (512,512)
        Ct.append(Thread(target=Coral_TPU_Thread.AI_thread,
            args=(resultsQ, inframeQ, cameraLock, nextCamera, Ncameras,
                    PREPROCESS_DIMS, confidence, verifyConf, "TPU", blobThreshold,  yoloQ)))
        Ct[0].start()
        sleepCount=0
        while Coral_TPU_Thread.__Thread__ is False:
            sleepCount+=1
            time.sleep(1.0)
            client.publish("AI/Status", "Coral TPU Thread is starting " + str(sleepCount), 2, True)
            if sleepCount >= 30:
                client.publish("AI/Status", "[ERROR] Coral_TPU_Thread failed to start, exiting...", 2, True)
                print('[ERROR] Coral_TPU_Thread failed to start, exiting...')
                QUIT = True
        if not QUIT:
            client.publish("AI/Status", "Coral TPU thread is running.", 2, True)

    # ** setup and start openvino CPU AI thread.
    if nCPUthreads is True:
        print("\n[INFO] starting OpenVINO CPU AI Thread ...")
        client.publish("AI/Status", "Starting OpenVINO MobilenetSSD_v2 thread.", 2, True)
        import OpenVINO_SSD_Thread
        CPUt = list()
        if yolo8_verify or OVyolo8_verify:
            OpenVINO_SSD_Thread.__VERIFY_DIMS__ = (640,640)
        if TPUyolo8_verify:
            OpenVINO_SSD_Thread.__VERIFY_DIMS__ = (512,512)
        # We no longer instance the model here and pass it to the thread, instance it in the thread.
        CPUt.append(Thread(target=OpenVINO_SSD_Thread.AI_thread,
                    args=(resultsQ, inframeQ, cameraLock, nextCamera, Ncameras,
                    PREPROCESS_DIMS, confidence, verifyConf, "SSDv2_IR10_CPU", blobThreshold, yoloQ)))
        CPUt[0].start()
        # wait for OpenVINO_SSD_Thread to start, so I can see any error messages can be tough to tell which thread they are from.
        sleepCount=0
        while OpenVINO_SSD_Thread.__Thread__ is False:
            sleepCount+=1
            time.sleep(1.0)
            while OpenVINO_SSD_Thread.__CONVERTING__ is True:
                if sleepCount == 1:
                    print('Converting MobilenetSSD_v2 to openvino, be patient!')
                    client.publish("AI/Status", "Converting MobilenetSSD_v2 to openvino, be patient!", 2, True)
                    sleepCount+=1
                    toggle = 1
                time.sleep(3.0)
                if toggle == 1:
                    client.publish("AI/Status", "Converting MobilenetSSD_v2 working...", 2, True)
                else:
                    client.publish("AI/Status", "Converting MobilenetSSD_v2 still working...", 2, True)
                toggle = (toggle+1)%2
            client.publish("AI/Status", "OpenVINO CPU Thread is starting " + str(sleepCount), 2, True)    
            if sleepCount >= 30:
                client.publish("AI/Status", "[ERROR] OpenVINO_SSD_Thread failed to start, exiting...", 2, True)
                print('[ERROR] OpenVINO_SSD_Thread failed to start, exiting...')
                QUIT = True
        if not QUIT:
            client.publish("AI/Status", "OpenVINO MobilenetSSD_v2 thread is running.", 2, True)
            

    if OVyolo8_verify:
        # Start openvino yolo8 thread
        print("\n[INFO] OpenVINO yolo_v8 verification thread is starting ... ")
        client.publish("AI/Status", "Starting OpenVINO yolo8 verification thread.", 2, True)
        yolo8ov=list()
        yolo8ov.append(Thread(target=yolo8OpenvinoVerification_Thread.yolo8ov_thread, args=(resultsQ, yoloQ)))
        yolo8ov[0].start()
        # wait for yolo thread to be running
        sleepCount=0
        while yolo8OpenvinoVerification_Thread.__Thread__ is False:
            sleepCount+=1
            time.sleep(1.0)
            client.publish("AI/Status", "OpenVINO yolo8 verification thread starting " + str(sleepCount), 2, True)
            while yolo8OpenvinoVerification_Thread.__CONVERTING__ is True:
                if sleepCount == 1:
                    print('Downloading and converting yolo8 openvino model, be patient!')
                    client.publish("AI/Status", "Converting Ultralytics OpenVINO Yolo8 model, be patient!", 2, True)
                    sleepCount+=1   
                    toggle = 1
                time.sleep(3.0)
                if toggle == 1:
                    client.publish("AI/Status", "Converting openvino yolo8 working...", 2, True)
                else:
                    client.publish("AI/Status", "Converting openvino yolo8 still working...", 2, True)
                toggle = (toggle+1)%2
            client.publish("AI/Status", "OpenVINO yolo8 verification starting " + str(sleepCount), 2, True)    
            if sleepCount >= 30:
                print('[ERROR] OpenVINO yolo8 thread failed to start, exiting...')
                client.publish("AI/Status", "[ERROR] OpenVINO yolo8 thread failed to start, exiting...", 2, True)
                QUIT = True
        if not QUIT:
            print("[INFO] OpenVINO yolo_v8 verification thread is running. ")
            client.publish("AI/Status", "OpenVINO yolo8 verification thread is running.", 2, True)


    if yolo8_verify or TPUyolo8_verify:
        if TPUyolo8_verify:
            print("\n[INFO] Ultralytics TPU yolo_v8 verification thread is starting... ")
            client.publish("AI/Status", "Starting Ultralytics TPU yolo8 verification Thread.", 2, True)
        else:
            # Start Ultralytics yolo8 verification thread
            print("\n[INFO] Ultralytics CUDA yolo_v8 verification thread is starting... ")
            client.publish("AI/Status", "Starting Ultralytics CUDA yolo8 verification Thread.", 2, True)
        yolo8=list()
        yolo8.append(Thread(target=yolo8_verification_Thread.yolov8_thread,args=(resultsQ, yoloQ)))
        yolo8[0].start()
        # wait for yolo thread to be running
        sleepCount=0
        while yolo8_verification_Thread.__Thread__ is False:
            sleepCount+=1
            time.sleep(1.0)
            while yolo8_verification_Thread.__CONVERTING__ is True:
                if sleepCount == 1:
                    print('Downloading and converting yolo8 model, be patient!')
                    client.publish("AI/Status", "Converting Ultralytics model, be patient!", 2, True)
                    sleepCount+=1
                    toggle = 1
                time.sleep(3.0)
                if toggle == 1:
                    client.publish("AI/Status", "Converting yolo8 model working...", 2, True)
                else:
                    client.publish("AI/Status", "Converting yolo8 still working...", 2, True)
                toggle = (toggle+1)%2
            client.publish("AI/Status", "Yolo8 verification thread starting " + str(sleepCount), 2, True)    
            if sleepCount >= 30:
                if TPUyolo8_verify:
                    client.publish("AI/Status", "[ERROR] TPU yolo8 thread failed to start, exiting...", 2, True)
                    print('[ERROR] TPU yolo8 thread failed to start, exiting...')
                else:
                    client.publish("AI/Status", "[ERROR] CUDA yolo8 thread failed to start, exiting...", 2, True)
                    print('[ERROR] CUDA yolo8 thread failed to start, exiting...')
                QUIT = True
        if not QUIT:
            if TPUyolo8_verify:
                client.publish("AI/Status", "Ultralytics TPU yolo8 verification thread is running.", 2, True)
                print("[INFO] Ultralytics TPU yolo_v8 verification thread is running. ")
            else:
                # Start Ultralytics yolo8 verification thread
                client.publish("AI/Status", "Ultralytics CUDA yolo8 verification thread is running.", 2, True)
                print("[INFO] Ultralytics CUDA yolo_v8 verification thread is running. ")



    # starting rtsp threads can take a long time, send image and message to dasboard to indicate progress
    img = np.zeros(( imwinHeight, imwinWidth, 3), np.uint8)
    img[:,:] = (127,127,192)
    retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    client.publish("ImageBuffer/!Starting Camera stream threads, this can take awhile.", bytearray(img_as_jpg), 0, False)

    # *** start camera reading threads
    ### Try moving camera threads start up until after verification thread started
    o = list()
    if Nonvif > 0:
        import onvif_Thread
        print("\n[INFO] starting " + str(Nonvif) + " Onvif Camera Threads ...")
        client.publish("AI/Status", "Starting " + str(Nonvif) + " Onvif Camera Threads...", 2, True)
        for i in range(Nonvif):
            onvif_Thread.__CamName__ = CamName
            o.append(Thread(target=onvif_Thread.onvif_thread, args=(inframeQ[i], i, CameraURL[i])))
            o[i].start()
        time.sleep(1.0)     # so node-red UI can have a chance to see the message.
    if Nrtsp+Nfisheye > 0:
        global threadLock
        global threadsRunning
        threadLock = Lock()
        threadsRunning = 0
        for i in range(Nrtsp):
            rtsp_thread.__CamName__ = CamName
            o.append(Thread(target=rtsp_thread, args=(inframeQ[i+Nonvif], i+Nonvif, rtspURL[i])))
            o[i+Nonvif].start()
            client.publish("AI/Status", "Starting Camera : " + str(CamName[i+Nonvif])+ " RTSP Thread.", 2, True)
            time.sleep(6.0)
        FEoffset=FishEyeOffset
        for i in range(Nfisheye):
            Nfe=len(PTZparam[i])-1  # first entry is camera resolution, not PTZ view parameters
            #print(PTZparam[i])
            ### def FErtsp_thread(inframeQ, Nfe, FEoffset, PTZparam, camn, URL):
            o.append(Thread(target=FErtsp_thread, args=(inframeQ, Nfe, FEoffset, PTZparam[i], FEoffset+i, FErtspURL[i])))  # for virtual camera
            o[i+Nonvif+Nrtsp].start()
            client.publish("AI/Status", "Starting Fisheye Camera : " + str(CamName[FEoffset+i])+ " Thread.", 2, True)
            FEoffset+=Nfe
        # make sure rtsp threads are all running
        while threadsRunning < Nrtsp+Nfisheye:
            client.publish("AI/Status", str(threadsRunning) + " Of " + str(Nrtsp+Nfisheye) + " RTSP Threads Running", 2, True)
            time.sleep(2.0)
        print("\n[INFO] All " + str(Nrtsp+Nfisheye) + " RTSP Camera Sampling Threads are running.")



    #*************************************************************************************************************************************
    # *** enter main program loop (main thread)
    # loop over frames from the camera and display results from AI_thread
    excount=0
    aliveCount=0
    SEND_ALIVE=100  # send MQTT message approx. every SEND_ALIVE/fps seconds to reset external "watchdog" timer for auto reboot.
    waitCnt=0
    detectCount=0
    prevUImode=UImode
    currentDT = datetime.datetime.now()
    # *** MQTT send a blank image to the dashboard UI
    print("[INFO] Clearing dashboard ...")
    img = np.zeros(( imwinHeight, imwinWidth, 3), np.uint8)
    img[:,:] = (127,192,127)
    retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    client.publish("ImageBuffer/!AI has Started.", bytearray(img_as_jpg), 0, False)
    #start the FPS counter
    print("[INFO] starting the FPS counter ...")
    fps = FPS().start()
    print("\n[INFO] AI/Status: Python AI2 code is running." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"))
    client.publish("AI/Status", "Python AI2 code running." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"), 2, True)

    while not QUIT:
        try:
            try:
                (img, cami, personDetected, dt, ai, bp, yolo_frame) = resultsQ.get(True,0.100)  # perhaps yolo_frame should be zoom_frame instead
            except Exception as e:
                #print(e)
                waitCnt+=1
                img=None
                aliveCount = (aliveCount+1) % SEND_ALIVE   # MQTTcam images stop while Lorex reboots, recovers eventually so keep alive
                if aliveCount == 0:
                    client.publish("AmAlive", "true", 0, False)
                ##cv2.waitKey(1)
                continue
            if img is not None:
                fps.update()    # update the FPS counter
                # setup for file saving
                folder=dt.strftime("%Y-%m-%d")
                filename=dt.strftime("%H_%M_%S.%f")
                filename=filename[:-4] + "_" + ai  #just keep tenths, append AI source
                # setup for local save of yolo frame of zoomed image of detection
                # currently detection images saved by node-red if -ls option not active, I'm currently rethinking this
                yfolder=str(detectPath + "/" + folder)
                if os.path.exists(yfolder) == False:
                    os.mkdir(yfolder)
                    if not __DEBUG__ and not NoSaveZoom:
                        if os.path.exists(str(yfolder + "/zoom")) == False:
                            os.mkdir(str(yfolder + "/zoom"))     # put detection zoom into sub-folder
                #''' Debug code to see verification failure images
                if __DEBUG__:
                    ##if (OVyolo8_verify or yolo8_verify) and yolo_frame is not None:
                    if yolo_frame is not None:
                        if personDetected:
                            ##outName=str(yfolder + "/" + filename + "_" + "zoom_Cam" + str(cami) +"_AI.jpg")
                            outName=str(yfolder + "/zoom/" + filename + "_zoom-" + CamName[cami] +"-AI.jpg")
                        else:
                            if bp[0] == -1: # failed AI zoom redetection
                                ##outName=str(yfolder + "/" + filename + "_" + "ZnoV_Cam" + str(cami) +".jpg")
                                outName=str(yfolder + "/" + filename + "_ZnoV-" + CamName[cami] +"-.jpg")
                            if bp[0] == -2: # failed Yolo zoom detection
                                ##outName=str(yfolder + "/" + filename + "_" + "YnoV_Cam" + str(cami) +".jpg")
                                outName=str(yfolder + "/" + filename + "_YnoV-" + CamName[cami] +"-.jpg")
                        cv2.imwrite(outName, yolo_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                #'''
                if personDetected:  # personDetected implies yolo_frame is not None
                    detectCount+=1
                    if CLAHE:   # create CLAHE frame
                        if nodeRedFull:
                            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                        else:
                            lab = cv2.cvtColor(yolo_frame, cv2.COLOR_BGR2LAB)
                        lab_planes = cv2.split(lab)
                        lab_planes[0] = clahe.apply(lab_planes[0])
                        lab = cv2.merge(lab_planes)
                        CLAHE_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                        retv, img_as_jpg = cv2.imencode('.jpg', CLAHE_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])    # write clahe image to node-red
                    else:
                        if nodeRedFull:
                            retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])        # write full frame to node-red controller
                        else:
                            retv, img_as_jpg = cv2.imencode('.jpg', yolo_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])    # write zoomed image to node-red
                    if retv:
                        if nodeRedFull:
                            ##outName=str("AIdetection/!detect/" + folder + "/" + filename + "_" + "Full_Cam" + str(cami) +"_AI.jpg")
                            outName=str("AIdetection/!detect/" + folder + "/alert/" + filename + "-" + CamName[cami] +"-AI.jpg")
                        else:
                            ##outName=str("AIdetection/!detect/" + folder + "/" + filename + "_" + "Zoom_Cam" + str(cami) +"_AI.jpg")
                            outName=str("AIdetection/!detect/" + folder + "/alert/" + filename + "_Zoom-" + CamName[cami] +"-AI.jpg")
                        outName=outName + "!" + str(bp[0]) + "!" + str(bp[1]) + "!" + str(bp[2]) + "!" + str(bp[3]) + "!" + str(bp[4]) + "!" + str(bp[5]) + "!" + str(bp[6]) + "!" + str(bp[7])
                        client.publish(str(outName), bytearray(img_as_jpg), 0, False)
                        ##print(outName)  # log detections
                        if not NoLocalSave or __DEBUG__:
                            # save all AI person detections and zoom image no matter the ALARM_MODE, may change this later to not save in IDLE mode.
                            # part of Debug code to see yolo verification images
                            if not __DEBUG__ and not NoSaveZoom:
                                ##outName=str(yfolder + "/" + filename + "_" + "zoom_Cam" + str(cami) +"_AI.jpg")
                                outName=str(yfolder + "/zoom/" + filename + "_zoom-" + CamName[cami] +"-AI.jpg")
                                cv2.imwrite(outName, yolo_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])  # yolo frame is zoom frame if not y4v or y8v option
                            ##outName=str(yfolder + "/" + filename + "_" + "full_Cam" + str(cami) +"_AI.jpg")
                            outName=str(yfolder + "/" + filename + "-" + CamName[cami] +"-AI.jpg")
                            cv2.imwrite(outName, img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

                    else:
                        print("[INFO] conversion of np array to jpg in buffer failed!")
                        continue
                # send image for live display in dashboard
                if ((CameraToView == cami) and (UImode == 1 or (UImode == 2 and personDetected))) or (UImode ==3 and personDetected):
                    if personDetected:
                        ##topic=str("ImageBuffer/!" + filename + "_" + "Cam" + str(cami) +"_AI.jpg")
                        topic=str("ImageBuffer/!" + filename + "-" + CamName[cami] +"-AI.jpg")
                    else:
                        retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
                        if retv:
                            ##topic=str("ImageBuffer/!" + filename + "_" + "Cam" + str(cami) +".jpg")
                            topic=str("ImageBuffer/!" + filename + "-" + CamName[cami] +"-.jpg")
                        else:
                            print("[INFO] conversion of numpy array to jpg in buffer failed!")
                            continue
                    client.publish(str(topic), bytearray(img_as_jpg), 0, False)
                # display the frame to the screen if enabled, in normal usage display is 0 (off)
                if dispMode:
                    name=str("Live_" + CamName[cami])
                    cv2.imshow(name, cv2.resize(img, (imwinWidth, imwinHeight)))
                    key = cv2.waitKey(1) ###& 0xFF
                    ###if key == ord("q"): # if the `q` key was pressed, break from the loop
                    ###    QUIT=True   # exit main loop
                    if (OVyolo8_verify or yolo8_verify) and show_yolo and yolo_frame is not None:
                        if personDetected:
                            cv2.imshow("yolo_verify", yolo_frame)
                        else:
                            cv2.imshow("yolo_reject", yolo_frame)
                        key = cv2.waitKey(1) ### & 0xFF
                        ###if key == ord("q"): # if the `q` key was pressed, break from the loop
                        ###    QUIT=True   # exit main loop
                        if personDetected and show_zoom:
                            cv2.imshow("detection_zoom", yolo_frame)
                            cv2.waitKey(1)
                else:   # Handle -z and/or -y if dispMode is False
                    if show_zoom:
                        if personDetected:
                            cv2.imshow("detection_zoom", yolo_frame)
                            cv2.waitKey(1)
                    if (OVyolo8_verify or yolo8_verify) and show_yolo and yolo_frame is not None:
                        if personDetected:
                            cv2.imshow("yolo_verify", yolo_frame)
                        else:
                            cv2.imshow("yolo_reject", yolo_frame)
                        key = cv2.waitKey(1) ### & 0xFF

                aliveCount = (aliveCount+1) % SEND_ALIVE
                if aliveCount == 0:
                    client.publish("AmAlive", "true", 0, False)
                    cv2.waitKey(1)      # try to keep detection_zoom window display alive
                if prevUImode != UImode:
                    img = np.zeros(( imwinHeight, imwinWidth, 3), np.uint8)
                    img[:,:] = (154,127,100)
                    retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
                    client.publish("ImageBuffer/!AI Mode Changed.", bytearray(img_as_jpg), 0, False)
                    prevUImode=UImode
            ##else:   # img is None
            ##    cv2.waitKey(1)
        # if "ctrl+c" is pressed in the terminal, break from the loop
        except KeyboardInterrupt:
            QUIT=True   # exit main loop
            ##continue
        except Exception as e:
            currentDT = datetime.datetime.now()
            print(" **** Main Loop Error: " + str(e)  + currentDT.strftime(" -- %Y-%m-%d %H:%M:%S.%f"))
            excount=excount+1
            if excount <= 3:
                continue    # hope for the best!
            else:
                break       # give up! Hope watchdog gets us going again!
    #end of while not QUIT  loop
    #*************************************************************************************************************************************

    # *** Clean up for program exit
    fps.stop()    # stop the FPS counter timer
    currentDT = datetime.datetime.now()
    print("\n\n[INFO] Program Exit signal received:" + currentDT.strftime("  %Y-%m-%d %H:%M:%S"))
    # display FPS information
    print("*** AI processing approx. FPS: {:.2f} ***".format(fps.fps()))
    print("    [INFO] Run elapsed time: {:.2f}".format(fps.elapsed()))
    print("    [INFO] Frames processed by AI system: " + str(fps._numFrames))
    print("    [INFO] Person Detection by AI system: " + str(detectCount))
    print("    [INFO] Main loop waited for resultsQ: " + str(waitCnt) + " times.\n")

    # Send a blank image the dashboard UI
    print("[INFO] Clearing dashboard ...")
    img = np.zeros((imwinHeight, imwinWidth, 3), np.uint8)
    img[:,:] = (32,32,32)
    retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    client.publish("ImageBuffer/!AI has Exited.", bytearray(img_as_jpg), 0, False)
    time.sleep(1.0)
    currentDT = datetime.datetime.now()
    client.publish("AI/Status", "Python AI2 exiting." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"), 2, True)
    print("[INFO] AI/Status: Python AI2 exiting." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"))

    # stop Yolo v8 AI Thread
    if OVyolo8_verify:
        print("\n[INFO] Stopping OpenVINO yolo8 verification Thread ...")
        yolo8OpenvinoVerification_Thread.__Thread__ = False
        yolo8ov[0].join()
        print("[INFO] OpenVINO yolov8 verification Thread has exited.\n")
        client.publish("AI/Status", "OpenVINO yolov8 verification Thread has exited.", 2, True)
    # Stop and wait for capture threads to exit
    if Nonvif > 0:
        print("[INFO] Stopping Onvif camera threads ...")
        client.publish("AI/Status", "Waiting for ONVIF camera Threads to exit...", 2, True)
        onvif_Thread.__onvifThread__ = False
        for i in range(Nonvif):
            o[i].join()
        print("[INFO] All ONVIF camera threads have exited.")
        client.publish("AI/Status", "All ONVIF camera Threads have exited.", 2, True)
    if Nrtsp > 0:
        print("[INFO] Stopping RTSP camera threads ...")
        client.publish("AI/Status", "Waiting for RTSP camera Threads to exit...", 2, True)
        __rtspThread__ = False
        for i in range(Nrtsp):
            o[i+Nonvif].join()
        print("[INFO] All RTSP camera threads have exited.")
        client.publish("AI/Status", "All RTSP camera Threads have exited.", 2, True)
    if Nfisheye > 0:
        print("[INFO] Stopping Fisheye camera threads ...")
        client.publish("AI/Status", "Waiting  for Fisheye camera Threads to exit...", 2, True)
        __fisheyeThread__ = False
        for i in range(Nfisheye):
            o[i+Nonvif+Nrtsp].join()
        print("[INFO] All Fisheye camera threads have exited.")
        client.publish("AI/Status", "All Fisheye camera Threads have exited.", 2, True)

    # stop TPU AI thread
    if nCoral is True:
        print("\n[INFO] Stopping TPU Thread ...")
        Coral_TPU_Thread.__Thread__ = False   # maybe my QUITf() was clenaer, but I can stage the thread exits for debugging.
        Ct[0].join()
        print("[INFO] Coral TPU AI Thread has exited.\n")
        client.publish("AI/Status", "Coral TPU thread has exited.", 2, True)

    # Stop OpenVINO CPU thread
    if nCPUthreads is True:
        print("\n[INFO] Stopping openvino CPU AI  Thread ...")
        OpenVINO_SSD_Thread.__Thread__ = False
        CPUt[0].join()
        print("[INFO] CPU AI Thread has exited.\n")
        client.publish("AI/Status", "CPU AI  Thread has exited.", 2, True)

    #$$$#  stop yolo verify thread
    if yolo8_verify or TPUyolo8_verify:
        if TPUyolo8_verify:
            print("\n[INFO] Stopping TPU yolo8 verification Thread ...")
        else:
            print("\n[INFO] Stopping CUDA yolo8 verification Thread ...")
        yolo8_verification_Thread.__Thread__ = False
        yolo8[0].join()
        if TPUyolo8_verify:
            print("[INFO] yolov8 TPU verification Thread has exited.\n")
            client.publish("AI/Status", "TPU yolov8 verification Thread has exited.", 2, True)
        else:
            print("[INFO] yolov8 CUDA verification Thread has exited.\n")
            client.publish("AI/Status", "CUDA yolov8 verification Thread has exited.", 2, True)

    # destroy all windows if we are displaying them
    if args["display"] > 0:
        cv2.destroyAllWindows()

    # clean up MQTT
    client.publish("AI/Status", "AI2 Program has exited.", 2, True)
    time.sleep(2.0)
    client.disconnect()     # normal exit, Will message should not be sent.
    currentDT = datetime.datetime.now()
    print("[INFO] Stopping MQTT Threads." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"))
    client.loop_stop()      ### Stop MQTT thread


    # bye-bye
    currentDT = datetime.datetime.now()
    print("Program Exit." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"))
    print("$$$**************************************************************$$$")
    print("")
    print("")




# *** RTSP Sampling Thread
#******************************************************************************************************************
# rtsp stream sampling thread
### 20JUN2019 wbk much improved error handling, can now unplug & replug a camera, and the thread recovers
def rtsp_thread(inframeQ, camn, URL):
    global threadLock
    global threadsRunning
    global __rtspThread__
    global __CamName__
    global CamName
    
    __rtspThread__ = True
    __CamName__ = CamName
    ocnt=0
    Error=False
    Error2=False
    currentDT = datetime.datetime.now()
    print("[INFO] RTSP stream sampling thread " + __CamName__[camn] + " is starting..." + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
    Rcap=cv2.VideoCapture(URL)
    Rcap.set(cv2.CAP_PROP_BUFFERSIZE, 2)     # doesn't throw error or warning in python3, but not sure it is actually honored
    threadLock.acquire()
    currentDT = datetime.datetime.now()
    print("[INFO] RTSP stream sampling thread " + __CamName__[camn] + " is running..." + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
    threadsRunning += 1
    threadLock.release()
    while __rtspThread__ is True:
         # grab the frame
        try:
            if Rcap.isOpened() and Rcap.grab():
                gotFrame, frame = Rcap.retrieve()
            else:
                frame = None
                if not Error:
                    Error=True
                    currentDT = datetime.datetime.now()
                    print('[Error!] RTSP Camera '+ __CamName__[camn] + ': ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S") +
                          '\n    ' + URL[0:38] + '\n        Will close and re-open Camera ' + __CamName__[camn] +' RTSP stream in attempt to recover.')
                # try closing the stream and reopeing it, I have one straight from China that does this error regularly
                # NOTE this only detects rstp connection loss, if the DVR sends black frames on camera loss this will not detect it!
                Rcap.release()
                if not Error2:
                    time.sleep(60.0)     # does this help or hurt or no difference? Always Seems to take a minute of more to recover
                Rcap=cv2.VideoCapture(URL)
                Rcap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                if not Rcap.isOpened() :
                    if not Error2:
                        Error2=True
                        currentDT = datetime.datetime.now()
                        ##print('   [Error2!] RTSP stream'+ str(camn) + ' re-open failed!' + currentDT.strftime(" %Y-%m-%d %H:%M:%S") +
                        ##     '\n   Will loop closing and re-opening Camera' + str(camn) +' RTSP stream, further messages suppressed.')
                        print('   [Error2!] RTSP stream '+ __CamName__[camn] + ' re-open failed!' + currentDT.strftime(" %Y-%m-%d %H:%M:%S") +
                              '\n   Will loop closing and re-opening Camera ' + __CamName__[camn] +' RTSP stream, further messages suppressed.')
                    time.sleep(30.0)
                continue
            if gotFrame: # path for sucessful frame grab, following test is in case error recovery is in progress
                if Error:   # log when it recovers
                    currentDT = datetime.datetime.now()
                    print('[$$$$$$] RTSP Camera '+ __CamName__[camn] + ' has recovered: --- ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S") +
                          '\n    ' + URL[0:38] + "\n")
                    Error=False    # after geting a good frame, enable logging of next error
                    Error2=False
        except Exception as e:
            frame = None
            currentDT = datetime.datetime.now()
            print('[Exception] RTSP stream '+ __CamName__[camn] + ': ' + str(e) + '\n    ' + URL[0:38] + ' :  ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
            time.sleep(10.0)
        try:
            if frame is not None:
                imageDT=datetime.datetime.now()
                if inframeQ.full():
                    [_,_,_]=inframeQ.get(False)    # remove oldest sample to make space in queue
                    ocnt+=1     # if happens here shouldn't happen below
                inframeQ.put((frame.copy(), camn, imageDT), False)   # no block if queue full, go grab fresher frame
        except: # most likely queue is full, Python queue.full() is not 100% reliable
            # a large drop count for rtsp streams is not a bad thing as we are trying to keep the input buffers nearly empty to reduce latency.
            ocnt+=1
    Rcap.release()
    print("RTSP stream sampling thread " + __CamName__[camn] + " is exiting, dropped frames " + str(ocnt) + " times.")




## Fisheye Window snippet
# --> https://github.com/daisukelab/fisheye_window
class FishEyeWindow(object):
    """Fisheye Window class
    You can get image out of your fisheye image for desired view.
    1. Create instance by feeding image sizes.
    2. Call buildMap to set the view you want.
       This calculates the map for the 'remap.'
    3. Call getImage that simply remaps.
    """
    def __init__(
            self,
            srcWidth,
            srcHeight,
            destWidth,
            destHeight
        ):
        # Initial parameters
        self._srcW = srcWidth
        self._srcH = srcHeight
        self._destW = destWidth
        self._destH = destHeight
        self._al = 0
        self._be = 0
        self._th = 0
        self._R  = srcWidth / 2
        self._zoom = 1.0
        # Map storage
        self._mapX = np.zeros((self._destH, self._destW), np.float32)
        self._mapY = np.zeros((self._destH, self._destW), np.float32)
    def buildMap(self, alpha=None, beta=None, theta=None, R=None, zoom=None):
        # Set the angle parameters
        self._al = (alpha, self._al)[alpha == None]
        self._be = (beta, self._be)[beta == None]
        self._th = (theta, self._th)[theta == None]
        self._R = (R, self._R)[R == None]
        self._zoom = (zoom, self._zoom)[zoom == None]
        # Build the fisheye mapping
        al = self._al / 180.0
        be = self._be / 180.0
        th = self._th / 180.0
        A = np.cos(th) * np.cos(al) - np.sin(th) * np.sin(al) * np.cos(be)
        B = np.sin(th) * np.cos(al) + np.cos(th) * np.sin(al) * np.cos(be)
        C = np.cos(th) * np.sin(al) + np.sin(th) * np.cos(al) * np.cos(be)
        D = np.sin(th) * np.sin(al) - np.cos(th) * np.cos(al) * np.cos(be)
        mR = self._zoom * self._R
        mR2 = mR * mR
        mRsinBesinAl = mR * np.sin(be) * np.sin(al)
        mRsinBecosAl = mR * np.sin(be) * np.cos(al)
        centerV = int(self._destH / 2.0)
        centerU = int(self._destW / 2.0)
        centerY = int(self._srcH / 2.0)
        centerX = int(self._srcW / 2.0)
        # Fill in the map, slows dramatically with large view (destination) windows
        for absV in range(0, int(self._destH)):
            v = absV - centerV
            vv = v * v
            for absU in range(0, int(self._destW)):
                u = absU - centerU
                uu = u * u
                upperX = self._R * (u * A - v * B + mRsinBesinAl)
                lowerX = np.sqrt(uu + vv + mR2)
                upperY = self._R * (u * C - v * D - mRsinBecosAl)
                lowerY = lowerX
                x = upperX / lowerX + centerX
                y = upperY / lowerY + centerY
                _v = (v + centerV, v)[centerV <= v]
                _u = (u + centerU, u)[centerU <= u]
                self._mapX.itemset((_v, _u), x)
                self._mapY.itemset((_v, _u), y)

    def getImage(self, img):
        # Look through the window
        output = cv2.remap(img, self._mapX, self._mapY, cv2.INTER_LINEAR)
        #output = cv2.remap(img, self._mapX, self._mapY, cv2.INTER_CUBIC) # no significant improvement on 4 Mpixel test image
        return output


# create virtual cameras from PTZ crops from a fisheye camera rtsp stream
# Note the PTZ param are string variables read from the fisheye.rtsp text file
# created with the interactive fisheye_window C++ utility program.
def FErtsp_thread(inframeQ, Nfe, FEoffset, PTZparam, camn, URL):
    global __fisheyeThread__
    global threadLock
    global threadsRunning
    __fisheyeThread__ = True
    ocnt=[]
    for i in range(Nfe):
        ocnt.append(0)      # init counter array
    fe=[]
    Error=False
    Error2=False
##    print(PTZparam)
    threadLock.acquire()
    mapFilename="fisheye" +str(camn)+ "_map"
    try:
      filehandler = open(mapFilename, 'rb')
      currentDT = datetime.datetime.now()
      print( "Loading " + mapFilename + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
      fe = pickle.load(filehandler)
      filehandler.close()
    except:
      currentDT = datetime.datetime.now()
      print( "Creating " + mapFilename + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
      for i in range(Nfe):
        if i == 0:
            srcW=int(PTZparam[0][0])
            srcH=int(PTZparam[0][1])
        # PTZparam = [ [srcW,srcH], [destW, destH, alpha, beta, theta, zoom], [...] ] chosen with fisheye_window  C++ utility
        print("FE" +str(camn)+ " PTZview" +str(i)+ " " +str(PTZparam[i+1]))
        fe.append(FishEyeWindow(srcW, srcH, int(PTZparam[i+1][0]), int(PTZparam[i+1][1])))    # instance a view with desired output image size
        fe[i].buildMap(alpha=float(PTZparam[i+1][2]), beta=float(PTZparam[i+1][3]),
                       theta=float(PTZparam[i+1][4]), zoom=float(PTZparam[i+1][5]))    # build map for this PTZ view
      currentDT = datetime.datetime.now()
      print("Saving FE" +str(camn)+ " virtual PTZ views as: " + mapFilename + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
      filehandler = open(mapFilename, 'wb')
      pickle.dump(fe, filehandler)
      filehandler.close()
    currentDT = datetime.datetime.now()
    print("[INFO] Fisheye Camera RTSP stream FE" + str(camn) + " is opening..." + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
    Rcap=cv2.VideoCapture(URL)
    Rcap.set(cv2.CAP_PROP_BUFFERSIZE, 2)     # doesn't throw error or warning in python3, but not sure it is actually honored
#    threadLock.acquire()
    currentDT = datetime.datetime.now()
    print("[INFO] Fisheye RTSP stream sampling thread" + str(camn) + " is running..." + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
    threadsRunning += 1
    threadLock.release()
    while __fisheyeThread__ is True:
         # grab the frame
        try:
            if Rcap.isOpened() and Rcap.grab():
                gotFrame, frame = Rcap.retrieve()
            else:
                frame = None
                if not Error:
                    Error=True
                    currentDT = datetime.datetime.now()
                    print('[Error!] RTSP Camera'+ str(camn) + ': ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S") +
                          '\n   ' + URL[0:38] + '\n   Will close and re-open Camera' + str(camn) +' RTSP stream in attempt to recover.')
                # try closing the stream and reopeing it, I have one straight from China that does this error regularly
                # NOTE this only detects rstp connection loss, if the DVR sends black frames on camera loss this will not detect it!
                Rcap.release()
                time.sleep(5.0)     # does this help or hurt?
                Rcap=cv2.VideoCapture(URL)
                Rcap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                if not Rcap.isOpened() :
                    if not Error2:
                        Error2=True
                        currentDT = datetime.datetime.now()
                        print('[Error2!] RTSP stream'+ str(camn) + ' re-open failed! $$$  --- ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S") +
                              '\n   Will loop closing and re-opening Camera' + str(camn) +' RTSP stream, further messages suppressed.')
                    time.sleep(10.0)
                continue
            if gotFrame: # path for sucessful frame grab, following test is in case error recovery is in progress
                if Error:   # log when it recovers
                    currentDT = datetime.datetime.now()
                    print('[$$$$$$] RTSP Camera'+ str(camn) + ' has recovered: --- ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S") +
                          '\n   ' + URL[0:38] + "\n")
                    Error=False    # after geting a good frame, enable logging of next error
                    Error2=False
        except Exception as e:
            frame = None
            currentDT = datetime.datetime.now()
            print('[Exception] RTSP stream'+ str(camn) + ': ' + str(e) + '\n ' + URL[0:38] + ' --- ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
            time.sleep(10.0)
        if frame is not None:
            imageDT = datetime.datetime.now()
            for i in range(Nfe):
                try:
                    if inframeQ[FEoffset+i].full():
                        [_,_,_]=inframeQ[FEoffset+i].get(False)    # remove oldest sample to make space in queue
                        ocnt[i]+=1   # it this happens here, it shouldn't happen below
                    PTZview=fe[i].getImage(frame)
                    inframeQ[FEoffset+i].put((PTZview.copy(), FEoffset+i, imageDT), True)  ## force this frame to complete in all queues
                except: # most likely queue is full, Python queue.full() is not 100% reliable
                    # a large drop count for rtsp streams is not a bad thing as we are trying to keep the input buffers nearly empty to reduce latency.
                    ocnt[i]+=1

    Rcap.release()
    print("RTSP Fisheye Camera sampling thread" + str(camn) + " is exiting ...")
    for i in range(Nfe):
        print("   Fisheye Cam "+ str(FEoffset+i) +" dropped frames " + str(ocnt[i]) + " times.")






# python boilerplate
if __name__ == '__main__':
    main()


