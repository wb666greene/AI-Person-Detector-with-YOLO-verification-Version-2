# AI-Person-Detector-with-YOLO-verification-Version-2
This is "Version 2" of the repo: https://github.com/wb666greene/AI-Person-Detector-with-YOLO-Verification.  The major difference is a much easier install since nearly everything can be done with PIP and a python virtual environment.  All the support for old hardware not capable of running yolo8 has been removed, since yolo8 is the key to the very low false alert rate.  I've tested it on Ubuntu 20.04 and 22.04, other Linux distributions with openvino and/or cuda support should work.  
### New 5SEP2024:
Ultralytics export of yolo8 model to Coral TPU edgetpu.tflite format.  The "nano" model is just not good enough.  The "small" model seems to work well, but had to reduce the size from 640 to 512 so it would fit in the TPU.  With two TPUs most any system capable of decoding multiple HD/UHD video streams should now be usable if you can find the TPU driver support for it.  This brings the Raspberry Pi 4/5 back into play as a host for a small system.  If you want try it on a Raspberry Pi these should help get you started with the drivers:
```
https://www.hackster.io/stefanw1/yolov8-on-raspberry-pi5-with-coral-tpu-8efd23
https://docs.ultralytics.com/guides/raspberry-pi/
https://github.com/StefansAI/Yolov8_Rpi5_CoralUSB/tree/main
https://github.com/DAVIDNYARKO123/edge-tpu-silva
```
### New 31AUG2024:
Everything is now pip installable in a virtual environment, so running it on Windows or WSL shouldn't be too difficult, it is not anything I'm interested in doing, but if you manage to run it on Windows or WSL please submit your instructions/patches so I can add them here.  Raise issues and I'll try to help out, especially if I broke Windows/Linux compatability in my python code (which is likely becasue of file naming conventions, especially case sensitivity).

These are my notes for a virgin installation of Ubuntu 22.04 and installation of OpenVINO 2024.3 release 
for using Intel integrated GPU or Nvidia GPU for yolo verification.  Motivated by 
https://igor.technology/installing-coral-usb-accelearator-python-3-10-ubuntu-22/
Which allows using the TPU on system python 3.10 eliminating the need for virtual environments for the basic AI,
although I still strongly recommend using virtual environments since breaking old code has never been much of a consideration for OpenVINO.
It has also been installed and tested on Ubuntu-Mate 20.04, comments in the copy-and-paste command sections note the differences.

Here is an image sequence that shows the software in action, made from the detection images as a solicitor walks from my neighbor's yard, across the street, to my front door, leaves his flyer, and proceeds across my yard on to the next house. He walks across the field of view of multiple cameras with various resolutions of 4K, 5Mpixel, & 1080p, this system used GTX1060 CUDA yolo8 and Coral M.2 TPU: 
https://youtu.be/XZYyE_WsRLI

This is the first frame of the sequence so you can get an idea of the sensitivity of the system.
![07_03_48 82_TPU_CliffwoodSouth_AI](https://github.com/user-attachments/assets/ec331062-cfd3-4e68-95ba-3e749ee691a2)


# 0) Install Ubuntu 22.04
I used Ubuntu-Mate.  Customize it as you see fit.  It has also been tested on Ubuntu 20.04, with differences in the installation commands pointed out when you'll encounter them.

If you are new to Linux, first thing after installing Ubuntu-Mate, go through the "Welcome" tutorial to learn the basics and then go to the "Control Center" and open "MATE Tweak" and from the sidebar select "Panel".  If you choose "Redmon" from the dropdown you'll get a destop that resembles Windows, if you choose "Cupetino" you'll get a Mac-like desktop.  If you can tolerate the Ubuntu "Unity" Desktop, fine, but I can't, and if coming from Windows or Mac be prepared for "eveything you know is wrong" when it comes to using the Unity desktop UI.

All my Python code, OpenVINO, node-red, Coral TPU drivers (if you need/want them) and CUDA (if you are using nVidia GPU instead of Intel) should be available on Windows, but I've not tested it.  But you'll lose the "housekeeping" functionality in some shell scripts that are called via node-red exec nodes in my minimal web based "controller".  Minimal is a design feature, I wanted a "set and forget" appliance that runs 24/7/365 and is controlled by your "home automation" or a web browser to set one of three modes of operation.  "Notify" mode sends Email and audio alerts,  "Audio" uses "Espeak" speech synthsizer to announce what camera a person has been detected on, and Idle just saves detection without any nodifications.

Use your username where I have "ai" and your hostname where I have "YouSeeX1".
To avoid having to edit the scripts used by node-red make a sym link in /home:
```
sudo ln -s /home/YourUserName /home/ai
```

# 1a) Install needed packages
Make sure system is up to date:
```
sudo apt update
sudo apt upgrade
```
I like loging in remotely over via ssh, as running "headless" is a design goal, but it can all be done with a terminal window as well. OpenSSH is not installed by default so either install "OpenSSH" using "Control Center Software Boutique" or in a terminal window do:
```
sudo apt install ssh
```
Now in a terminl window or your remote shell login, install these extra packages with:
```
sudo apt install git curl samba python3-dev python3-pip python3.10-venv espeak mosquitto mosquitto-dev mosquitto-clients cmake
# if using 20.04 change python3.10-venv to python3.8-venv in the above command.
```
# 1b) Setup to use the Coral TPU if you plan to use one for SSD or YOLO8.
NOTE: on my i9-12900 I get ~42 frames per second for MobilenetSSD_v2 with both OpenVINO CPU and Coral TPU, but I believe that this is limited by the aggregate frame rate of my six onvif cameras.  It is on lesser hardware, like i3, where the TPU is well worth its modest cost, adding two lets you do yolo8 on the TPU as well.

Add the "current" coral repo:
```
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install libedgetpu1-std
```
If you want TPU to run full speed:
```
sudo apt install libedgetpu1-max
```

If you want to use the M.2, mPCIe, etc. TPUs:
Depending on when and where you install you can get different kernel versions.
Virgin install on 1AUG2024 on Celeron got: 6.5.0-45-generic #45~22.04.1-Ubuntu
on my i9 setup in 2022 I got: 5.15.0-113-generic #123-Ubuntu
Don't install this if you have v6 kernel but it is not fatal, can "sudo apt remove" it later if necessary.
```
sudo apt install gasket-dkms
```
NOTE: the dkms build fails on kernel v6.+ what worked to fix it:
```
sudo apt remove gasket-dkms     # only if you did apt install gasket-dkms on v6+ kernel
sudo apt install dkms devscripts debhelper -y
git clone https://github.com/google/gasket-driver.git
cd gasket-driver; debuild -us -uc -tc -b; cd ..
sudo dpkg -i gasket-dkms_1.0-18_all.deb
```
Next setup the apex user and verify the PCI-E installation:
```
sudo sh -c "echo 'SUBSYSTEM==\"apex\", MODE=\"0660\", GROUP=\"apex\"' >> /etc/udev/rules.d/65-apex.rules"
sudo groupadd apex
#!!! Make sure the following command is being done as ai user, not as sudo -i
sudo adduser $USER apex
```
Now reboot the system.
Once rebooted, verify that the accelerator module is detected:
```
lspci -nn | grep 089a
```
You should see something like this:
03:00.0 System peripheral: Device 1ac1:089a
The 03 number and System peripheral name might be different, because those are host-system specific, 
but as long as you see a device listed with 089a then you're okay to proceed.

Also verify that the PCIe driver is loaded:
```
ls /dev/apex_0
```
You should simply see the name repeated back:
/dev/apex_0

Plug in a USB TPU and do:
```
lsusb
```
And you should see something like: "ID 18d1:9302 Google Inc." in the listing, or "ID 1a6e:089a Global Unichip Corp." for older TPUs.
If in doubt, do lsusb before plugging in the TPU, and then do it again to see if the device is added.

#### Finally download the pycoral python module for python 3.10
This article showed me how:
https://igor.technology/installing-coral-usb-accelearator-python-3-10-ubuntu-22/
Download the python "wheel" files for python10 from:
```
https://github.com/hjonnala/snippets/tree/main/wheels/python3.10
# I downloaded them to the TPU_python3.10 directory.
```

#### If using 20.04 you need the python3.8 versions instead:
```
python3 -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
# This is the easiest way, but it doesn't support python newer than 3.9.
```
We'll use these when we create the virtual environment below.

# 2) Create Python Virtual environment and install the needed python modules
Note Conda works well too, and I prefer it if you need a different python version than the system python3, but Conda  has issues being launched from a shell script via a node-red exec node.  If you know a solution, please send it to me.  I'm not very good with GitHub so you may need to help me with doing "pull requests" if it is not something you can just Email to me.
### Genarally you only need to setup one of these virtual environments.
All include OpenVINO, so that CPU MobilenetSSD_v2 can be used for the initial AI, although a TPU works better.
## 2a) Create a virtual environment to to use with OpenVINO and YOLO8, named y8ovv.
NOTE: If using CUDA skip to section 2b and create a venv named y8cuda. If using TPU yolo skip to section 2c below and create a venv named y8tpu.  I had conflicts trying to get both OpenVINO iGPU and CUDA working in the same environment.  It is not really a big limitation as it seems that CUDA and OpenCL can't both be used at the same time for GPU acceleration.
```
python3 -m venv y8ovv
```
Next "activate" the virtual environment:
```
source y8ovv/bin/activate
```
Note that the prompt changes from: ai@YouSeeX1: to: (y8ovv) ai@YouSeeX1:
Use pip to install the needed modules:
```
pip install -U pip setuptools
pip install imutils paho-mqtt requests
pip install opencv-python
pip install "openvino>=2024.2.0" "nncf>=2.9.0"
pip install "torch>=2.1" "torchvision>=0.16" "ultralytics==8.2.24" onnx tqdm opencv-python --extra-index-url https://download.pytorch.org/whl/cpu
# if using TPU
pip install TPU_python3.10/tflite_runtime-2.5.0.post1-cp310-cp310-linux_x86_64.whl
pip install TPU_python3.10/pycoral-2.0.0-cp310-cp310-linux_x86_64.whl
# or if using 20.04, easier but doesn't support newer than python 3.9
python3 -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0

```
If necessary, exit the virtual environment with "deactive" and the (y8ovv) prefix should disappear from the prompt.

Install the Intel GPU driver:
```
sudo apt-get install intel-opencl-icd
```
Add the ai user (your login if you didn't use ai as your username at installation) to the render group:
Make sure the following command is being done as user, not as sudo -i
```
sudo adduser $USER render
```
Now log out and login or reboot the system. GPU doesn't work if you don't do this!
Instructions for using CUDA will be given below, but you will still need OpenVINO for the MobilenetSSD_v2 initial AI if not using the Coral TPU, whos instructions are also given below.


## 2b) Install CUDA and create virtual environment:
### 31AUG2024, I learned that CUDA is now pip installable, making the setup very similar to the openvino version.
First install cuda. I'm not a cuda expert, this is what I installed on an i3-10100F with GTX950.  
Nice background is here, but you can skip all these and just do the pip installation:
```
https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#linux  # Use sidebar to select "3.1.9 Ubuntu-->3.1.9.2 Runfile Installer"
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
https://developer.nvidia.com/cuda-toolkit-archive

# My initial success with cuda and cudnn came from what I learned from these:
https://medium.com/@juancrrn/installing-cuda-and-cudnn-in-ubuntu-20-04-for-deep-learning-dad8841714d6
https://davidblog.si/2022/12/24/setting-up-cuda-and-cudnn-in-ubuntu-22-04/
```
I think it is safest to use the Ubuntu repo version of the nvidia driver. I always choose the driver marked "tested" which in this case was nvidia-driver-535 using the "Additional Drivers" tab in the "Software & Updates" app if Noveau driver was installed by default.  Install it and reboot the system. Open a terminal and run the nvidia-smi command, if it works, so far so good.  

## Make the virtual environment for CUDA:
Get the latest PyTorch here: https://pytorch.org/get-started/locally/
Older versions (what I used): https://pytorch.org/get-started/previous-versions/
The websites give you a pip install command you can cut and paste to install your version of torch with cuda support that you use below after the pip install nvidia-cuda-runtime-cu12.

```
python3 -m venv y8cuda
source y8cuda/bin/activate
# promp changes to: (y8cuda)
pip install --upgrade setuptools pip wheel
pip install nvidia-pyindex
pip install nvidia-cuda-runtime-cu12
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install imutils paho-mqtt requests
pip install opencv-python
pip install "openvino>=2024.2.0" "nncf>=2.9.0"
pip install ultralytics
# Only if uisng TPU
pip install TPU_python3.10/tflite_runtime-2.5.0.post1-cp310-cp310-linux_x86_64.whl
pip install TPU_python3.10/pycoral-2.0.0-cp310-cp310-linux_x86_64.whl
# or if using 20.04, easier but doesn't support newer than python 3.9
python3 -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
``
In the virtual environment test the cuda/pytorch install with:
```
(y8cuda) ai@GTX950:~$ python
```
Python 3.10.12 (main, Jul 29 2024, 16:56:48) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.cuda.is_available()
True
>>> #Ctrl-d to exit
```
## 2c) Create virtual environment for TPU yolo8:
(Don't forget sudo apt install python3.X-venv, if you didn't do it in step 0)
```
# set up virtual environment for yolo8 on Coral TPU
python3 -m venv y8tpu
source y8tpu/bin/activate
# prompt should change: to  (y8tpu) ai@hostname:~$
pip install -U pip setuptools
pip install TPU_python3.10/tflite_runtime-2.5.0.post1-cp310-cp310-linux_x86_64.whl
pip install TPU_python3.10/pycoral-2.0.0-cp310-cp310-linux_x86_64.whl
# or if using 20.04, easier but doesn't support newer than python 3.9
python3 -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
pip install opencv-python
#install openvino, not strictly necessary, but need it if you want CPU MobilenetSSD_v2 AI
pip install "openvino>=2024.2.0" "nncf>=2.9.0"
pip install "torch>=2.1" "torchvision>=0.16" "ultralytics==8.2.24" onnx tqdm opencv-python --extra-index-url https://download.pytorch.org/whl/cpu
pip install imutils paho-mqtt requests
```
Nothing special needs to be done except for installing the TPU drivers for multiple TPUs.  Two USB can skip the M.2 steps, The USB should all be taken care of by the apt install.

# 3) Clone this repo
```
git clone https://github.com/wb666greene/AI-Person-Detector-with-YOLO-verification-Version-2.git
mv AI-Person-Detector-with-YOLO-verification-Version-2 AI2
```
#### Don't skip the mv (rename) of the directory to AI2, otherwise you'll need to edit the node-red scripts to account for the different names. 

```
cd AI2
# make sure the five *.sh scripts have the execute bit set.
chmod ugo+x *.sh
```
NOTE.  The models are too large to upload to GitHub and for some it may not be allowed.  The Ultralytics yolo8 model is automatically downloaded and converted to openvino format if the files are not there already.  But the MobilnetSSDv2 for the TPU and openvino are not.  

### You can download the TPU model from:  
```
https://raw.githubusercontent.com/google-coral/test_data/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
```
Put it in the AI2/mobilenet_ssd_v2 folder, the coco_labels.txt should already be there.
If you have a Coral TPU install its support as shown in section 5)

### For the openvino CPU initial detection, download SSD MobileNet V2 model:
```
cd $HOME # should be one level above the AI2 directory
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar -zxf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
```
First time AI2.py runs with the openvino CPU thread it will automatically convert the model and save it to the AI2/mobilenet_ssd_v2 directory, eliminating the conversion time for subsequent runs.
Unfortunately openvino 2024.x conversion produces a model that has multiple outputs, which broke my code. I believe the issue stems from openvino 2022.1 that introduced a new IR11 format and the backwards compatabily seems to have been lost with 2024 versions.  I made my code automatically convert the downloaded model and use the multiple outputs, but there is still an issue in that it only detects one person in the frame even if multiple persons are present. Since I break out of the detection loop after the first person is detected with a confidence above the threshold, it is not a show-stopper, but could cause issues.  It works fine in limited testing, but I've an issue open at the Intel OpenVINO Community Forum:

https://community.intel.com/t5/Intel-Distribution-of-OpenVINO/Having-trouble-with-model-convert-and-tensorflow-model-with-pip/m-p/1624526#M31331

And will update the code and this README.md if a solution is found.


## Next we need to tell the python code how to talk to the cameras. 
Two types of cameras are supported, Onvif and RTSP.  Onvif is an overly complited "standard" that is rarely implimented fully or correctly but if you can retrieve an image with an HTTP request it is an Onvif camera for our purposes.  RTSP opens a connection on port 554 and returns a video stream, these are the most common type of cameras.  Be aware that RING, SimplySafe, Arlo, Blink etc. don't generally allow direct access to the video streams or still images.  Also the low end securtiy DVRs like Swann, NightOwl also usually lack support for RTSP streams.  Before you buy, make sure the camera or DVRs you are condidering support RTSP streams and or "Onvif snapshots".

To tell the python code how to use your cameras you need to create a cameraURL.txt file for Onvif cameras or a cameraURL.rtsp file for RTSP cameras.  A cameraURL.txt file should one line per camera containing the HTTP URL to retrieve and image and after a space contain the optional name for what the camera is viewing.
A cameraURL.txt file for two cameras would look like this:
```
http://192.168.2.144:85/images/snapshot.jpg Driveway
http://admin:password@192.168.2.197:80/tmpfs/auto.jpg  Garage
```
Note than some cameras don't require any authorization to return the image, others do and the URLs can be very different format.  It is easy to test your URL simpley by entering into a web browser and you should see and image after connecting and a fresh image after refreshing the page.

A cameraURL.rtsp file should look like this:
```
rtsp://admin:passwd@192.168.2.28:554/cam/realmonitor?channel=1&subtype=0  MailBox
rtsp://admin:passwd%@192.168.2.28:554/cam/realmonitor?channel=7&subtype=0  DriveWay
rtsp://admin:passwd@192.168.2.49:554/cam/realmonitor?channel=3&subtype=0  Garage
rtsp://admin:PassWd@192.168.2.49:554/cam/realmonitor?channel=3&subtype=0  Patio
rtsp://admin:PaSwrd@192.168.2.196:554/h264Preview_01_main FrontYard
rtsp://192.168.2.77:554/stream0?username=admin&password=CE04588A3231F95BEE71848F5958CF4E BackYard
```
The The IP addresses will be what your router assigns, my example shows two security DVRs at xxx.xxx.xxx.28 and xxx.xxx.xxx.49 and two IP or "netcams" the last netcam generates the password as part of the Onvif setup and can be "tricky" to figure out, but they are not generally unique.  Best way I know to test your RTSP URLs is with VLC and "Open Network Connection" and enter the RTSP URL string.  Main negative of RTSP streams is the latency is typically 2-4 seconds behind "real time".

### Make sure the system is connected to the Internet, so the first run can download and convert the yolo8 models.
Once the camera URLs are specified (both types of camera files are allowed) we can run a quick test, make sure the virtual environment is active and open up a command window (terminal):
#### Some python modules may get updated during the conversion, be patient it can take a couple of minutes.
This is why it is strongly recommended to do the first run in a terminal so you can see the downloading and conversion take place, and see any error messages should you have forgetten a step.
```
source y8ovv/bin/activate
python AI2.py -d -y8ovv
```
Or if using cuda:
```
source y8cuda/bin/activate
python AI2.py -d -y8v
```
Or if using a TPU:
```
source y8tpu/bin/activate
python AI2.py -d -y8tpu
```
Or if using two TPUs
```
source y8tpu/bin/activate
python AI2.py -d -tpu -y8tpu
```
This will start a thread for each video camaera and an OpenVINO SSD CPU (or TPU) AI thread and and a yolo8 verification thread, and display live results on the screen.  Here is an image of it running on a Lenovo IdeaPad i3 laptop doing six Onvif cameras:
![IdeaPad](https://github.com/user-attachments/assets/647f3c54-a265-406a-833d-0ef1a2883eec)  Notifications and the housekeeping functions are done with node-red which we will install next.  Press Ctrl-C in the terminal window to exit the AI python code.  My GTX950 had a YOLO8 inference time of about 59 mS with about 5 mS pre & post processing times.
#### If you have a "fish eye" camera and want to use it, message me for how to.

# 4) Install Node-red
Node-red has a very active and helpful community with lots of good tutorials, a vist to their support forum is recommended: https://discourse.nodered.org/

Exit the Python Virtual environment by typing "deactivate" in the terminal where you ran the AI code test.  Install node-red, choose N for Pi specific modules:
```
bash <(curl -sL https://raw.githubusercontent.com/node-red/linux-installers/master/deb/update-nodejs-and-nodered)
```
As the script runs It asks you a few questions, these are my answers:
```
would you like to customise the settings now (y/N) ?  Y
=====================================
This tool will help you create a Node-RED settings file.

✔ Settings file · /home/ai/.node-red/settings.js

User Security
=============
✔ Do you want to setup user security? · No

Projects
========
The Projects feature allows you to version control your flow using a local git repository.

✔ Do you want to enable the Projects feature? · Yes
✔ What project workflow do you want to use? · manual - you must manually commit changes

Editor settings
===============
✔ Select a theme for the editor. To use any theme other than "default", you will need to install @node-red-contrib-themes/theme-collection in your Node-RED user directory. · default

✔ Select the text editor component to use in the Node-RED Editor · monaco (default)

Node settings
=============
✔ Allow Function nodes to load external modules? (functionExternalModules) · Yes

```
To complete the node-red setup, in the terminal window do:
```
cd $HOME
node-red-stop
cd ~/.node-red
npm i node-red-dashboard node-red-node-email node-red-contrib-image-output node-red-node-base64
npm install point-in-polygon
```

Now have to make a minor change to the node-red settings file (should still be in the .node-red hidden directory):
```
sudo nano settings.js
```
My "in alert region" node-red filter needs an extra JavaScript module that is not available in node-red by default.
So must edit .node-red/settings.js and add (around lin 510 on new install 30JUL2024), ^/ will let you jump to line 510
and edit it to look like this:
```
      functionGlobalContext: {
          insidePolygon:require('point-in-polygon'),
      }
```
Exit nano with: Ctrl-X Y
restart node red with (remember we stopped it previously):
```
node-red-start
```
And leave the script by typing Ctrl-C in the termainal after the node-red "start-up" messages, then make node-red start automatically with:
```
sudo systemctl enable nodered.service
```

If you didn't do this is step 0), do it now to avoid having to edit all the scripts used by node-red exec nodes in the sample controller flow:
```
sudo ln -s /home/YourUserName /home/ai
```

To install the basic controller, open web browser (Chromium is recommended) and point it at YourHostName:1880 (or localhost:1880 if not installing remotely) and follow through the "Welcome to Node-RED 4.0" steps to see what is new and different from prior versions. When you get the "projects" do "Create Project" and fill in the dialogs.  I chose NO to security and encryption since no external connections are accepted by my firewall, do what works for you.

Open the node-red "Hamburger" (three parallel bars) menu in the upper right corner of the editor and select "Import".  Press the "new flow" buton in the dialog box click the "select a file to import" button.  Navigate to the BasicAI2Controller.json. file.  Afterwards, click the Deploy button and you should be ready to go.  you'll have to set up the Email addresses in the "Setup Email" node and set up your smtp Email account credentials in the "Email Notification" node.  You will also have to edit the AI2/StartAI.sh file to use the correct options to start the AI2 Python code, hopefully the comments in the file help guide you.
#### If "headless" remove the password prompt from the sudo command.  This is required for the Reboot.sh and PowerOff.sh scripts to work when launched from node-red UI.
```
# This is sucidial if host machine accepts connections from outside your subnet!
sudo visudo
# add after the last line of the file (#includedir /etc/sudoers.d):
ai ALL=(ALL) NOPASSWD:ALL
```


# 5) Some useful options:
### It is best to put AI detections on a seperate drive or USB stick, but not necessary,
I had to mount the external device to create the /media/ai directory where it will mount,
then unmount it and do:
```
sudo mkdir /media/ai
sudo mkdir /media/ai/AI
sudo chown ai.ai /media/ai
sudo chown ai.ai /media/ai/AI
```
#### then edit /etc/fstab so it mounts on bootup:
```
sudo nano /etc/fstab
```
Add entry like this, changing /dev and mount point as desired (here AI is the drive's ext4 label name)
Note tgat /dev/sda1 is usually the name of a USB stick for Linux system on eEMC, SD card or NVMe drive, but not always
Hint, use the Linux "Disks" utility programe to get the device name of the partition to mount.
```
# My YouSeeToo-X1 system is booting from eEMC and using NVMe drive to store images:
/dev/nvme0n1p1 /media/ai/AI	ext4	defaults,nofail,noatime	0	3
# For a USB stick
/dev/sdb1	/media/ai/AI	ext4	defaults,nofail,noatime	0	3
```
Then make a symlink in /home/ai:
```
cd $HOME
ln -s /media/ai/AI detect
```
Reboot and verify the external drive is mounted.

### I also like to setup a samba server so you can view the detection images from other machines on your local subnet.
Configue samba file sharing.  Since this is to be an "appliance" this is the best way to look at AI detection files, edit samba config:
```
sudo nano /etc/samba/smb.conf
```
#### Add this is in the [global] section:
```
    mangled names = no
    follow symlinks = yes
    wide links = yes
    unix extensions = no
    # Ubuntu 20.04 seems to require these:
    server min protocol = NT1
    client min protocol = NT1
```
#### Make the homes section be like this:
```
[homes]
   comment = Home Directories
   browseable = yes
   read only = no
   writeable = yes
   create mask = 0775
   directory mask = 0775
```
#### set samba password, I use the ai login password for simplicity:
```
sudo smbpasswd -a ai
```

# 6) Open Source Alternatives.
My system was designed to be an addon to an existing DVR/NVR which seemed far easier than re-inventing this wheel, although it works fine with just IP or "net cams".  Most consumer priced DVR/NVR have really poor user interfaces, so my AI detection gives a much more efficient way to find regions of interest in the recordings using the timestamps built into the filenames than the typically lame timeline "scrubbing" of most consumer DVR/NVR.  I installed my first DVR in 2014 and in these 10 years I've gone back and actually looked at recorded video maybe half a dozen times since, but the 24/7 recordings can be "nice to have" if something really bad happens.

## Frigate
If you want more tradational NVR/DVR type features along with AI person detection I'd suggest looking at Frigate:

https://docs.frigate.video/

https://github.com/blakeblackshear/frigate

It runs as a "Docker" container and can integrate with the Home Assistant project:

https://www.home-assistant.io/

## Zoneminder
Zoneminder is another open source project to re-invent the DVR/NVR and add AI object detection features.
I ran a very old version of Zoneminder for many years starting circa 2005, well before AI object detection was available. It is a long running and stable project.

https://zoneminder.com/

https://github.com/ZoneMinder/zoneminder

## Motion, Motion Plus
Another long running project, I used it to turn a RaspberryPi Zero W with PiCam into a WiFi connected camera that rode on my friend's model RailRoad.  It has continued to evolve, adding object detecion with OpenCV HOG, Haar, and DNN modules.

https://motion-project.github.io/index.html

The most interesting new feature in Motion Plus is "Sound Frequency Processing" where it can listen for certain audio frequency ranges and send an alert when detected.

https://motion-project.github.io/motionplus_examples.html#sound_sample

## iSpy
Another "full featured" DVR/NVR replacement available for Windows, OSX and Linux.
I've run into it when searching for camera URLs they have a pretty large database of these.

https://www.ispyconnect.com/

https://github.com/ispysoftware/iSpy

https://www.ispyconnect.com/cameras

## Bluecherry
I've little experience with this one, it has a free limited version, and $$$ licence required for full features.
Probably its best feature is they've a repository for simple insttalation.

https://www.bluecherrydvr.com/

https://github.com/bluecherrydvr/unity


# 7) Some performance summaries on different hardware.
Note: "dropped" is number of detections that were dropped because writing to the output queue timed out, "results dropped" are frames without detection that were dropped writing to the output queue, the time out is much shorter for frames with no detection. Dropping an image without a person detected only has consequences for the live display.  The "detections failed zoom-in verification" are frames with a person that failed the Mobilenet detection when frame was cropped around the detected person and then failed the re-detect, this can greatly reduce the load on the Yolo thread.  These are most common with fixed objects in the frame that mis-detect as a person when the light is just "right", these tend to come in bursts of seconds to minutes long, usually as the sun is rising, setting, or approaching noon.

### Lenovo Ideapad3 i3-1115GS4 @3Ghz 4 cores, Mesa UHD graphics (TGL GT2), 8GB RAM, Ubuntu-Mate 20.04
This system cost me ~$160 as an "openbox close-out" at Microcenter when 12th gen systems were released.
Using four 4K rtsp streams at ~3 fps per camera for a test run, with node-red installed and running command:
```
(y8ovv) python AI2.py -d -nsz -y8ovv # -nsz doesn't save zoomed images, since the node-red does it.
[INFO] Program Exit signal received:  2024-08-29 15:26:01
*** AI processing approx. FPS: 12.83 ***
    [INFO] Run elapsed time: 69751.02
    [INFO] Frames processed by AI system: 894734
    [INFO] Person Detection by AI system: 1643
    [INFO] Main loop waited for resultsQ: 272851 times.
[INFO] Stopping OpenVINO yolo8 verification Thread ...
Yolo v8 frames Verified: 1643, Rejected: 365,  Waited: 68934 seconds.
    Verified dropped: 0, results dropped: 0, results.put() exceptions: 0
[INFO] Stopping openvino CPU AI  Thread ...
OpenVINO CPU MobilenetSSD AI thread ovCPU, waited: 174222 dropped: 162 of 894906 images.  AI: 13.07 inferences/sec
    ovCPU Persons Detected: 2008,  Frames with no person: 892898
    ovCPU 15682 detections failed zoom-in verification.
    ovCPU Detections dropped: 0, results dropped: 162, resultsQ.put() exceptions: 0

# As above but but with six 4K cameras and two 5Mpixel cameras:
[INFO] Program Exit signal received:  2024-09-01 16:31:13
*** AI processing approx. FPS: 16.18 ***
    [INFO] Run elapsed time: 11962.06
    [INFO] Frames processed by AI system: 193522
    [INFO] Person Detection by AI system: 397
    [INFO] Main loop waited for resultsQ: 10323 times.
[INFO] Stopping OpenVINO yolo8 verification Thread ...
Yolo v8 frames Verified: 397, Rejected: 47,  Waited: 11768 seconds.
    Verified dropped: 0, results dropped: 0, results.put() exceptions: 0 
OpenVINO CPU MobilenetSSD AI thread ovCPU, waited: 1278 dropped: 343 of 193875 images.  AI: 17.16 inferences/sec
    ovCPU Persons Detected: 444,  Frames with no person: 193431
    ovCPU 12130 detections failed zoom-in verification.
    ovCPU Detections dropped: 0, results dropped: 343, resultsQ.put() exceptions: 0
# At only ~16fps instead of the expected ~24fps from the eight cameras, it looks like eight cameras
# is too much for this system.  For this system I'd say that six 4K cameras is about the max.

# Repeat above test with USB3 TPU and the same eight cameras:
(y8ovv) python AI2.py -d -nsz -y8ovv -tpu
[INFO] Program Exit signal received:  2024-09-01 19:56:41
*** AI processing approx. FPS: 24.09 ***
    [INFO] Run elapsed time: 11532.35
    [INFO] Frames processed by AI system: 277863
    [INFO] Person Detection by AI system: 1199
    [INFO] Main loop waited for resultsQ: 49168 times.

Yolo v8 frames Verified: 1199, Rejected: 89,  Waited: 11119 seconds.
    Verified dropped: 0, results dropped: 0, results.put() exceptions: 0

TPU, waited: 45124 dropped: 553 out of 278426 images.  AI: 24.85 inferences/sec
   TPU Persons Detected: 1288,  Frames with no person: 277138
   TPU 8555 TPU detections failed zoom-in verification.
   TPU Detections dropped: 0, results dropped: 553, results.put() exceptions: 0
# Adding the $60 USB3 TPU (~$30 if you have M.2 slot available) has at least
# doubled the number of 4K cameras that can be supported on a somewhat weak processor.
```
### Intel i3-10100F @3.6GHz 4 cores, NVidia GTX950 graphics (768 cuda cores), 16 GB RAM, Mate 22.04
Using four 4K rtsp streams at ~3 fps per camera for a test run, without node-red installed, running command:
```
(y8cuda) python AI2.py -d -y8v
[INFO] Program Exit signal received:  2024-09-01 10:07:48
*** AI processing approx. FPS: 12.95 ***
    [INFO] Run elapsed time: 73886.00
    [INFO] Frames processed by AI system: 956514
    [INFO] Person Detection by AI system: 2923
    [INFO] Main loop waited for resultsQ: 344086 times.
[INFO] Stopping CUDA yolo8 verification Thread ...
Yolo v8 frames Verified: 2923, Rejected: 336,  Waited: 72746 seconds.
    Verified dropped: 0 results dropped: 0 results.put() exceptions: 0 
[INFO] Stopping openvino CPU AI  Thread ...
OpenVINO CPU MobilenetSSD AI thread ovCPU, waited: 209732 dropped: 145 of 956669 images.  AI: 13.46 inferences/sec
    ovCPU Persons Detected: 3259,  Frames with no person: 953410
    ovCPU 35381 detections failed zoom-in verification.
    ovCPU Detections dropped: 0, results dropped: 145, resultsQ.put() exceptions: 0

# As above but but with six 4K cameras and two 5Mpixel cameras:
[INFO] Program Exit signal received:  2024-09-01 13:03:20
*** AI processing approx. FPS: 24.05 ***
    [INFO] Run elapsed time: 8181.96
    [INFO] Frames processed by AI system: 196799
    [INFO] Person Detection by AI system: 951
    [INFO] Main loop waited for resultsQ: 6890 times.
Yolo v8 frames Verified: 961, Rejected: 80,  Waited: 7853 seconds.
    Verified dropped: 10 results dropped: 1 results.put() exceptions: 0
OpenVINO CPU MobilenetSSD AI thread ovCPU, waited: 1196 dropped: 405 of 197225 images.  AI: 25.97 inferences/sec
    ovCPU Persons Detected: 1041,  Frames with no person: 196184
    ovCPU 15834 detections failed zoom-in verification.
    ovCPU Detections dropped: 0, results dropped: 405, resultsQ.put() exceptions: 0
# This system easily handles 8 cameras at ~3fps per camera, although this could be
# near the practical limit since the yolo thread is dropping some verified frames (~1%).
# I have overlapping camera views so a person can be in the view of two cameras at the
# time, which increases the load on the yolo thread.
```
### IOT class, YouYeeTwoX1, Celeron N5105 @2GHz 4 cores Mesa UHD graphics (JSL), 8GB RAM, M.2 TPU Mate 22.04
Using four 4K rtsp streams at ~3fps per camera, node-red installed, no VENV, running command:
```
python3 AI2.py -tpu -nsz -d -y8ovv
[INFO] Program Exit signal received:  2024-09-01 20:13:24
*** AI processing approx. FPS: 13.86 ***
    [INFO] Run elapsed time: 10667.59
    [INFO] Frames processed by AI system: 147845
    [INFO] Person Detection by AI system: 769
    [INFO] Main loop waited for resultsQ: 62097 times.

Yolo v8 frames Verified: 769, Rejected: 20,  Waited: 10407 seconds.
    Verified dropped: 0, results dropped: 0, results.put() exceptions: 0

TPU, waited: 59899 dropped: 159 out of 148014 images.  AI: 14.65 inferences/sec
   TPU Persons Detected: 789,  Frames with no person: 147225
   TPU 8182 TPU detections failed zoom-in verification.
   TPU Detections dropped: 0, results dropped: 159, results.put() exceptions: 0
```
Easily supports 4 UHD (4K) cameras, but not very usable without Coral TPU. A test run with 9 UHD
cameras did 28.4 fps which is basically processing every frame for 9 cameras at ~3 fps per camera.
This board is ~$160, by time you add M.2 TPU, power supply and M.2 SSD
you are over $230, which is why I've lost interest in IOT class hardware
and think "last year's model bottom of the line" laptop close-outs or
"buisness class" refubished laptops are a better bet if cost matters
more than physical space, since basically has to be "headless" if cost matters.
Here is a photo of the YouYeeTwoX1, it is a bit larger than a RaspberryPi:
![YouyeetooX1](https://github.com/user-attachments/assets/54c51644-a9e0-4af6-bb9f-7d885c1ffd94)
Running headless, only the coaxial power connector and Ethernet cable would be plugged in, along with a powered speaker, not shown in the photo, if you want audio (usually you will).
### Old i5-4200U with iGPU not compatible with openvino using two TPUs and 9 UHD (4K) cameras.
```
# command:
(y8tpu) python AI2.Py -d -nsz -tpu -y8tpu
*** AI processing approx. FPS: 27.17 ***
    [INFO] Run elapsed time: 5684.75
    [INFO] Frames processed by AI system: 154463
    [INFO] Person Detection by AI system: 818
    [INFO] Main loop waited for resultsQ: 8148 times.

Yolo v8 frames Verified: 818, Rejected: 83,  Waited: 5336 seconds.
    Verified dropped: 0 results dropped: 0 results.put() exceptions: 0

TPU, waited: 5745 dropped: 564 out of 155037 images.  AI: 27.95 inferences/sec
   TPU Persons Detected: 901,  Frames with no person: 154136
   TPU 4762 TPU detections failed zoom-in verification.
   TPU Detections dropped: 0, results dropped: 564, results.put() exceptions: 0
```
Since each camera is ~3fps this is basically processing every frame.  Running with only a single TPU and using CPU for the SSD AI, basically got 9.5fps, which is usable for 4 cameras, but not optimum. You can start with a single tpu and three or four cameras, and then add another TPU (USB ~$60, M.2 ~$30) when adding more cameras.
