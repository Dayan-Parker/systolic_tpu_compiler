import argparse
import cv2
import depthai as dai
import numpy as np
import time

# --- CONSTANTS ---
NEURAL_FPS = 8 
STEREO_FPS = 30
IP_ADDRESS = None

parser = argparse.ArgumentParser()
parser.add_argument("--depthSource", type=str, default="stereo", choices=["stereo", "neural"])
args = parser.parse_args()

# --- 1. BUILD PIPELINE ---
pipeline = dai.Pipeline()

# Define Sources
camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)

# Define Outputs (XLinkOut for Host)
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutNN = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")
xoutDepth.setStreamName("depth")

# --- 2. CONFIGURE NODES ---
# Cameras
camRgb.setPreviewSize(640, 400) # Match model input size
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(STEREO_FPS)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoLeft.setFps(STEREO_FPS)

monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
monoRight.setFps(STEREO_FPS)

# Stereo Depth
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A) # Align depth to RGB
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())
stereo.setSubpixel(True) # Better accuracy for distant objects

# Network (YOLOv8n / Custom FSOCO)
spatialDetectionNetwork.setBlobPath("perception\perception\yolo\runs\detect\fsoco_training\run_high_power\weights\best.pt")
spatialDetectionNetwork.setConfidenceThreshold(0.5)  # Very important confidence threshold
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100) # 1 meter
spatialDetectionNetwork.setDepthUpperThreshold(12000) # 12 meters

# YOLO Specifics (Update based on your model)
spatialDetectionNetwork.setNumClasses(4) 
spatialDetectionNetwork.setCoordinateSize(4)
spatialDetectionNetwork.setAnchors([]) 
spatialDetectionNetwork.setAnchorMasks({})
spatialDetectionNetwork.setIouThreshold(0.5)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

camRgb.preview.link(spatialDetectionNetwork.input)
stereo.depth.link(spatialDetectionNetwork.inputDepth)

spatialDetectionNetwork.out.link(xoutNN.input)
camRgb.preview.link(xoutRgb.input)
stereo.depth.link(xoutDepth.input)

# --- 3. DEVICE CONNECTION & OPTIMIZATION ---

# Helper to find device
device_info = None
if IP_ADDRESS:
    device_info = dai.DeviceInfo(IP_ADDRESS)

with dai.Device(pipeline, device_info) as device:
    
    # !!! CRITICAL FOR OAK-D PRO !!!
    # Turn on IR Laser Dot Projector (0 to 1200 mA)
    # This adds texture to asphalt so stereo depth works
    device.setIrLaserDotProjectorBrightness(1000) 
    
    # Optional: Turn on Floodlight for night testing
    # device.setIrFloodLightBrightness(0)

    print(f"Connected to: {device.getMxId()}")
    print("IR Projector: ON (1000mA)")

    # Output Queues
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        # Get data
        inRgb = qRgb.tryGet()
        inDet = qDet.tryGet()
        inDepth = qDepth.tryGet()

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            
            # If we have detections, draw them
            if inDet is not None:
                detections = inDet.detections
                for detection in detections:
                    # ROI Mapping
                    x1 = int(detection.xmin * 640)
                    y1 = int(detection.ymin * 400)
                    x2 = int(detection.xmax * 640)
                    y2 = int(detection.ymax * 400)

                    # Draw Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Draw Coordinates (X, Y, Z in mm)
                    cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)}mm", 
                                (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

            cv2.imshow("rgb", frame)

        # Optional: Show depth map
        if inDepth is not None:
            depthFrame = inDepth.getFrame()
            depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
            cv2.imshow("depth", depthFrameColor)

        if cv2.waitKey(1) == ord('q'):
            break