# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import glob
from sort import *
import utills
import plot
tracker = Sort()
memory = {}
counter = 0

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

mouse_pts = []

def get_mouse_points(event, x, y, flags, param):   
    global mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_pts) < 4:
            cv2.circle(image, (x, y), 5, (0, 0, 255), 10)
        else:
            cv2.circle(image, (x, y), 5, (255, 0, 0), 10)
            
        if len(mouse_pts) >= 1 and len(mouse_pts) <= 3:
            cv2.line(image, (x, y), (mouse_pts[len(mouse_pts)-1][0], mouse_pts[len(mouse_pts)-1][1]), (70, 70, 70), 2)
            if len(mouse_pts) == 3:
                cv2.line(image, (x, y), (mouse_pts[0][0], mouse_pts[0][1]), (70, 70, 70), 2)
        
        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))

# load the COCO class labels our YOLO model was trained on
#labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
#LABELS = open(labelsPath).read().strip().split("\n")
LABELS = open("yolo-coco/coco.names").read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("yolo-coco/yolov3.cfg", "yolo-coco/yolov3.weights")
#net = cv2.dnn.readNetFromDarknet("yolo-coco/yolov3.cfg", "yolo-coco/yolov3.weights")
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
count = 0
vs = cv2.VideoCapture("input/sample.mp4")
writer = None
(W, H) = (None, None)

frameIndex = 0

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

# set mouse callback 

cv2.namedWindow("image")
cv2.setMouseCallback("image", get_mouse_points)
np.random.seed(4)

# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break
	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	scale_w, scale_h = utills.get_scale(W, H)
	# first frame will be used to draw ROI and horizontal and vertical 180 cm distance(unit length in both directions)
	if count == 0:
		while True:
			image = frame
			cv2.imshow("image", image)
			cv2.waitKey(1)
			if len(mouse_pts) == 8:
				cv2.destroyWindow("image")
				break
               
		points = mouse_pts 
	
	# Using first 4 points or coordinates for perspective transformation. The region marked by these 4 points are 
    # considered ROI. This polygon shaped ROI is then warped into a rectangle which becomes the bird eye view. 
    # This bird eye view then has the property property that points are distributed uniformally horizontally and 
    # vertically(scale for horizontal and vertical direction will be different). So for bird eye view points are 
    # equally distributed, which was not case for normal view.
	src = np.float32(np.array(points[:4]))
	dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
	prespective_transform = cv2.getPerspectiveTransform(src, dst)

    # using next 3 points for horizontal and vertical unit length(in this case 180 cm)
	pts = np.float32(np.array([points[4:7]]))
	warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]
        
    # since bird eye view has property that all points are equidistant in horizontal and vertical direction.
    # distance_w and distance_h will give us 180 cm distance in both horizontal and vertical directions
    # (how many pixels will be there in 180cm length in horizontal and vertical direction of birds eye view),
    # which we can use to calculate distance between two humans in transformed view or bird eye view
	distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
	distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
	#pnts = np.array(points[:4], np.int32)
	#cv2.polylines(frame, [pnts], True, (70, 70, 70), thickness=2)

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if classID==0:
			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
				if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
	
	dets = []
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			dets.append([x, y, x+w, y+h, confidences[i]])

	np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
	dets1 = np.asarray(dets)
	tracks = tracker.update(dets1)

	boxes = []
	indexIDs = []
	c = []
	previous = memory.copy()
	memory = {}

	for track in tracks:
		boxes.append([int(track[0]), int(track[1]), int(track[2]), int(track[3])])
		indexIDs.append(int(track[4]))
		memory[indexIDs[-1]] = boxes[-1]
	if len(boxes) > 0:
		i = int(-1)
		# Here we will be using bottom center point of bounding box for all boxes and will transform all those
        # bottom center points to bird eye view
		person_points = utills.get_transformed_points(boxes, prespective_transform)
        
        # Here we will calculate distance between transformed points(humans)
		distances_mat, bxs_mat = utills.get_distances(boxes, person_points, distance_w, distance_h)
		#risk_count = utills.classify(distances_mat)

		frame1 = np.copy(frame)
        
        # Draw bird eye view and frame with bouding boxes around humans according to risk factor    
		bird_image = plot.bird_eye_view(frame, distances_mat, person_points, scale_w, scale_h)
		img, color = plot.social_distancing_view(frame1, bxs_mat, boxes)
		for box in boxes:
    		# extract the bounding box coordinates
			(x, y) = (int(box[0]), int(box[1]))
			(w, h) = (int(box[2]), int(box[3]))
			text = "{}".format(indexIDs[i])
			cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
			i+=1		
			
	# saves image file
	cv2.imwrite("output/frame-{}.png".format(frameIndex), img)

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter("output/output.avi", fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)
		bird_movie = cv2.VideoWriter("output/bird_eye_view.avi", fourcc, 30, (int(width * scale_w), int(height * scale_h)))
		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

	# write the output frame to disk
	writer.write(img)

	# increase frame index
	frameIndex += 1

	if frameIndex >= 200:
		print("[INFO] cleaning up...")
		writer.release()
		vs.release()
		exit()

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()