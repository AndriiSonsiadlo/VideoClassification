from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained serialized model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to  label binarizer")
ap.add_argument("-i", "--input", required=True,
	help="path to our input video")
ap.add_argument("-o", "--output", required=True,
	help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=128,
	help="size of queue for averaging")
args = vars(ap.parse_args())


print("[INFO] loading model and class names...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())
# mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args["size"])


vs = cv2.VideoCapture(args["input"])
video_frames_count = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
writer = None
(W, H) = (None, None)
for frame_counter in range(video_frames_count):

	_, frame = vs.read()
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	resized_frame = cv2.resize(frame, (64, 64))


	normalized_frame = resized_frame / 255


	preds = model.predict(np.expand_dims(normalized_frame, axis=0))[0]
	output = frame.copy()
########################################################################
# writer = None
# (W, H) = (None, None)
# # loop over frames from the video file stream
# while True:
# 	# read the next frame from the file
# 	(grabbed, frame) = vs.read()
# 	# if the frame was not grabbed, then we have reached the end
# 	# of the stream
# 	if not grabbed:
# 		break
# 	# if the frame dimensions are empty, grab them
# 	if W is None or H is None:
# 		(H, W) = frame.shape[:2]
#
#
# 	# clone the output frame, then convert it from BGR to RGB
# 	# ordering, resize the frame to a fixed 224x224, and then
# 	# perform mean subtraction
# 	output = frame.copy()
# 	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# 	frame = cv2.resize(frame, (64, 64)).astype("float32")
# 	frame -= mean
	###########################################################


	# preds = model.predict(np.expand_dims(frame, axis=0))[0]

	Q.append(preds)
	results = np.array(Q).mean(axis=0)
	print(results)
	i = np.argmax(results)
	label = lb[i]

	result_score = round(results[i], 2)

	text = "activity: {} {}".format(label, result_score)
	cv2.putText(output, text, (0,20), cv2.FONT_HERSHEY_SIMPLEX,
		0.7, (0, 0, 0), 2)
	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)
	writer.write(output)
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

print("[INFO] cleaning up...")
writer.release()
vs.release()