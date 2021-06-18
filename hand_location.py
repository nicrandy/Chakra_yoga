import mediapipe as mp
import cv2
import time
from sklearn import svm
import pickle
import keyboard as kb
import os
from random import sample
import array

# Prepare DrawingSpec for drawing the face landmarks later.
mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=3)


source = 0
def test_cap():
	loop = False
	global source
	while loop == False:
		cap = cv2.VideoCapture(source)
		if cap is None or not cap.isOpened():
			source += 1
		else:
			break
test_cap()


cap = cv2.VideoCapture(source)
#save time start and stop processing
prev_frame_time = 0
new_frame_time = 0

#location to save video to
path = "C:/Users/User/Videos/timelapse/"
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc=cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter(path+'z_test.avi',fourcc, 7, (frame_width,frame_height))



#start loop for BlazePose network
mp_pose = mp.solutions.pose
while(True):
	with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.1, model_complexity=2) as pose:

		# Convert the BGR image to RGB and process it with MediaPipe Pose.
		ret, image = cap.read()


		image = cv2.flip(image,1)
		# image = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
		new_frame_time = time.time()#to calculate fps
		results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
		pose_landmarks = results.pose_landmarks

		#adjust output image size
		output_increase = 2
		image_hight, image_width, _ = image.shape
		enlarge_hight = int(output_increase*image_hight)
		enlarge_width = int(output_increase*image_width)
		large_image = (enlarge_width,enlarge_hight)

		#track desired body part and only use needed landmarks
		body_part = 0
		
		#track and display fps if desired
		fps = 1/(new_frame_time-prev_frame_time)
		prev_frame_time = new_frame_time
		fps = str(int(fps))

		if results.pose_landmarks != None:
			landmarks_to_save = []
			pose_landmarks = [[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in pose_landmarks.landmark]
			print("landmark locations : ", pose_landmarks)
			annotated_image = image.copy()
			left_hand_x = 0
			left_hand_y = 0
			left_hand_z = 0
			right_hand_x = 1
			right_hand_y = 1
			right_hand_z = 0
			for poses in results.pose_landmarks.landmark:
				if (body_part > 10 and body_part < 17) or (body_part > 22):
					landmarks_to_save.append(pose_landmarks[body_part][0])
					landmarks_to_save.append(pose_landmarks[body_part][1])
					landmarks_to_save.append(pose_landmarks[body_part][2])
					xloc = int(pose_landmarks[body_part][0] * image_width)
					yloc = int(pose_landmarks[body_part][1] * image_hight)
					font_size = -(pose_landmarks[body_part][2] *3)
					# cv2.putText(annotated_image, landmark_names[body_part], (xloc, yloc), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 2, cv2.LINE_AA)
				if body_part == 22:
					right_hand_x = pose_landmarks[21][0]
					right_hand_y = pose_landmarks[21][1]
					right_hand_z = pose_landmarks[21][2]
					right_hand_vis = pose_landmarks[21][3]
				if body_part == 21:
					left_hand_x = pose_landmarks[22][0]
					left_hand_y = pose_landmarks[22][1]
					left_hand_z = pose_landmarks[22][2]
					left_hand_vis = pose_landmarks[22][3]
				body_part += 1

			#display info on hand locations
			print("left hand x", left_hand_x, " y ", left_hand_y)
			lh_x = left_hand_x * image_width
			lh_y = left_hand_y * image_hight
			cv2.putText(annotated_image, "Left Hand", (int(lh_x),int(lh_y)), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 0, 0), 2, cv2.LINE_AA,bottomLeftOrigin = False)
			cv2.putText(annotated_image, "LH z: " +str(left_hand_z), (20,30), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 0), 2, cv2.LINE_AA,bottomLeftOrigin = False)

			rh_x = right_hand_x * image_width
			rh_y = right_hand_y * image_hight
			cv2.putText(annotated_image, "Right Hand", (int(rh_x),int(rh_y)), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 0, 0), 2, cv2.LINE_AA,bottomLeftOrigin = False)
			cv2.putText(annotated_image, "RH z: " +str(right_hand_z), (20,60), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 0), 2, cv2.LINE_AA,bottomLeftOrigin = False)
			out.write(annotated_image)

			# Draw pose landmarks.
			mp_drawing.draw_landmarks(
				image=annotated_image,
				landmark_list=results.pose_landmarks,
				connections=mp_pose.POSE_CONNECTIONS,
				landmark_drawing_spec=drawing_spec,
				connection_drawing_spec=drawing_spec)
		
			#display fps if desired
			# cv2.putText(annotated_image, "FPS:"+fps, (7, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)
			annotated_image = cv2.resize(annotated_image,large_image,interpolation = cv2.INTER_AREA)

			cv2.imshow('Pose',annotated_image)
		else:
			# cv2.putText(image, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
			image = cv2.resize(image,large_image,interpolation = cv2.INTER_AREA)
			cv2.putText(image, "Source: " + str(source), (300,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA,bottomLeftOrigin = False)
			cv2.imshow('Pose',image)
   
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	
# cv2.release()
cv2.destroyAllWindows()
out.release()