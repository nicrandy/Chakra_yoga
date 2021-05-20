import mediapipe as mp
import cv2
import time
from sklearn import svm
import pickle
import keyboard as kb
import os

mp_pose = mp.solutions.pose

current_pose = 3
avg_pose_percent_array = [0,0,0,0,0]

# Prepare DrawingSpec for drawing the face landmarks later.
mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=3)

#load saved SVC model
filename = 'pose_classifier.pkl'
loaded_model = pickle.load(open(filename,'rb'))

#link to list with txt of pose names
pose_list = "pose_list.txt"
def pose_name(classifier):
	with open(pose_list) as f:
		poses = f.readlines()
	return poses[classifier]

def example_pose_image(pose):
	#example poses image folder
	example_images = 'example_poses'
	target_pose = ''
	for root, dirs, files in os.walk(example_images):
		find_pose = str(pose) + '.jpg'
		for file in files:
			if file == find_pose:
				target_pose = example_images + "/" + find_pose
				print(target_pose)
	return target_pose

#get probability percent based on target pose and output of SVC
def get_pose_from_landmarks(landmarks,pose):
	print(landmarks_to_save)
	# landmarks_to_save = np.asarray(landmarks_to_save)
	prediction = loaded_model.predict_proba([landmarks_to_save])
	print(prediction)
	pose_probability = prediction[0][pose]
	pose_probability = int(100*pose_probability)
	return pose_probability

cap = cv2.VideoCapture(0)
#save time start and stop processing
prev_frame_time = 0
new_frame_time = 0
while(True):
	with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.1) as pose:
		# Convert the BGR image to RGB and process it with MediaPipe Pose.
		ret, image = cap.read()
		new_frame_time = time.time()
		results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
		pose_landmarks = results.pose_landmarks

		#adjust output image size
		output_increase = 1.5
		image_hight, image_width, _ = image.shape
		enlarge_hight = int(output_increase*image_hight)
		enlarge_width = int(output_increase*image_width)
		large_image = (enlarge_width,enlarge_hight)

		#paste example image on input image
		target_pose_image_location = example_pose_image(current_pose)
		img = cv2.imread(target_pose_image_location)		
		#adjust example size
		example_width = 150
		example_height = 150
		img = cv2.resize(img,(example_width,example_height),interpolation = cv2.INTER_AREA)
		print(img.shape)
		print(image.shape)
		x_offset = image.shape[1] - example_width
		y_offset = image.shape[0]-example_height
		x_end = image.shape[1]
		y_end = image.shape[0]
		image[y_offset:y_end,x_offset:x_end] = img
		
		#track desired body part and only use needed landmarks
		body_part = 0
		
		#track and display fps if desired
		fps = 1/(new_frame_time-prev_frame_time)
		prev_frame_time = new_frame_time
		fps = str(int(fps))
		desired_pose = current_pose
		if results.pose_landmarks != None:
			landmarks_to_save = []
			pose_landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in pose_landmarks.landmark]
			print("landmark locations : ", pose_landmarks)
			annotated_image = image.copy()
			for poses in results.pose_landmarks.landmark:
				if (body_part > 10 and body_part < 17) or (body_part > 22):
					landmarks_to_save.append(pose_landmarks[body_part][0])
					landmarks_to_save.append(pose_landmarks[body_part][1])
					landmarks_to_save.append(pose_landmarks[body_part][2])
					xloc = int(pose_landmarks[body_part][0] * image_width)
					yloc = int(pose_landmarks[body_part][1] * image_hight)
					font_size = -(pose_landmarks[body_part][2] *3)
					# cv2.putText(annotated_image, landmark_names[body_part], (xloc, yloc), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 2, cv2.LINE_AA)
				body_part += 1
			#get average of recent pose percentages to smooth out fluctuations
			
			pose_percent = get_pose_from_landmarks(landmarks_to_save, desired_pose)
			avg_pose_percent_array.pop(0)
			avg_pose_percent_array.append(pose_percent)
			avg_percent = sum(avg_pose_percent_array) / len(avg_pose_percent_array)
			print("avg_posearray:", avg_pose_percent_array)
			print("avg percent:", avg_percent)
			avg_percent = str(avg_percent)

			pose_class = pose_name(desired_pose)
			pose_class = pose_class.rstrip(pose_class[-1])
			cv2.rectangle(annotated_image,(0,0), (image_width,50),(0,0,0),-1)
			cv2.putText(annotated_image, "Pose: "+str(pose_class)+" " + avg_percent+"%", (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA,bottomLeftOrigin = False)
			cv2.putText(annotated_image, "Change pose <- or -> key", (50, image_hight-20), cv2.FONT_HERSHEY_SIMPLEX, .75, (100, 150, 255), 2, cv2.LINE_AA,bottomLeftOrigin = False)

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
			cv2.imshow('Pose',image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	
	#change to next or previous pose
	if (kb.is_pressed("right")):
		print("changed pose")
		if current_pose < 81:
			current_pose += 1
		else:
			current_pose = 0
	if (kb.is_pressed("left")):
		print("changed pose")
		if current_pose > 0:
			current_pose -= 1
		else:
			current_pose = 81

# cv2.release()
cv2.destroyAllWindows()
