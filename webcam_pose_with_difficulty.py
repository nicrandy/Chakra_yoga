import mediapipe as mp
import cv2
import time
from sklearn import svm
import pickle
import keyboard as kb
import os
from random import sample
import array

#global variables
current_pose = 0 #track current pose number
avg_pose_percent_array = [0,0,0,0,0] #track percentages of poses

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
	return target_pose

#difficulty is from 0-4 for beginner to impossible
def pose_difficulty_selecter(difficulty,poses_to_select):
	#link to list with pose difficulties
	pose_difficulty = "pose_difficulty_list.txt"
	pose_image_links = []
	pose_numbers = []
	with open(pose_difficulty) as f:
		pose_diff_list=[]
		for line in f:
			if line[0] != '#':
				pose_diff_list.append(line.split())
		i = 0
		for poses in pose_diff_list:
			pose_diff_list[i] = sample(poses,poses_to_select)
			i += 1
		for pose in pose_diff_list[difficulty]:
			pose_image_links.append(example_pose_image(pose))
			pose_numbers.append(pose)
	print("Pose image links ", pose_image_links)
	return pose_image_links,pose_numbers

print(pose_difficulty_selecter(0,7))

#get probability percent based on target pose and output of SVC
def get_pose_from_landmarks(landmarks,pose):
	print(landmarks_to_save)
	# landmarks_to_save = np.asarray(landmarks_to_save)
	prediction = loaded_model.predict_proba([landmarks_to_save])
	print(prediction)
	pose_probability = prediction[0][pose]
	pose_probability = int(100*pose_probability)
	return pose_probability

#adjust the current pose tracker up or down
def change_pose(up_down):
	global current_pose
	if up_down == 1:
		print("------------changed pose", current_pose)
		if current_pose < (len(current_pose_list) - 1):
			current_pose += 1
		else:
			current_pose = 0
	if up_down == -1:
		print("--------------changed pose",current_pose)
		if current_pose > 0:
			current_pose -= 1
		else:
			current_pose = (len(current_pose_list) - 1)

#calculate pose scores
current_pose_score = []
def calculate_score(current_pose_percentage):
	global current_pose_score
	if timer_started == True:
		current_pose_score.append(current_pose_percentage)
		length = len(current_pose_score)
		total = sum(current_pose_score)
		score = total / length
		grade = 'C'
		if score < 10:
			grade = 'D'
		if score >= 10 and score < 30:
			grade = 'C'
		if score >= 30 and score < 50:
			grade = 'B'
		if score >= 50 and score < 70:
			grade = 'A'
		if score > 85:
			grade = 'S'
		return grade

#for countdown timer
time_per_pose = 15 #adjust number of seconds per pose
timer_started = False
start_pose_time = time.time()
def countdown_timer(avg_percent):
	global start_pose_time
	global timer_started
	if avg_percent > 10 and timer_started == False:
		start_pose_time = time.time()
		timer_started = True
	time_remaining = time_per_pose - (time.time() - start_pose_time)
	print("time_remaining:", time_remaining)
	if time_remaining <= 0 and timer_started == True:
		avg_pose_percent_array = [0,0,0,0,0]
		current_pose_score = []
		change_pose(1)
		timer_started = False
		time.sleep(2)
	return time_remaining


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
current_pose_list,current_pose_numbers = pose_difficulty_selecter(0,10)

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

		#paste example image on input image
		print("example pose : ",current_pose_list[current_pose])
		img = cv2.imread(current_pose_list[current_pose])	
		#adjust example size
		ratio = img.shape[0]/img.shape[1]
		print("ratio = ",ratio)
		example_width = 200
		example_height = int((200 * ratio))
		img = cv2.resize(img,(example_width,example_height),interpolation = cv2.INTER_AREA)
		print(img.shape)
		print(image.shape)
		x_offset = image.shape[1] - example_width
		y_offset = image.shape[0]-example_height
		x_end = image.shape[1]
		y_end = image.shape[0]
		#put black rectange on right side
		cv2.rectangle(image,((image.shape[1]-180),0), (image_width,image_hight),(0,0,0),-1)
		image[y_offset:y_end,x_offset:x_end] = img
		
		#track desired body part and only use needed landmarks
		body_part = 0
		
		#track and display fps if desired
		fps = 1/(new_frame_time-prev_frame_time)
		prev_frame_time = new_frame_time
		fps = str(int(fps))

		desired_pose = int(current_pose_numbers[current_pose])
		print("current pose number : ", desired_pose)
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
			avg_percent = 0
			if timer_started == True or pose_percent > .01:
				avg_percent = sum(avg_pose_percent_array) / len(avg_pose_percent_array)
			else:
				avg_percent = 0
			print("avg_posearray:", avg_pose_percent_array)
			print("avg percent:", avg_percent)
			avg_percent_string = str(avg_percent)
			pose_scoring = calculate_score(int(avg_percent))
			print("pose scoring",pose_scoring)

			#get name of current pose
			pose_class = pose_name(int(current_pose_numbers[current_pose]))
			pose_class = pose_class.rstrip(pose_class[-1])
			

			#control timer
			time_remaining = countdown_timer(avg_percent)
			if timer_started == True:
				cv2.putText(annotated_image, "Time: " + str(int(time_remaining)), ((x_offset +20), (y_offset-95)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA,bottomLeftOrigin = False)
			else:
				cv2.putText(annotated_image, "Ready", ((x_offset +20), (y_offset-95)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA,bottomLeftOrigin = False)

			cv2.putText(annotated_image, pose_scoring, ((x_offset +20), (y_offset-35)), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50), 1, cv2.LINE_AA,bottomLeftOrigin = False)	
			cv2.putText(annotated_image, str(pose_class), ((x_offset +20), (y_offset-5)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA,bottomLeftOrigin = False)
			cv2.putText(annotated_image, avg_percent_string+"%", ((x_offset +20), (y_offset-65)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA,bottomLeftOrigin = False)
			#display camera source for testing 
			cv2.putText(annotated_image, "Source: " + str(source), ((x_offset +20), (y_offset-200)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA,bottomLeftOrigin = False)
			
			#cv2.putText(annotated_image, "Change pose <- or -> key", (50, image_hight-20), cv2.FONT_HERSHEY_SIMPLEX, .75, (100, 150, 255), 2, cv2.LINE_AA,bottomLeftOrigin = False)

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
			cv2.putText(image, "Source: " + str(source), ((x_offset +20), (y_offset-200)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA,bottomLeftOrigin = False)

			cv2.imshow('Pose',image)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	
	#change to next or previous pose
	if (kb.is_pressed("right")):
		change_pose(1)
	if (kb.is_pressed("left")):
		change_pose(-1)

# cv2.release()
cv2.destroyAllWindows()
