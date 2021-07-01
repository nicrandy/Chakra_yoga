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

#callibrate each pose. [pose number, average, remove lowest 20% then get average then * .2]
pose_callibrations = [[0, 12.63, 6.11], [1, 14.53, 6.44], [2, 63.1, 16.15], [3, 41.65, 14.75], [4, 37.21, 12.87], [5, 59.04, 16.88], 
						[6, 7.07, 4.02], [7, 54.59, 16.19], [8, 48.21, 14.51], [9, 29.25, 9.6], [10, 44.97, 14.56], [11, 28.53, 13.21], 
						[12, 35.89, 10.71], [13, 23.11, 9.39], [14, 37.74, 12.01], [15, 26.93, 9.38], [16, 24.76, 9.59], [17, 50.48, 14.41], 
						[18, 38.0, 12.19], [19, 17.19, 6.94], [20, 23.64, 8.16], [21, 32.98, 10.69], [22, 51.73, 15.75], [23, 26.41, 8.27], 
						[24, 12.07, 7.26], [25, 18.62, 6.94], [26, 56.31, 14.99], [27, 26.09, 11.28], [28, 36.57, 13.04], [29, 28.97, 12.2], 
						[30, 36.74, 11.78], [31, 48.68, 15.13], [32, 31.01, 10.13], [33, 34.47, 12.09], [34, 7.36, 2.91], [35, 28.1, 11.1], 
						[36, 34.58, 12.27], [37, 38.97, 13.2], [38, 46.86, 14.45], [39, 38.62, 11.95], [40, 27.0, 9.57], [41, 3.04, 1.53], 
						[42, 44.36, 14.09], [43, 3.34, 1.64], [44, 50.94, 15.01], [45, 20.29, 10.08], [46, 14.83, 6.11], [47, 14.91, 6.39], 
						[48, 34.02, 13.87], [49, 15.9, 7.69], [50, 29.25, 10.35], [51, 26.16, 10.94], [52, 24.88, 9.8], [53, 9.95, 4.75], 
						[54, 48.77, 16.33], [55, 21.29, 7.68], [56, 56.69, 16.57], [57, 51.2, 15.7], [58, 25.36, 10.12], [59, 15.56, 5.54], 
						[60, 47.75, 14.22], [61, 15.33, 6.43], [62, 10.25, 5.79], [63, 35.89, 11.01], [64, 40.87, 13.11], [65, 33.13, 11.26], 
						[66, 25.33, 9.26], [67, 4.13, 1.74], [68, 55.61, 15.74], [69, 31.14, 10.4], [70, 9.68, 4.8], [71, 54.95, 16.32], 
						[72, 17.88, 6.98], [73, 46.57, 13.71], [74, 36.82, 11.81], [75, 57.09, 16.23], [76, 32.81, 10.88], [77, 11.8, 5.56], 
						[78, 39.38, 13.58], [79, 19.7, 7.48], [80, 38.37, 12.8], [81, 2.16, 0.64]]




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
	print("Pose numbers : ",pose_numbers)
	#Below 3 lines can be used to create a specific routine of poses
	# pose_numbers = ['7','8','9','10','16','22','44','57']
	# pose_image_links = ['example_poses/7.jpg', 'example_poses/8.jpg', 'example_poses/9.jpg', 'example_poses/10.jpg', 'example_poses/16.jpg',
	# 				    'example_poses/22.jpg', 'example_poses/44.jpg', 'example_poses/57.jpg']
	# print("Pose image links ", pose_image_links)

	return pose_image_links,pose_numbers

# print(pose_difficulty_selecter(0,5))

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

timer_started = False
start_pose_time = time.time()
#time for each pose
time_per_pose = 15
def countdown_timer(pose_class,avg_percent):
	global start_pose_time
	global timer_started
	if avg_percent > pose_callibrations[pose_class][2] and timer_started == False:
		start_pose_time = time.time()
		timer_started = True
	time_remaining = time_per_pose - (time.time() - start_pose_time)
	print("time_remaining:", time_remaining)
	if time_remaining < 0 and timer_started == True:
		change_pose(1)
		timer_started = False
		time.sleep(1)

	return time_remaining

cap = cv2.VideoCapture(0)
#location to save video to
path = "C:/Users/User/Videos/"
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc=cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter(path+'yoga_test.avi',fourcc, 20, (frame_width,frame_height))

#save time start and stop processing
prev_frame_time = 0
new_frame_time = 0
current_pose_list,current_pose_numbers = pose_difficulty_selecter(0,7)

#start loop for BlazePose network
mp_pose = mp.solutions.pose
while(True):
	with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.1, model_complexity=1) as pose:

		# Convert the BGR image to RGB and process it with MediaPipe Pose.
		ret, image = cap.read()
		# image = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
		new_frame_time = time.time()#to calculate fps
		results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
		pose_landmarks = results.pose_landmarks

		#adjust output image size
		output_increase = 1.5
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
			avg_percent = sum(avg_pose_percent_array) / len(avg_pose_percent_array)
			print("avg_posearray:", avg_pose_percent_array)
			print("avg percent:", avg_percent)
			avg_percent_string = str(avg_percent)

			#get graded score
			pose_scoring = calculate_score(int(avg_percent))
			print("pose scoring",pose_scoring)

			pose_class = pose_name(int(current_pose_numbers[current_pose]))
			pose_class = pose_class.rstrip(pose_class[-1])
			

			#control timer
			time_remaining = countdown_timer(desired_pose,avg_percent)
			if timer_started == True:
				cv2.putText(annotated_image, "Time: " + str(int(time_remaining)), ((x_offset +20), (y_offset-30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA,bottomLeftOrigin = False)
			else:
				cv2.putText(annotated_image, "Ready", ((x_offset +20), (y_offset-30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA,bottomLeftOrigin = False)
			# cv2.putText(annotated_image, pose_scoring, ((x_offset +20), (y_offset-95)), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50), 1, cv2.LINE_AA,bottomLeftOrigin = False)	
			cv2.putText(annotated_image, "#" + str(desired_pose) + ":" + pose_class, ((x_offset +20), (y_offset-5)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA,bottomLeftOrigin = False)
			
			cv2.putText(annotated_image, "Current: " + avg_percent_string, ((x_offset +20), (y_offset-145)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA,bottomLeftOrigin = False)
			cv2.putText(annotated_image, "Begin: " + str(pose_callibrations[desired_pose][2]), ((x_offset +20), (y_offset-125)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA,bottomLeftOrigin = False)
			cv2.putText(annotated_image, "Goal: " + str(pose_callibrations[desired_pose][1]), ((x_offset +20), (y_offset-105)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA,bottomLeftOrigin = False)



			#cv2.putText(annotated_image, "Change pose <- or -> key", (50, image_hight-20), cv2.FONT_HERSHEY_SIMPLEX, .75, (100, 150, 255), 2, cv2.LINE_AA,bottomLeftOrigin = False)

			# Draw pose landmarks.
			mp_drawing.draw_landmarks(
				image=annotated_image,
				landmark_list=results.pose_landmarks,
				connections=mp_pose.POSE_CONNECTIONS,
				landmark_drawing_spec=drawing_spec,
				connection_drawing_spec=drawing_spec)
			#save image to video file
			out.write(annotated_image)
		
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
		change_pose(1)
	if (kb.is_pressed("left")):
		change_pose(-1)

# cv2.release()
cv2.destroyAllWindows()
out.release()
