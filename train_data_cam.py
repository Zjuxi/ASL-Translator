# this is a way to use the nn for real world purposes

import cv2
import mediapipe as mp
import os

# get an address that doesn't exist
latestFile = [] # d, h, m, x
for alph in range(26):
    letter = chr(alph + ord('a'))

    num = 0
    img_name = letter + str(num)
    address = "train2/" + img_name + ".jpg"

    while os.path.isfile(address):
        num += 1
        img_name = letter + str(num)
        address = "train2/" + img_name + ".jpg"

    latestFile.append(num)

print(latestFile)

# set up camera
cap = cv2.VideoCapture(0)
fps = 30
wait = int(1000/fps) - 1
landmarkError = False
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
font = cv2.FONT_HERSHEY_SIMPLEX

letter = input("what letter: ")

# Set up mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(max_num_hands = 1, min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while (cap.isOpened()):
        ret, frame = cap.read()

        if not ret:
            break

        # Image pre-processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR 2 RGB
        image = cv2.flip(image, 1)  # Flip on horizontal
        image.flags.writeable = False  # Set flag
        results = hands.process(image)  # Detecting hands
        image.flags.writeable = True  # Set flag to true
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Image post-processing

        if results.multi_hand_landmarks:

            # draw hands
            for num, hand in enumerate(results.multi_hand_landmarks):

                if len(hand.landmark) != 21: # check if all landmarks found
                    landmarkError = True
                    break

                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(42, 232, 118), thickness=3,
                                                                 circle_radius=5),
                                          mp_drawing.DrawingSpec(color=(83, 42, 232), thickness=3, circle_radius=3),
                                          )

            if not landmarkError:

                # rectangle logic
                corner1 = [9999, 9999]
                corner2 = [0, 0]

                # rectangle handling
                for hand_landmarks in results.multi_hand_landmarks:
                    for i, landmark in enumerate(hand_landmarks.landmark):

                        x = landmark.x
                        y = landmark.y
                        z = landmark.z

                        if x < corner1[0]:
                            corner1[0] = x
                        if y < corner1[1]:
                            corner1[1] = y
                        if x > corner2[0]:
                            corner2[0] = x
                        if corner2[1] < y:
                            corner2[1] = y

                # draw rectangle
                corner1[0] = int(frame_width * corner1[0]) - 25
                corner1[1] = int(frame_height * corner1[1]) - 25
                corner2[0] = int(frame_width * corner2[0]) + 25
                corner2[1] = int(frame_height * corner2[1]) + 25

                if corner1[0] > frame_width:
                    corner1[0] = frame_width
                if corner1[1] > frame_height:
                    corner1[1] = frame_height
                if corner2[0] < 0:
                    corner2[0] = 0
                if corner2[1] < 0:
                    corner2[1] = 0

                corner1 = tuple(corner1)
                corner2 = tuple(corner2)
                rectColor = (0, 0, 0)
                rectThick = 2
                image = cv2.rectangle(image, corner1, corner2, rectColor, rectThick)

        cv2.imshow('cam', image)
        key = cv2.waitKey(wait)

        if key == ord('c'):
            flipped_frame = cv2.flip(frame, 1)
            index = ord(letter) - ord('a')
            file_name = "train2/" + letter + str(latestFile[index]) + ".jpg"
            latestFile[index] += 1
            cv2.imwrite(file_name, flipped_frame)

        if key == ord('q'):
            break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()