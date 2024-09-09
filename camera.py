# this is a way to use the nn for real world purposes

import cv2
import torch
import mediapipe as mp
from train import ASLClassifier, input_size, num_classes

landmarks = []

# Load model
model = ASLClassifier(input_size, num_classes)
model.load_state_dict(torch.load('asl_classifier.pth'))

# set up camera
cap = cv2.VideoCapture(0)
fps = 30
wait = int(1000/fps)
landmarkError = False
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
font = cv2.FONT_HERSHEY_SIMPLEX

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

        img_list = []

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

                # note down world landmarks
                for world_landmarks in results.multi_hand_world_landmarks:
                    for i, landmark in enumerate(world_landmarks.landmark):
                        # Extract the landmark coordinates
                        x = landmark.x
                        y = landmark.y
                        z = landmark.z

                        img_list.append(x)
                        img_list.append(y)
                        img_list.append(z)

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

                # predict letter
                img_tensor = torch.tensor(img_list, dtype=torch.float32)
                data = img_tensor.unsqueeze(0)
                output = model(data)
                _, predicted_letter = torch.max(output, 1)

                letter = chr(predicted_letter.item() + ord('a'))

                # output predicted letter
                cv2.putText(image,
                            f'prediction: {letter}',
                            (corner1[0], corner1[1] - 10),
                            font, 1.2,
                            (55, 55, 200),
                            4,
                            cv2.LINE_4)

        cv2.imshow('cam', image)
        key = cv2.waitKey(wait)

        if key == ord('q'):
            break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()