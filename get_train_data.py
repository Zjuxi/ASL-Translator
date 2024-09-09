# this is a program to get data to use to train

import cv2, numpy, csv
import mediapipe as mp

# Set up mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hand_dict = {
    "land0x": [], "land0y": [], "land0z": [],  # WRIST
    "land1x": [], "land1y": [], "land1z": [],  # THUMB_CMC
    "land2x": [], "land2y": [], "land2z": [],  # THUMB_MCP
    "land3x": [], "land3y": [], "land3z": [],  # THUMB_IP
    "land4x": [], "land4y": [], "land4z": [],  # THUMB_TIP
    "land5x": [], "land5y": [], "land5z": [],  # INDEX_FINGER_MCP
    "land6x": [], "land6y": [], "land6z": [],  # INDEX_FINGER_PIP
    "land7x": [], "land7y": [], "land7z": [],  # INDEX_FINGER_DIP
    "land8x": [], "land8y": [], "land8z": [],  # INDEX_FINGER_TIP
    "land9x": [], "land9y": [], "land9z": [],  # MIDDLE_FINGER_MCP
    "land10x": [], "land10y": [], "land10z": [],  # MIDDLE_FINGER_PIP
    "land11x": [], "land11y": [], "land11z": [],  # MIDDLE_FINGER_DIP
    "land12x": [], "land12y": [], "land12z": [],  # MIDDLE_FINGER_TIP
    "land13x": [], "land13y": [], "land13z": [],  # RING_FINGER_MCP
    "land14x": [], "land14y": [], "land14z": [],  # RING_FINGER_PIP
    "land15x": [], "land15y": [], "land15z": [],  # RING_FINGER_DIP
    "land16x": [], "land16y": [], "land16z": [],  # RING_FINGER_TIP
    "land17x": [], "land17y": [], "land17z": [],  # PINKY_MCP
    "land18x": [], "land18y": [], "land18z": [],  # PINKY_PIP
    "land19x": [], "land19y": [], "land19z": [],  # PINKY_DIP
    "land20x": [], "land20y": [], "land20z": [],   # PINKY_TIP
    "letter": []
}

size = (1280, 720)

with mp_hands.Hands(static_image_mode=True,
                    max_num_hands=1,
                    min_detection_confidence=0.8,
                    min_tracking_confidence=0.5) as hands:
    for alph in range(26):
        letter = chr(alph + ord('a'))

        for num in range(30):
            img_name = letter + str(num)
            address = "train2/" + img_name + ".jpg"

            # open img
            image = cv2.imread(address)

            # process img
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # if landmarks detected
            if results.multi_hand_world_landmarks:
                hand_dict['letter'].append(letter)

                # draw landmarks
                for num, hand in enumerate(results.multi_hand_landmarks):

                    # Verify if all 21 landmarks are detected
                    if len(hand.landmark) != 21:
                        print(f"problem with: {img_name}, missing landmarks for hand {num + 1}")


                    mp_drawing.draw_landmarks(image_rgb, hand, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(42, 232, 118), thickness=3,
                                                                     circle_radius=5),
                                              mp_drawing.DrawingSpec(color=(83, 42, 232), thickness=3, circle_radius=3),
                                              )

                # write to dictionary
                for hand_landmarks in results.multi_hand_world_landmarks:
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        # Extract the landmark coordinates
                        x = landmark.x
                        y = landmark.y
                        z = landmark.z
                        hand_dict['land' + str(i) + 'x'].append(x)
                        hand_dict['land' + str(i) + 'y'].append(y)
                        hand_dict['land' + str(i) + 'z'].append(z)

                # show img
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                '''
                cv2.imshow('image', image_bgr)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    img_height, img_width = image.shape[:2]
                    print("(manual) problem with:", img_name, "size = (" + str(img_width) + ", " + str(img_height) + ")")
                cv2.destroyAllWindows()'''

            else:
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image_bgr,
                            f'problem with {img_name}',
                            (50, 100),
                            font, 3,
                            (55, 55, 200),
                            10,
                            cv2.LINE_4)

                img_height, img_width = image.shape[:2]
                print(f"problem with: {img_name}, size({img_width}, {str(img_height)})")
                cv2.imshow('image', image_bgr)
                cv2.waitKey(500)

                cv2.destroyAllWindows()

with open('train3.csv', mode='w', newline='') as file:
    # Create a DictWriter object, passing the fieldnames (keys of the dictionary)
    writer = csv.DictWriter(file, fieldnames=hand_dict.keys())

    # Write the header
    writer.writeheader()

    # Write the dictionary data
    dict_len = len(hand_dict[next(iter(hand_dict))])
    for i in range(dict_len):
        row = {key: hand_dict[key][i] for key in hand_dict.keys()}
        writer.writerow(row)