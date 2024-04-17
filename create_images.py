import os
import cv2

DATA_DIR = './data1'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 1
dataset_size = 100

# Attempt to find a valid camera index dynamically
valid_camera_index = None
for index in range(10):  # Try indices from 0 to 9
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        valid_camera_index = index
        cap.release()
        break

# Check if a valid camera index is found
if valid_camera_index is None:
    print("Error: No camera found. Check if a camera is connected.")
    exit()

# Open the camera with the valid index
cap = cv2.VideoCapture(valid_camera_index)

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()



