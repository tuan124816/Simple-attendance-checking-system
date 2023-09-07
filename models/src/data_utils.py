# model/src/data_utils.py
import os
import cv2

def create_processed_data_directory():

    processed_directory = "processed_data"
    if os.path.exists(processed_directory):
        print(f"{processed_directory} already exists.")
    else:
        print(f"{processed_directory} is not exist")
        # create directory
        os.mkdir(processed_directory)
        print(f"{processed_directory} created successfully.")


def capture_student_images(user_id):
    # Capture and save student images for training
    cap = cv2.VideoCapture(0)  # Use the appropriate camera index (e.g., 0 for default camera)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_num = 0
    interval = 5

    while True:
        ret, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.08, 7)

        for (x, y, w, h) in faces:
            crop_img = frame[y:y + w, x:x + w]
            roi_color = cv2.resize(crop_img, (160, 160))

            if face_num % interval == 0:
                img_filename = os.path.join('data/processed', user_id, f'frame_{face_num}.jpg')
                cv2.imwrite(img_filename, roi_color)

            face_num += 1

        cv2.imshow('Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def preprocess_image(image):
    # Preprocess the input image before face recognition
    preprocessed_image = cv2.resize(image, (160, 160))
    
    # You can add additional preprocessing steps here, such as normalization, etc.
    
    return preprocessed_image

