import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import Normalizer, LabelEncoder
import pickle
from datetime import datetime
from openpyxl import Workbook

def recognize():
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FPS, 20)

    detector = MTCNN()
    embedder = FaceNet()
    out_encoder = LabelEncoder()
    in_encoder = Normalizer(norm='l2')
    filename_path = 'models/weights/svm_saved.sav'
    encoder_file_path = 'models/weights/out_encoder.pkl'
    loaded_model = pickle.load(open(filename_path, 'rb'))
    loaded_out_encoder = pickle.load(open(encoder_file_path, 'rb'))
    wb = Workbook()
    sheet = wb.active
    sheet['A1'] = 'Name'
    sheet['B1'] = 'Time'
    # We need to check if camera
    # is opened previously or not
    if (vid.isOpened() == False): 
        print("Error reading video file")
    # frame_width = vid.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
    # frame_height = vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))

    size = (frame_width, frame_height)

    #result = cv2.VideoWriter(f'final.avi', cv2.VideoWriter_fourcc(*'MJPG'),10, size)
    interval = 5  #muốn cách bao nhiêu ảnh
    frame_count = 0 
    attended = set()
    while(True):
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        if not ret:
            # If the frame was not successfully read, break out of the loop
            break

        #result.write(frame)
        # Display the resulting frame
        # cv2.imshow('frame', frame) 
        if frame_count % interval == 0:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect_faces(image)
            if len(result) > 0:
                for each_result in result:
                    bounding_box = each_result['box']
                    keypoints = each_result['keypoints']
                    crop_img = image[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]]
                    resized_image = cv2.resize(crop_img, (160, 160))
                    # save_img = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
                    embeddings = embedder.embeddings([resized_image]) #########################
                    testX = in_encoder.transform(embeddings)
                    y_pred_encoded = loaded_model.predict(testX)
                    y_pred = loaded_out_encoder.inverse_transform(y_pred_encoded)
                    time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    reg_frame = cv2.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3]), (255,0,0), 2)
                    reg_frame = cv2.cvtColor(reg_frame, cv2.COLOR_BGR2RGB)
                    cv2.putText(reg_frame, text=str(y_pred[0]), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, thickness=3, org=(bounding_box[0], bounding_box[1]-10), color=(0,255,0))
                    if y_pred[0] in attended:
                        continue
                    else:
                        attended.add(y_pred[0])
                        sheet.append([y_pred[0], time_now])

                

        cv2.imshow('frame', reg_frame) 
        
        frame_count += 1
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            wb.save(filename="attend list.xlsx")
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()