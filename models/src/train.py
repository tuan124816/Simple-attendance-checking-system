import cv2
import os
from keras_facenet import FaceNet
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle


def train_new_model():
    faces = []
    Id = []
    IDUnique = []
    for subdir in os.listdir('processed_data'): #liệt kê mọi folder trong folder directory
        if os.path.isdir(os.path.join('processed_data', subdir)):
            IDUnique.append(subdir) # append mọi tên folder 
    for id in IDUnique:
        for facedir in os.listdir(os.path.join('processed_data', id)):
            face = cv2.imread(os.path.join('processed_data', id, facedir))
            faces.append(face)
            Id.append(id)

    # face embedder
    embedder = FaceNet()
    embeddings = embedder.embeddings(faces)
    # split train and test set
    X_train, X_test, y_train, y_test = train_test_split(embeddings, Id, test_size=.1, random_state=42)

    # 
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(X_train)
    testX = in_encoder.transform(X_test)
    out_encoder = LabelEncoder()
    out_encoder.fit(y_train)
    trainy = out_encoder.transform(y_train)
    testy = out_encoder.transform(y_test)
    model = SVC(kernel='linear')
    model.fit(trainX, trainy)

    # predict
    yhat_train = model.predict(trainX)
    yhat_test = model.predict(testX)
    # score
    score_train = accuracy_score(trainy, yhat_train)
    score_test = accuracy_score(testy, yhat_test)
    # summarize
    print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

    # filename = 'svm_saved.sav'
    # pickle.dump(model, open(filename, 'wb'))
    # encoder_file_name = 'out_encoder.pkl'
    # pickle.dump(out_encoder, open(encoder_file_name, 'wb'))
    with open('models/weights/svm_saved.sav', 'wb') as file:
        pickle.dump(model, file)
    with open('models/weights/out_encoder.pkl', 'wb') as file:
        pickle.dump(out_encoder, file)