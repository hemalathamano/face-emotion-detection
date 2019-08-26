import cv2
import numpy as np
font = cv2.FONT_HERSHEY_SIMPLEX

age_list = ['(0 - 2)', '(4 - 6)', '(8 - 12)', '(15 - 20)', '(25 - 32)', '(38 - 43)', '(48 - 53)', '(60 - 80)']
gender_list = ['Male', 'Female']
# MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
MODEL_MEAN_VALUES = (200, 250, 300)
age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
#gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#cap = cv2.VideoCapture(0)
img=cv2.imread('download (2).jpg')
while 1:
    # Capture frame-by-frame
    #ret, frame = cap.read()
   # cv2.imshow("Faces found", img)

    # Our operations on the frame come here
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # To draw a rectangle in a face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

 # Get Face
        face_img = img[y:y + h, h:h + w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            # Predict Age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        print("Age Range: " + age)

            # Predict Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        print("Gender : " + gender)

    overlay_text = "%s " %(age)
    cv2.putText(img, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
