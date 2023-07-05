import os
import pickle
import cv2
import face_recognition
import numpy as np

#3rd Module
#Here RGB is converted to BGR as Camera is capturing in GBR

print("[INFO] Starting Video Capture")
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


# Load the encoding file
print("[INFO] Loading Encode File")
file = open('face_encoding.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, faceIds = encodeListKnownWithIds
# print(faceIds)
print("[INFO] Encode File Loaded")

modeType = 0
counter = 0
id = -1
imgface = []
avgDis=[]
avgResult=0

print("Press ESC for Quiting Recognition")
while True:
    print("----------------------------")
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)


    if faceCurFrame:
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print("matches", matches)
            # print("[RECOGNITION] faceDis: ", faceDis)

            matchIndex = np.argmin(faceDis)
            avgDis.append(faceDis[matchIndex])
            avgResult=sum(avgDis)/len(avgDis)
            # print("Average: " + str(avgResult))
            percent_match= 100 - int(faceDis[matchIndex] * 100)
            threshold=0.51
            if avgResult<threshold:
                print("[LOG] [RECOGNITION] Face MATCHED ==> Similarity : [",percent_match,"%]")
            elif avgResult>=threshold:
                print("[LOG] [RECOGNITION] Face NOT MATCHED ==> Similarity : [",percent_match,"%]")
            # print("Match Index", matchIndex)
            NAME = ""
            if matches[matchIndex]:
                rgb=(255,255,255)
                # print(faceDis[matchIndex])
                if faceDis[matchIndex]<threshold:
                    rgb=(0,255,0)
                    NAME=faceIds[matchIndex]
                    print("[LOG] [RECOGNITION] Known Face Detected: ",faceIds[matchIndex])

                elif faceDis[matchIndex]>=threshold:
                    rgb=(0,0,225)
                    NAME="Unknown"
                    print("[LOG] [RECOGNITION] Unknown Face Detected")

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2),rgb , 2)
                cv2.putText(img, NAME, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, rgb, 1)

    cv2.imshow("Face Recognition", img)

    if cv2.waitKey(1) % 256 == 27:
        # ESC pressed
        print("[INFO] Escape hit, Screen Closing...\nDone")
        break