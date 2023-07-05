import cv2
import numpy as np
import face_recognition
import pickle
import os

# Importing face images
folderPath = 'original_image' #Image Folder Path
pathList = os.listdir(folderPath)
print(pathList)
imageList = []
personID_List = []
personName_List=[]

print("[INFO] Starting Encoding")
print("[INFO] Fetching Face IDs")
for path in pathList:
    imageList.append(cv2.imread(os.path.join(folderPath, path)))
    personName_List=os.path.splitext(path)[0].split("_")
    personID_List.append(personName_List[0])
print(personID_List)

print("[INFO] Adding Face IDs to recognition")

# image Correction flow
def avg_pixelVAL(img_file):
    img_file = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)
    avg_pixel_val = np.mean(np.array(img_file))
    return avg_pixel_val
def checkFACE(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if (len(face_recognition.api.face_encodings(img)) != 0):
        return True
    else:
        return False

def findEncodings(imagesList):
    i=-1
    encodeList = []
    rejected_nameList=[]
    reject_encodeList =[]
    for img in imagesList:
        i += 1
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faceENCODE=face_recognition.api.face_encodings(img)
        if len(faceENCODE)!=0:
            #Face Found
            encode = faceENCODE[0]
            encodeList.append(encode)
            # print("[INFO] [RECOGNITION] "+face_recognition.api.face_encodings(img))

            print("[LOG] [RECOGNITION] File Added in Encoding:"+pathList[i])
        else: # Face Not Found
            rejected_nameList.append(pathList[i])
            reject_encodeList.append(img)
            print("[LOG] [RECOGNITION] File Skipped to Encoding:"+pathList[i])
    #Enhnacing and Encoding
    print("---------------------------------------\n[INFO] Enhancing Skipped Images\n---------------------------------------")
    index=0
    for img_file in reject_encodeList:
        img_edit = img_file
        avgP=avg_pixelVAL(img_edit)
        #If Image is Dark then It will Lighten the image
        if avgP > 100:
            # converting to LAB color space
            lab = cv2.cvtColor(img_edit, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            # Applying CLAHE to L-channel
            # feel free to try different values for the limit and grid size:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)

            # merge the CLAHE enhanced L-channel with the a and b channel
            limg = cv2.merge((cl, a, b))

            # Converting image from LAB Color model to BGR color spcae
            img_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            cv2.imwrite("enhanced_image" + "/lab_" + str(rejected_nameList[index]), img_enhanced)

        else:
            # print("[INFO] [IMG CORRECTION]: " + str(pathList[index]))
            lookUpTable = np.empty((1, 256), np.uint8)
            for i in range(256):
                lookUpTable[0, i] = np.clip(pow(i / 255.0, 0.45) * 255.0, 0, 255)  # Change 0.45 as per need
            img_enhanced = cv2.LUT(img_file, lookUpTable)
            img_enhanced = cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2BGR)

            cv2.imwrite("enhanced_image" + "/gamma_" + str(rejected_nameList[index]), img_enhanced)



        if checkFACE(img_enhanced):
            encode = face_recognition.api.face_encodings(img_enhanced)[0]
            encodeList.append(encode)
            print("[LOG] [CORRECTION] [ENHANCED] File Added in Encoding:"+rejected_nameList[index]+"\t\t[PIXEL AVG:"+str(avgP.round(2))+"]")

        else:
            print("[LOG] [ENHANCED] File Discarded from Encoding:"+rejected_nameList[index]+"\t\t[PIXEL AVG:"+str(avgP.round(2))+"]")

        index+=1
    return encodeList

print("[INFO] Encoding STARTED")
encodeListKnown = findEncodings(imageList)
encodeListKnownWithIds = [encodeListKnown, personID_List]
print("[INFO] Encoding COMPLETED")


file = open("face_encoding.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("[INFO] Encoding File Saved")