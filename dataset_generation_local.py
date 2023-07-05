import cv2
import face_recognition
#Image Saving Path
savingPATH="original_image/"
# initializing Camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

# For each person, enter one numeric face id
face_id = input('\n Enter Name/id  ==> ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")

# Initialize individual sampling face count
count = 1
print("[INSTRUCTIONS]\n==>Press Space Bar to take photo\n==>Press ESC for Exit")
while (True):

    ret, img = cam.read()
    img_org = img.copy()
    # img = cv2.flip(img, -1) # flip video image vertically
    faceCurFrame = face_recognition.face_locations(img)

    if faceCurFrame:
        for faceLoc in faceCurFrame:
            y1, x2, y2, x1 = faceLoc
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('image', img)

    if not ret:
        break
    key = cv2.waitKey(1) % 256

    if key == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif key == 32:
        # SPACE pressed
        # Save the captured image into the datasets folder
        cv2.imwrite(savingPATH + str(face_id) + "_" + str(count) + ".jpg", img_org)
        print("[LOG] Image Captured (Image Count:" + str(count) + ")")
        count += 1

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()


