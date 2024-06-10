import cv2

cam = cv2.VideoCapture(0)
cam.set(3, 640) # lebar kamera
cam.set(4, 480) # tinggi kamera

faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    retV, frame = cam.read()
    if not retV:
        break

    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(abuAbu, 1.3, 5) # frame, scaleFractor, min Neighbours
    
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow('Webcamku', frame)
    esc = cv2.waitKey(1) & 0xFF
    if esc == 27 or esc == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()