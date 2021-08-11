import cv2
import numpy as np
import face_recognition

imgelon = face_recognition.load_image_file('imagebasic/elon musk.png')
imgelon = cv2.cvtColor(imgelon, cv2.COLOR_BGR2RGB)

imgtest = face_recognition.load_image_file('imagebasic/elon test.png')
imgtest = cv2.cvtColor(imgtest, cv2.COLOR_BGR2RGB)

facelock = face_recognition.face_locations(imgelon)[0]
encodeelon = face_recognition.face_encodings(imgelon)[0]
cv2.rectangle(imgelon, (facelock[3], facelock[0]), (facelock[1], facelock[2]), (255, 0, 255), 2)

facelocktest = face_recognition.face_locations(imgtest)[0]
encodeelontest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest, (facelocktest[3], facelocktest[0]), (facelocktest[1], facelocktest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeelon], encodeelontest)
facedis = face_recognition.face_distance([encodeelon], encodeelontest)
print(results, facedis)

cv2.putText(imgtest,f'{results} {round(facedis[0],2)}',(50, 50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('elon musk', imgelon)
cv2.imshow('elon test', imgtest)
cv2.waitKey(0)