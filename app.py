import time
import cv2 
from flask import Flask, render_template, Response
# import face_recognition
import os
import numpy as np
import math
from lib import face_confidence

app = Flask(__name__)
sub = cv2.createBackgroundSubtractorMOG2()  # create background subtractor

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

# class FaceRecognition:
#     face_locations = []
#     face_encodings = []
#     face_names = []
#     known_face_encodings = []
#     known_face_names = []
#     process_current_frame = True

#     def __init__(self):
#         self.encode_faces()

#     def encode_faces(self):
#         for image in os.listdir('faces'):
#             face_image = face_recognition.load_image_file(f"faces/{image}")
#             # print(face_image.shape)
#             face_encoding = face_recognition.face_encodings(face_image)[0]
#             # print(len(face_encoding))
#             self.known_face_encodings.append(face_encoding)
#             self.known_face_names.append(image.split(".")[0])
#         print(self.known_face_names)

#     def run_recognition(self):
#         video_capture = cv2.VideoCapture(0)

#         if not video_capture.isOpened():
#             print('Video source not found...')

#         while True:
#             ret, frame = video_capture.read()

#             # Only process every other frame of video to save time
#             if self.process_current_frame:
#                 # Resize frame of video to 1/4 size for faster face recognition processing
#                 small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

#                 # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
#                 rgb_small_frame = small_frame[:, :, ::-1]

#                 # Find all the faces and face encodings in the current frame of video
#                 self.face_locations = face_recognition.face_locations(rgb_small_frame)
#                 self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
                
#                 self.face_names = []
#                 for face_encoding in self.face_encodings:
#                     # See if the face is a match for the known face(s)
#                     matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
#                     name = "Unknown"
#                     confidence = '???'

#                     # Calculate the shortest distance to face
#                     face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

#                     best_match_index = np.argmin(face_distances)
#                     if matches[best_match_index]:
#                         name = self.known_face_names[best_match_index]
#                         confidence = face_confidence(face_distances[best_match_index])

#                     self.face_names.append(f'{name} ({confidence})')

#             self.process_current_frame = not self.process_current_frame

#             # Display the results
#             for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
#                 # Scale back up face locations since the frame we detected in was scaled to 1/4 size
#                 top *= 4
#                 right *= 4
#                 bottom *= 4
#                 left *= 4

#                 # Create the frame with the name
#                 cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#                 cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#                 cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
#             # Display the resulting image
#             # cv2.imshow('Face Recognition', frame)
#             frame = cv2.imencode('.jpg', frame)[1].tobytes()
#             yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#                     # Hit 'q' on the keyboard to quit!
#             if cv2.waitKey(1) == ord('q'):
#                 break

#         # Release handle to the webcam
#         video_capture.release()
#         cv2.destroyAllWindows()


# def gen():
#     """Video streaming generator function."""
#     cap = cv2.VideoCapture(0)

#     # Read until video is completed
#     while(cap.isOpened()):
#         ret, frame = cap.read()  # import image
#         if not ret: #if vid finish repeat
#             frame = cv2.VideoCapture(0)
#             continue
#         if ret:  # if there is a frame continue with code
#             image = cv2.resize(frame, (0, 0), None, 1, 1)  # resize image
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts image to gray
#             fgmask = sub.apply(gray)  # uses the background subtraction
#             kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
#             closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
#             opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
#             dilation = cv2.dilate(opening, kernel)
#             retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)  # removes the shadows
#             contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             minarea = 400
#             maxarea = 50000
#             for i in range(len(contours)):  # cycles through all contours in current frame
#                 if hierarchy[0, i, 3] == -1:  # using hierarchy to only count parent contours (contours not within others)
#                     area = cv2.contourArea(contours[i])  # area of contour
#                     if minarea < area < maxarea:  # area threshold for contour
#                         # calculating centroids of contours
#                         cnt = contours[i]
#                         M = cv2.moments(cnt)
#                         cx = int(M['m10'] / M['m00'])
#                         cy = int(M['m01'] / M['m00'])
#                         # gets bounding points of contour to create rectangle
#                         # x,y is top left corner and w,h is width and height
#                         x, y, w, h = cv2.boundingRect(cnt)
#                         # creates a rectangle around contour
#                         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                         # Prints centroid text in order to double check later on
#                         cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,.3, (0, 0, 255), 1)
#                         cv2.drawMarker(image, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, markerSize=8, thickness=3,line_type=cv2.LINE_8)
#         #cv2.imshow("countours", image)
#         frame = cv2.imencode('.jpg', image)[1].tobytes()
#         yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#         #time.sleep(0.1)
#         key = cv2.waitKey(20)
#         if key == 27:
#            break
   
        

@app.route('/video_feed')
def video_feed():
    # fr = FaceRecognition()
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(None,
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run()