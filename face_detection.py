import mediapipe as mp
import cv2
import numpy as np
import pygame

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.25
CLOSED_FRAMES = 20

fcounter = 0
alarm_active = False
pygame.mixer.init()
alarm = pygame.mixer.Sound("alarm.mp3")

face = mp.solutions.face_mesh
face_mesh = face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

def EAR(eye_points, lndmrks):
    v1 = np.linalg.norm([
        lndmrks[eye_points[1]].x - lndmrks[eye_points[5]].x,
        lndmrks[eye_points[1]].y - lndmrks[eye_points[5]].y
    ])
    v2 = np.linalg.norm([
        lndmrks[eye_points[2]].x - lndmrks[eye_points[4]].x,
        lndmrks[eye_points[2]].y - lndmrks[eye_points[4]].y
    ])
    h = np.linalg.norm([
        lndmrks[eye_points[0]].x - lndmrks[eye_points[3]].x,
        lndmrks[eye_points[0]].y - lndmrks[eye_points[3]].y
    ])
    return (v1 + v2) / (2.0 * h)

cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_ear = EAR(LEFT_EYE, landmarks)
        right_ear = EAR(RIGHT_EYE, landmarks)
        avg_ear = (left_ear + right_ear) / 2.0

        cv2.putText(frame, f'EAR: {avg_ear:.2f}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if avg_ear < EAR_THRESHOLD:
            fcounter += 1
            if fcounter > CLOSED_FRAMES and not alarm_active:
                alarm_active = True
                alarm.play()
        else:
            fcounter = 0
            alarm_active = False
            alarm.stop()

    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()