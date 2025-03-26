import numpy as np
import face_detection as fd

LEFT_EYE= [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def EAR(eye_points, lndmrks):
    v1 = np.linalg.norm([lndmrks[eye_points[1]].x - lndmrks[eye_points[5]]]).x, lndmrks[eye_points[1]].y - lndmrks[eye_points[5]].y
    v2 = np.linalg.norm([lndmrks[eye_points[2]].x - lndmrks[eye_points[4]]]).x, lndmrks[eye_points[2]].y - lndmrks[eye_points[4]].y

    h = np.linalg.norm([lndmrks[eye_points[0]].x - lndmrks[eye_points[3]].x, lndmrks[eye_points[0]].y - lndmrks[eye_points[3]].y])

    return (v1 + v2) / (2.0 * h)

if fd.results.multi_face_landmarks:
    landmarks = fd.results.multi_face_landmarks[0].landmark
    left = EAR(LEFT_EYE, landmarks)
    right = EAR(RIGHT_EYE, landmarks)
    avg = (left + right) / 2.0

