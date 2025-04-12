# drowsiness_app.py
import streamlit as st
import cv2
import numpy as np
import pygame
import mediapipe as mp
from threading import Thread, Event

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
EAR_THRESHOLD = 0.25
CLOSED_FRAMES = 20

# Global variables
stop_event = Event()
fcounter = 0
alarm_active = False

def calculate_ear(eye_points, landmarks):
    # Horizontal distance
    h1 = np.linalg.norm([
        landmarks[eye_points[0]].x - landmarks[eye_points[3]].x,
        landmarks[eye_points[0]].y - landmarks[eye_points[3]].y
    ])
    
    # Vertical distances
    v1 = np.linalg.norm([
        landmarks[eye_points[1]].x - landmarks[eye_points[5]].x,
        landmarks[eye_points[1]].y - landmarks[eye_points[5]].y
    ])
    v2 = np.linalg.norm([
        landmarks[eye_points[2]].x - landmarks[eye_points[4]].x,
        landmarks[eye_points[2]].y - landmarks[eye_points[4]].y
    ])
    
    return (v1 + v2) / (2.0 * h1)  # EAR formula

def detect_drowsiness():
    global fcounter, alarm_active
    pygame.mixer.init()
    alarm = pygame.mixer.Sound("alarm.mp3")
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        cap = cv2.VideoCapture(0)
        if stop_event.is_set():
            cap.release()
            return
        
        while not stop_event.is_set():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                try:
                    left_ear = calculate_ear(LEFT_EYE, landmarks)
                    right_ear = calculate_ear(RIGHT_EYE, landmarks)
                    avg_ear = (left_ear + right_ear) / 2.0
                except IndexError:
                    continue  # Skip if landmarks aren't detected properly

                if avg_ear < EAR_THRESHOLD:
                    fcounter += 1
                    if fcounter >= CLOSED_FRAMES and not alarm_active:
                        alarm_active = True
                        alarm.play()
                else:
                    fcounter = 0
                    if alarm_active:
                        alarm_active = False
                        alarm.stop()
            else:
                fcounter = 0  # Reset if face disappears

        cap.release()

def main():
    st.title("Drowsiness Detection System")
    st.write("""
    This system runs in the background and monitors for signs of drowsiness.
    Click 'Start' to begin monitoring and 'Stop' to end the session.
    """)

    if st.button("Start Detection"):
        if not stop_event.is_set():
            stop_event.clear()
            Thread(target=detect_drowsiness, daemon=True).start()
            st.success("Detection started! Continue working normally...")

    if st.button("Stop Detection"):
        stop_event.set()
        st.warning("Detection stopped")

    status = st.empty()
    
    while True:
        status.markdown(f"""
        ### Current Status:
        - **Alarm State:** {'🔴 ACTIVE' if alarm_active else '🟢 INACTIVE'}
        - **Eye Closure Duration:** {fcounter}/{CLOSED_FRAMES} frames
        """)
        if stop_event.is_set():
            Thread(target=detect_drowsiness, daemon=True).start()
            break

if __name__ == "__main__":
    main()