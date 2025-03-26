import EAR as ear
import pygame

fcounter = 0
alarm_active = False
pygame.mixer.init()
alarm = pygame.mixer.Sound("alarm")
EAR_THRESHOLD = 0.3
CLOSED_FRAMES = 20
if ear.avg < EAR_THRESHOLD:
    fcounter += 1
    if fcounter > CLOSED_FRAMES and not alarm_active:
        alarm_active = True
        print("WAKE UPPPPP")
        alarm.play()
    else:
        fcounter = 0
        alarm_active = False