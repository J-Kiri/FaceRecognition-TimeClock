from time import sleep

import cv2 as cv
import os

name = input("Insira o nome do funcionario: ")
os.makedirs(f"dataset/{name}", exist_ok = True) # Create folder with the person's name

cap = cv.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break
    cv.imshow("Capturing - Pressione Enter para capturar", frame)

    # Waits for the KeyPress to capture the photo of the person
    if cv.waitKey(1) == ord('n'):
        cv.imwrite(f"dataset/{name}/{count}.jpg", frame)
        print(f"Foto {count + 1} capturada!")
        count += 1

    # Waits 50 photos
    if count >= 50:
        break

    # cv.imshow("capturing", frame)
    #
    # while count < 50:
    #     cv.imwrite(f"dataset/{name}/{count}.jpg", frame)
    #     print(f"Foto {count + 1} capturada!")
    #     count += 1
    #     sleep(1)

    # Or q KeyPress
    if cv.waitKey(1) == ord('q') or count >= 50:
        break

# Release the camera and close windows
cap.release()
cv.destroyAllWindows()
