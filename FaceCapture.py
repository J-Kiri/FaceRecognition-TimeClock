import cv2 as cv
import os

name = input("Insira o nome do funcionario: ")
os.makedirs(f"dataset/{name}", exist_ok = True)

cap = cv.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break
    cv.imshow("Capturing - Pressione Enter para capturar", frame)

    if cv.waitKey(1) == ord('n'):
        cv.imwrite(f"dataset/{name}/{count}.jpg", frame)
        print(f"Foto {count + 1} capturada!")
        count += 1

    if count >= 50:
        break

    if cv.waitKey(1) == ord('q') or count >= 50:
        break

cap.release()
cv.destroyAllWindows()
