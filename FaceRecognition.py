import cv2 as cv
import numpy as np
import os
from deepface import DeepFace
from retinaface import RetinaFace

# Inicializa a câmera
liveCap = cv.VideoCapture(0)

if not liveCap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = liveCap.read()

    if not ret:
        print("Failed to capture frame")
        continue

    img = frame.copy()

    # Detecta faces
    faces = RetinaFace.detect_faces(img)

    if not faces:
        print("Nenhuma face detectada.")
    else:
        try:
            # Reconhecimento facial
            dfs = DeepFace.find(img_path=img, db_path="dataset", model_name="VGG-Face",
                                enforce_detection=False)

            # Verifica se há resultados válidos
            if isinstance(dfs, list) and len(dfs) > 0 and not dfs[0].empty:
                first_match = dfs[0].iloc[0]
                identity_path = first_match["identity"]

                # Normaliza o caminho para compatibilidade com Windows e Linux
                identity_parts = os.path.normpath(identity_path).split(os.sep)

                # Evita erro de índice
                if len(identity_parts) >= 2:
                    name = identity_parts[-2]  # Nome da pasta dentro de "dataset"
                else:
                    name = "Desconhecido"

                confidence = 1 - first_match["distance"]
                print(f"Recognized: {name} (Confidence: {confidence:.2f})")

                # Exibe o nome na imagem
                cv.putText(img, name, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                print("No matching face found")

        except ValueError:
            print("No face detected")

    # Exibe a imagem na tela
    cv.imshow("Detecção Facial", img)

    # Sai do loop ao pressionar 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha as janelas
liveCap.release()
cv.destroyAllWindows()
