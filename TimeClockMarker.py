import sqlite3
from datetime import datetime
import cv2 as cv
import os

DATABASE_FILE = "recognized_faces.db"

# Initialize the table and Creates the table
def initialize_database():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recognized_faces(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        imagePath TEXT NOT NULL,
        recognitionDate TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Saves the face's name, the image and the date to the database
def save_recognized_face(name, image):
    timestamp = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    imageFilename = f"{name}_{timestamp}.jpg"
    imagePath = os.path.join("recognized_faces", imageFilename)

    os.makedirs("recognized_faces", exist_ok = True)
    cv.imwrite(imagePath, image)

    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO recognized_faces (name, imagePath, recognitionDate)
        VALUES (?, ?, ?)
    ''', (name, imagePath, timestamp))
    conn.commit()
    conn.close()

    print(f"Saved recognized face: {name} at {timestamp}")