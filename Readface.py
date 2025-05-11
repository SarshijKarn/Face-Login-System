"""
@author: Sarshij K
"""
import cv2
import csv
import numpy as np
import os
from PIL import Image, ImageTk
import pandas as pd

name, Id = '', ''
dic = {
    'Name': name,
    'Ids': Id
}

def store_data():
    global name, Id, dic
    name = input("Enter Name: ").strip()
    Id = input("Enter Id: ").strip()
   
    dic = {
        'Ids': Id,
        'Name': name
    }
    return dic

# Function to check if entered ID is number
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def TakeImages():
    dict1 = store_data()

    if name.isalpha() and is_number(Id):
        fieldnames = ['Name', 'Ids']
        file_exists = os.path.isfile('Profile.csv')

        mode = 'w' if Id == '1' else 'a'
        with open('Profile.csv', mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists or Id == '1':
                writer.writeheader()
            writer.writerow(dict1)

        cam = cv2.VideoCapture(0)

        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                sampleNum += 1
                path = os.path.join("TrainingImage", f"{name}.{Id}.{sampleNum}.jpg")
                cv2.imwrite(path, gray[y:y + h, x:x + w])

            cv2.imshow('Capturing Face for Login', img)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum > 60:
                break

        cam.release()
        cv2.destroyAllWindows()

        print(f"Images Saved for Name: {name} with ID: {Id}")
        print("Images save location is 'TrainingImage/'")

    else:
        if name.isalpha():
            print('Enter a proper numeric ID.')
        elif is_number(Id):
            print('Enter a proper name (alphabetic only).')
        else:
            print('Enter both proper Name and ID.')

TakeImages()
