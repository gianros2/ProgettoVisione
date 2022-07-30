import os
import shutil
import random
import cv2

def creaDirectory():
    for f in folders:
        os.makedirs(f, exist_ok=True)

path = 'path contenente i frames'
#Creiamo 5 cartelle con i nomi specificati nella variabile folders
folders = ('Dario', 'Gianni', 'Rocco', 'cv', 'test')
creaDirectory()
#Specifichiamo quante immagini selezionare (in questo caso 100)
n = 100

#Copiamo all'interno di ogni cartella n (100) .jpg casuali contenuti in frames
for f in folders:
    files = os.listdir(path)
    for file_name in random.sample(files, n):
        shutil.move(os.path.join(path, file_name), f)