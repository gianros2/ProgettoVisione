
import glob, os
import pandas as pd

def count(path, frames, label):
    os.chdir(path)
    for file in glob.glob("*.txt"):
        frames += 1
        with open(file, 'r') as f:
	        for line in f:
                    label += 1
    print(path)
    os.chdir('../../..')
    return frames, label 

os.chdir('../')
folder = 'dataset'

subfolders = [ f.name for f in os.scandir(folder) if f.is_dir() ]

Adidas_cv_frames = 0
Adidas_cv_label = 0
Adidas_train_frames = 0
Adidas_train_label = 0
Fedex_cv_frames = 0
Fedex_cv_label = 0
Fedex_train_frames = 0
Fedex_train_label = 0
PS3_cv_frames = 0
PS3_cv_label = 0
PS3_train_frames = 0
PS3_train_label = 0

path_list = []

for i in range(len(subfolders)):
    subdir = [f.name for f in os.scandir(folder+'/'+subfolders[i]) if f.is_dir() ]
    for j in range(len(subdir)):
        path = folder+'/'+subfolders[j]+'/'+subdir[i]
        path_list.append(path)
print(path_list)

for elements in path_list:
    if "adidas/cv" in elements:
        Adidas_cv_frames, Adidas_cv_label = count(elements, Adidas_cv_frames, Adidas_cv_label)
    elif "fedex/cv" in elements:
        Fedex_cv_frames, Fedex_cv_label = count(elements, Fedex_cv_frames, Fedex_cv_label)
    elif "ps3/cv" in elements:
        PS3_cv_frames, PS3_cv_label = count(elements, PS3_cv_frames, PS3_cv_label)
    elif "adidas/train" in elements:
        Adidas_train_frames, Adidas_train_label = count(elements, Adidas_train_frames,Adidas_train_label)
    elif "fedex/train" in elements:
        Fedex_train_frames, Fedex_train_label = count(elements, Fedex_train_frames, Fedex_train_label)
    elif "ps3/train" in elements:
         PS3_train_frames, PS3_train_label = count(elements, PS3_train_frames, PS3_train_label)
     

cols = pd.MultiIndex.from_tuples([("Adidas", "Training Set"), ("Adidas", "Cross Validation Set"),("Fedex", "Training Set"),
                                  ("Fedex", "Cross Validation Set"), ("PS3", "Training Set"), ("PS3", "Cross Validation Set")])

data=[[Adidas_train_frames,Adidas_cv_frames, Fedex_train_frames,Fedex_cv_frames,PS3_train_frames,PS3_cv_frames], 
      [Adidas_train_label,Adidas_cv_label, Fedex_train_label,Fedex_cv_label,PS3_train_label,PS3_cv_label]]

df = pd.DataFrame(data, columns=cols,index=('#Immagini','#Labels'))
df = df.transpose()
print(df)

df.to_excel("dataset/Dataset.xlsx", sheet_name = 'Data')