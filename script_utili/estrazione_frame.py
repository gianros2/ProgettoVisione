import os
import cv2

input_path = "input path contenente i video"
output_path = "output path che conterrà i frame estratti"
videos = []
#Ogni tupla corrisponde ai timecode da estrarre per ogni video
times = ((14, 19, 23, 28), (46, 54, 73, 81), (96, 102))

#Scansiona la cartella alla ricerca dei file mp4
for (path, dirs, files) in os.walk(input_path, topdown=True):
    for filename in files:
        if '.mp4' in filename:
            videos.append(input_path + filename)
            
#Ordina in modo crescente i nomi dei video aggiunti alla lista videos          
videos.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

#Scansiona ogni video ed estrae i frame in base al tempo
name_count = 0
for i, video in enumerate(videos):
    # print("Inizio elaborazione video : " + str(i))
    for j in range(0, len(times[i]), 2):
        count = 0
        time_single = times[i]
        start_time_ms = time_single[j] * 1000
        stop_time_ms = time_single[j+1] * 1000
        success = True
        vidcap = cv2.VideoCapture(video)

        fps = vidcap.get(cv2.CAP_PROP_FPS)
        # print("Saltiamo i frame in base alla lunghezza del video /5")
        frame_skip = fps // 5
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, time_single[j] * fps)

        while success and vidcap.get(cv2.CAP_PROP_POS_MSEC) <= stop_time_ms:
            success, image = vidcap.read()
            # print('Read a new frame: ', success)
            if count % frame_skip == 0:
                cv2.imwrite(output_path + "/frame%d.jpg" % name_count, image)
            count += 1
            name_count += 1

        vidcap.release()
        cv2.destroyAllWindows()

    print("Video numero %i terminato" % i)

print("Fine")

#fps = vidcap.get(cv2.CAP_PROP_FPS)
#vidcap.set(cv2.CAP_PROP_POS_FRAMES, time_single[j] * fps)

#while success and vidcap.get(cv2.CAP_PROP_POS_MSEC) < start_time_ms:
#   success, image = vidcap.read()