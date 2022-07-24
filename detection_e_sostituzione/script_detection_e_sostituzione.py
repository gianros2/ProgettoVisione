import numpy as np
import cv2

def kmeans_algo(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001),
            rounds,
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

def swap(image_input, swap_image_input, id):
    image = image_input.copy()
    swap_image = swap_image_input.copy()
    dest = image_input.copy()
    kmeans = kmeans_algo(image, clusters=2)
    if id == 0:
        kmeans = kmeans_algo(image, clusters=3)

    gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    if id == 0:
        blackAndWhiteImage = cv2.bitwise_not(blackAndWhiteImage)

    # Iteriamo su ogni pixel partendo da ogni angolo dell'immagine e ci fermiamo nel momento in cui otteniamo il primo pixel bianco
    # Alla fine dei cicli for otterremo i 4 angoli del pannello pubblicitario
    dimensions = blackAndWhiteImage.shape
    row = dimensions[0] - 1
    col = dimensions[1] - 1
    top_left = (0,0)
    for i in range (0, row):
        if blackAndWhiteImage[i,0] == 255:
            top_left = (i,0)
            break
    bottom_left = (row,0)
    for i in range (row, 0, -1):
        if blackAndWhiteImage[i,0] == 255:
            bottom_left = (i,0)
            break
    top_right = (0,col)
    for i in range (0, row):
        if blackAndWhiteImage[i,col] == 255:
            top_right = (i,col)
            break
    bottom_right = (row,col)
    for i in range (row, 0, -1):
        if blackAndWhiteImage[i,col] == 255:
            bottom_right = (i,col)
            break
    blackAndWhiteImage[blackAndWhiteImage == 255] = 0
    blackAndWhiteImage[top_left] = 255
    blackAndWhiteImage[bottom_left] = 255
    blackAndWhiteImage[top_right] = 255
    blackAndWhiteImage[bottom_right] = 255

    # Omografia e sostituzione
    size = swap_image.shape
    pts_source = np.array(
                        [
                        [0,0],
                        [size[1] - 1, 0],
                        [size[1] - 1, size[0] -1],
                        [0, size[0] - 1 ]
                        ],dtype=float
                        )
    pts_dst = np.array([[top_left[1], top_left[0]], [top_right[1], top_right[0]], [bottom_right[1], bottom_right[0]], [bottom_left[1], bottom_left[0]]])
    h, status = cv2.findHomography(pts_source, pts_dst)
    temp = cv2.warpPerspective(swap_image, h, (image.shape[1], image.shape[0]))
    # ~
    cv2.fillConvexPoly(dest, pts_dst.astype(int), 0, 16)
    return dest + temp


def detect_and_swap(darknet, layer_names, image, swap_adidas, swap_fedex, swap_ps3, confidence, threshold):
    boxes = []
    confidences = []
    classIDs = []
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    darknet.setInput(blob)
    outputs = darknet.forward(layer_names)

    for output in outputs:
        for detection in output:            
            # Estrae gli scores, classid, e la confidenza della previsione
            scores = detection[5:]
            classID = np.argmax(scores)
            conf = scores[classID]
            
            # Considerare solo le previsioni > soglia di confidenza
            if conf > confidence:
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, w, h = box.astype('int')

                # Tramite le coordinate del centro, larghezza e altezza otteniamo le coordinate dell'angolo in alto a sinistra
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(conf))
                classIDs.append(classID)
    
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

    # Iteriamo sulle bounding box
    if len(idxs) > 0:
        for i in idxs.flatten():
            # Estrazione delle coordinate dei bounding box
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            b_box = image[y:y+h, x:x+w]
            swap_image = swap_adidas.copy()
            if classIDs[i] == 1:
                swap_image = swap_fedex.copy()
            elif classIDs[i] == 2:
                swap_image = swap_ps3.copy()
            result = swap(b_box, swap_image, classIDs[i])
            image[y:y+h, x:x+w] = result
            
    return image


image_path = "inserire path dell'immagine"
weights_path = 'inserire path dei pesi'
config_path = 'inserire path del file .cfg'
names_path = 'inserire path del file .names'
swap_adidas_path = "inserire path dell'immagine di swap adidas"
swap_fedex_path = "inserire path dell'immagine di swap fedex"
swap_ps3_path = "inserire path dell'immagine di swap ps3"
video_path = 'inserire path del video'

confidence = 0.6
threshold = 0.6

labels = open(names_path).read().strip().split('\n')

darknet = cv2.dnn.readNetFromDarknet(config_path, weights_path)

layer_names = darknet.getLayerNames()
layer_names = [layer_names[i - 1] for i in darknet.getUnconnectedOutLayers()]

image = cv2.imread(image_path)
swap_adidas = cv2.imread(swap_adidas_path)
swap_fedex = cv2.imread(swap_fedex_path)
swap_ps3 = cv2.imread(swap_ps3_path)

on_video = False # impostare a true per effettuare la sostituzione sul video

if not on_video:
    image = detect_and_swap(darknet, layer_names, image, swap_adidas, swap_fedex, swap_ps3, confidence, threshold)

    cv2.imshow('detection', image)
    cv2.imwrite('result.jpg', image)
    cv2.waitKey(0)
else:
    cv2.startWindowThread()
    cap = cv2.VideoCapture(video_path)
    while(True):
        ret, frame = cap.read()
        image = detect_and_swap(darknet, layer_names, frame, swap_adidas, swap_fedex, swap_ps3, confidence, threshold)
        cv2.imshow('frame',image)
        if cv2.waitKey(1) & 0xFF == ord('e'):
            # interrompe il loop se l'utente preme "e"
            break
    cap.release()
    cv2.destroyAllWindows()