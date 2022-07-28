import numpy as np
from PIL import Image, ImageEnhance
import cv2

def kmeans_quant(image,k):
    i = np.float32(image).reshape(-1,3)
    condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,3000,0.0001)
    ret,label,center = cv2.kmeans(i, k , None, condition,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    final_img = center[label.flatten()]
    final_img = final_img.reshape(image.shape)
    return final_img

def find_top_left_c(quant, rows):
    top_left = (0, 0)
    for i in range(0, rows-1):
        top_left = (i, 0)
        if (quant[i, 0, :] == [255, 255, 255]).all() and (quant[i+1, 0, :] == [255, 255, 255]).all():
            break
    return top_left

def find_bottom_left_c(quant, rows):
    bottom_left = (rows, 0)
    for i in range(rows, 0, -1):
        bottom_left = (i, 0)
        if (quant[i, 0, :] == [255, 255, 255]).all() and (quant[i-1, 0, :] == [255, 255, 255]).all():
            break
    return bottom_left

def find_top_right_c(quant, rows, columns):
    top_right = (0, columns)
    for i in range(0, rows-1):
        top_right = (i, columns)
        if (quant[i, columns, :] == [255, 255, 255]).all() and (quant[i+1, columns, :] == [255, 255, 255]).all():
            break
    return top_right

def find_bottom_right_c(quant, rows, columns):
    bottom_right = (rows, columns)
    for i in range(rows, 0, -1):
        bottom_right = (i, columns)
        if (quant[i, columns, :] == [255, 255, 255]).all() and (quant[i-1, columns, :] == [255, 255, 255]).all():
            break
    return bottom_right

def swap(quant, swap_image, dest):
    rows = quant.shape[0] - 1
    columns = quant.shape[1] - 1

    top_left = find_top_left_c(quant, rows)

    bottom_left = find_bottom_left_c(quant, rows)

    top_right = find_top_right_c(quant, rows, columns)

    bottom_right = find_bottom_right_c(quant, rows, columns)

    pts_swap = np.array([[0, 0], [swap_image.shape[1]-1, 0], [swap_image.shape[1]-1, swap_image.shape[0]-1],[0, swap_image.shape[0]-1]])
    pts_bbox = np.array([top_left[::-1], top_right[::-1], bottom_right[::-1], bottom_left[::-1]])
    h, status = cv2.findHomography(pts_swap, pts_bbox)
    im_out = cv2.warpPerspective(swap_image, h, (dest.shape[1],dest.shape[0]))
    cv2.fillConvexPoly(dest, pts_bbox.astype(int), 0, 16)
    return im_out + dest


def detect_and_swap(darknet, layer_names, image, swap_adidas, swap_fedex, swap_ps3, confidence_threshold):
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
            if conf > confidence_threshold:
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, w, h = box.astype('int')

                # Tramite le coordinate del centro, larghezza e altezza otteniamo le coordinate dell'angolo in alto a sinistra
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(conf))
                classIDs.append(classID)
    
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, confidence_threshold)

    black_and_white = image.copy()
    black_and_white = kmeans_quant(black_and_white,4)
    ret,black_and_white = cv2.threshold(black_and_white,127,255,cv2.THRESH_BINARY)
    ret, binary_map = cv2.threshold(black_and_white,127,255,0)
    binary_map = cv2.cvtColor(binary_map, cv2.COLOR_RGB2GRAY)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)
    areas = stats[1:,cv2.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if areas[i] >= 200:   #keep
            result[labels == i + 1] = 255
    black_and_white[result[:,:] == 0] = 0
    # black_and_white = cv2.bitwise_not(black_and_white)

    # Iteriamo sulle bounding box
    if len(idxs) > 0:
        for i in idxs.flatten():
            # Estrazione delle coordinate dei bounding box
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            x = x - 10
            w = w + 10
            if x < 0 or y < 0:
                continue
            b_box_b = black_and_white[y:y+h, x:x+w]
            b_box = image[y:y+h, x:x+w]
            swap_image = swap_adidas.copy()
            if classIDs[i] == 1:
                swap_image = swap_fedex.copy()
            elif classIDs[i] == 2:
                swap_image = swap_ps3.copy()
            result = swap(b_box_b.copy(), swap_image.copy(), b_box.copy())
            image[y:y+h, x:x+w] = result
            
    return image

if __name__ == '__main__':
    image_path = 'path immagine'
    weights_path = '../darknet/pesi/yolov4-tiny/yolov4-tiny.weights'
    config_path = '../darknet/cfg/yolov4-tiny.cfg'
    names_path = '../darknet/data/yolo.names'
    swap_adidas_path = "lete.jpg"
    swap_fedex_path = "visa.jpg"
    swap_ps3_path = "nike.jpg"
    video_path = 'path video'

    confidence_threshold = 0.7

    labels = open(names_path).read().strip().split('\n')

    darknet = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    layer_names = darknet.getLayerNames()
    layer_names = [layer_names[i - 1] for i in darknet.getUnconnectedOutLayers()]

    swap_adidas = cv2.imread(swap_adidas_path)
    swap_fedex = cv2.imread(swap_fedex_path)
    swap_ps3 = cv2.imread(swap_ps3_path)

    #FILTRI
    #nike
    swap_adidas = cv2.convertScaleAbs(swap_adidas, alpha = 1, beta = 35) 
    #dhl
    swap_fedex = cv2.convertScaleAbs(swap_fedex, alpha = 1, beta = 25) 
    #Image
    swap_adidas = Image.fromarray(swap_adidas)
    swap_fedex = Image.fromarray(swap_fedex)
    # Contrasto
    enhancer = ImageEnhance.Contrast(swap_adidas)
    enhancer1 = ImageEnhance.Contrast(swap_fedex)
    #dhl
    swap_adidas = enhancer.enhance(1.25)
    #nike
    swap_fedex = enhancer1.enhance(0.7)
    #Opencv
    swap_adidas = np.asarray(swap_adidas)
    swap_fedex = np.asarray(swap_fedex)

    on_video = True # impostare a true per effettuare la sostituzione sul video

    if not on_video:
        image = cv2.imread(image_path)
        image = detect_and_swap(darknet, layer_names, image, swap_adidas, swap_fedex, swap_ps3, confidence_threshold)

        cv2.imshow('detection', image)
        cv2.imwrite('result.jpg', image)
        cv2.waitKey(0)
    else:
        cv2.startWindowThread()
        video = cv2.VideoCapture(video_path)
        frame_total = video.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_width = int(video.get(3))
        frame_height = int(video.get(4))
   
        size = (frame_width, frame_height)
        result = cv2.VideoWriter('result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, size)
        count = 0
        percent = 0
        while(True):
            ret, frame = video.read()
            if ret:
                count += 1
                image = detect_and_swap(darknet, layer_names, frame, swap_adidas, swap_fedex, swap_ps3, confidence_threshold)
                result.write(image)
                if int((count/frame_total)*100) != percent:
                    percent = int((count/frame_total)*100)
                    print(str(percent)+'%', end="\r", flush=True)
                if count == frame_total:
                    print('Video salvato!')
                if cv2.waitKey(1) & 0xFF == ord('e'): # interrompe il loop se l'utente preme "e"
                    break
            else:
                break
        video.release()
        cv2.destroyAllWindows()