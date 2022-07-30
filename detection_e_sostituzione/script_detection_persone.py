import cv2
import numpy as np


def detect_people(image, confidence_threshold):
    weights_path = '../darknet/pesi/people-detection/yolo-people-detection.weights'
    config_path = '../darknet/cfg/people-detection/yolo-people-detection.cfg'

    darknet = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    layer_names = darknet.getLayerNames()
    layer_names = [layer_names[i - 1] for i in darknet.getUnconnectedOutLayers()]
    
    boxes = []
    confidences = []
    classIDs = []
    coords = []
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    darknet.setInput(blob)
    outputs = darknet.forward(layer_names)

    for output in outputs:
        for detection in output:            
            scores = detection[5:]
            classID = np.argmax(scores)
            conf = scores[classID]
            
            if conf > confidence_threshold:
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, w, h = box.astype('int')

                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(conf))
                classIDs.append(classID)
    
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, confidence_threshold)

    if len(idxs) > 0:
        for i in idxs.flatten():
            if classIDs[i] == 0 or classIDs[i] == 1 or classIDs[i] == 2 or classIDs[i] == 4:
                # Estrazione delle coordinate dei bounding box
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]
                y = y - 3
                if x < 0 or y < 0:
                    continue
                coords.append((x, y, w, h))

    return coords