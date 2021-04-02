import cv2
import numpy as np

label_path = 'helmet.names'
labels = open(label_path).read().strip().split('\n')

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

weight_path = 'yolov4-tiny_custom_helmet_best.weights'
config_path = 'yolov4-tiny_custom_helmet.cfg'
net = cv2.dnn.readNetFromDarknet(config_path, weight_path)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

img = cv2.imread('image (5).jpg')
height, width, _ = img.shape

blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (608, 608), swapRB=True, crop=False)
net.setInput(blob)
layerOutputs = net.forward(ln)

boxes = []
confidences = []
classIDs = []

for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)
            x = int(center_x - w/2)
            y = int(center_y - h/2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            classIDs.append(class_id)

idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.7)

if len(idxs) > 0:
    for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

cv2.imshow("Frame", img)
cv2.imwrite('output_image.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



