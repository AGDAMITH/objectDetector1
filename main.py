import cv2
import numpy as np


classes = []
classFile = 'coco.names'
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()
# file_name = input("Enter image path :")
cap = cv2.VideoCapture("Volleyball.mp4")

while cap.isOpened():
        ret, img = cap.read()
        height, width, _ = img.shape

        blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

        net.setInput(blob)

        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

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
                    y = int(center_y - w/2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                (text_width, text_height) = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1.5, thickness=1)[0]
                box_coords = ((x, y+5), (x + text_width + 2, y-15))
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.rectangle(img, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
                cv2.putText(img, label + " " + confidence, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
        cv2.imshow("Output", img)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
