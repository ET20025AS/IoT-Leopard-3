import cv2
import numpy as np

# Laden des YOLOv4-Modells und seiner Gewichte
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Definieren der Klassen
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Definieren der Schwellenwerte für die Erkennung
conf_threshold = 0.5
nms_threshold = 0.4

# Starten der Kamera
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('Birds_12___4K_res.mp4')

while True:
    # Lesen des Kamerabildes
    success, img = cap.read()
    
    # Erstellen eines Blob-Objekts aus dem Eingabebild
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Setzen des Blob-Objekts als Eingabe für das Modell
    net.setInput(blob)

    # Durchführen der Vorwärtsdurchlauf-Operationen
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    # Erkennung von Vögeln
    bird_class_id = 14
    bird_boxes = []
    bird_confidences = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == bird_class_id and confidence > conf_threshold:
                center_x = int(detection[0] * img.shape[1])
                center_y = int(detection[1] * img.shape[0])
                width = int(detection[2] * img.shape[1])
                height = int(detection[3] * img.shape[0])
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                bird_boxes.append([left, top, width, height])
                bird_confidences.append(float(confidence))

    # Anwenden der Nicht-Maximum-Unterdrückung
    indices = cv2.dnn.NMSBoxes(bird_boxes, bird_confidences, conf_threshold, nms_threshold)

    # Markierung der erkannten Vögel
    colors = np.random.uniform(0, 255, size=(len(bird_boxes), 3))
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = bird_boxes[i]
            label = f"{classes[bird_class_id]}: {bird_confidences[i]:.2f}"
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Anzeigen des Bildes
    cv2.imshow("Bird Detection", img)
    
    # Abbrechen bei Drücken der "q"-Taste
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Freigabe der Ressourcen
cap.release()
cv2.destroyAllWindows()