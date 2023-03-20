import cv2
import numpy as np

# Laden des YOLOv4-Modells und seiner Gewichte
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Definieren der Klassen
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Definieren der Schwellenwerte für die Erkennung
conf_threshold = 0.8
nms_threshold = 0.4

# Starten der Kamera
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('Birds_12___4K_res.mp4')

# benennen Sie das Fenster um und setzen Sie den Modus auf cv2.WINDOW_NORMAL
cv2.namedWindow("Animal Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Animal Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Definieren der zu erkennenden Klassen
animal_classes = ["bird", "cat", "dog", "horse", "sheep", "cow", "bear", "person"]
animal_class_ids = [classes.index(animal_class) for animal_class in animal_classes]

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

    # Erkennung der Tiere
    animal_boxes = []
    animal_classes_detected = []
    animal_confidences = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id in animal_class_ids and confidence > conf_threshold:
                center_x = int(detection[0] * img.shape[1])
                center_y = int(detection[1] * img.shape[0])
                width = int(detection[2] * img.shape[1])
                height = int(detection[3] * img.shape[0])
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                animal_boxes.append([left, top, width, height])
                animal_classes_detected.append(animal_classes[animal_class_ids.index(class_id)])
                animal_confidences.append(float(confidence))

    # Anwenden der Nicht-Maximum-Unterdrückung
    indices = cv2.dnn.NMSBoxes(animal_boxes, animal_confidences, conf_threshold, nms_threshold)

    # Markierung der erkannten Tiere
    colors = np.random.uniform(0, 255, size=(len(animal_boxes), 3))
    species_array = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = animal_boxes[i]
            label = f"{animal_classes_detected[i]}: {animal_confidences[i]:.2f}"
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            species_array.append({"Species": animal_classes_detected[i], "Position": [x, y, w, h]})

    # Anzeigen des Bildes
    print(species_array)
    cv2.imshow("Animal Detection", img)
    
    # Abbrechen bei Drücken der "q"-Taste
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Freigabe der Ressourcen
cap.release()
cv2.destroyAllWindows()
