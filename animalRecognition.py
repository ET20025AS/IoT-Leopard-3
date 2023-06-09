import cv2
import numpy as np
import pyautogui
import threading
import json
import time
import paho.mqtt.client as mqtt
from flask import Flask, Response, render_template, jsonify, request

# Laden des YOLOv4-Modells und seiner Gewichte
#net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
#net = cv2.dnn.readNet("yolov7-tiny.weights", "yolov7-tiny.cfg")
net = cv2.dnn.readNet("yolov7.weights", "yolov7.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Definieren der Klassen
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Definieren der Schwellenwerte für die Erkennung
conf_threshold = 0.6
nms_threshold = 0.4

# Starten der Kamera
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('Birds_12___4K_res.mp4')

# Definieren der zu erkennenden Klassen
animal_classes = ["bird", "cat", "dog", "horse", "sheep", "cow", "bear", "person"]
animal_class_ids = [classes.index(animal_class) for animal_class in animal_classes]
detected_objects = []

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
# app = Flask(__name__)
app = Flask(__name__)

mqtt_client = mqtt.Client()


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


@app.route("/get_detected_objects", methods=["GET"])
def get_detected_objects():
    global detected_objects
    with lock:
        data = detected_objects.copy()
    return jsonify(data)


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/manual_control", methods=["POST"])
def manual_control():
    direction = request.form.get("direction")
    mqtt_client.publish("manual_control", direction)
    return "OK"


@app.route("/automatic_control", methods=["POST"])
def automatic_control():
    object_index = int(request.form.get("object_index"))
    with lock:
        selected_object = detected_objects[object_index]
    coordinates = json.dumps(selected_object["Position"])
    mqtt_client.publish("automatic_control", coordinates)
    return "OK"


def mqtt_connect():
    global mqtt_client
    try:
        mqtt_client.connect("localhost", 1883, 60)
        mqtt_client.publish("Status", "Animal Recognition Python Script running")
        print("connection with mqtt successfull")
    except:
        print("connection with mqtt not successfull")


def generate():
    global outputFrame, lock
    display_interval = 1  # Update the display every second
    last_display_time = 0
    while True:
        with lock:
            if outputFrame is None:
                continue

            img, timestamp = outputFrame

            current_time = time.time()

            # Update the time difference display every second
            if current_time - last_display_time >= display_interval:
                # Calculate the time difference
                time_difference = current_time - timestamp

                # Display the time difference on the frame
                time_text = f"Time_lag {time_difference:.2f}s"
                last_display_time = current_time

            cv2.putText(img, time_text, (img.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Aktuellen Frame in JPEG Format encoden (sparen von bandbreite + schneller)
            (flag, encodedImage) = cv2.imencode(".jpg", img)

            if not flag:
                continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


def object_detection():
    global net, classes, conf_threshold, nms_threshold, outputFrame, lock, detected_objects
    while True:
        # Lesen des Kamerabildes
        # success, img = cap.read()

        # Screenshot des Monitors machen
        screen = pyautogui.screenshot()

        # Record the time when the screenshot is taken
        timestamp = time.time()

        # Umwandeln des Screenshot-Objekts in ein NumPy-Array
        img = np.array(screen)
        # Convert the color format from RGB to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        resize_scale = 0.5  # Adjust this value to change the image resolution
        new_width = int(img.shape[1] * resize_scale)
        new_height = int(img.shape[0] * resize_scale)
        img = cv2.resize(img, (new_width, new_height))

        # Erstellen eines Blob-Objekts aus dem Eingabebild
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Setzen des Blob-Objekts als Eingabe für das Modell
        net.setInput(blob)

        # Durchführen der Vorwärtsdurchlauf-Operationen
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
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
        #colors = np.random.uniform(0, 255, size=(len(animal_boxes), 3))
        species_array = []
        animalId = 0
        
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = animal_boxes[i]
                label = f"{animal_classes_detected[i]} #id:{animalId}: {animal_confidences[i]:.2f}"
                #color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), [255, 0, 0], 4)
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 0], 2)
                species_array.append({"Species": animal_classes_detected[i], "id": animalId, "Position": [x, y, w, h]})
                animalId=animalId+1
                
        # Ausgabe der erkannten Objekte
        print(species_array)

        # Aktualisieren des aktuellen Frames. Aber erst dann, wenn generate thread nicht gerade outputFrame am lesen ist, um racecondition zu vermeiden
        with lock:
            detected_objects = species_array
            outputFrame = (img.copy(), timestamp)


if __name__ == '__main__':
    mqtt_connect()
    # start a thread that will perform object detection
    t = threading.Thread(target=object_detection)
    t.daemon = True
    t.start()
    # start the flask app
    app.run(host="0.0.0.0", port=8000, debug=True,
            threaded=True, use_reloader=False)
