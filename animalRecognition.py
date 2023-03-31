import cv2
import numpy as np
import pyautogui
import threading
import json
import time
from flask import Flask, Response, render_template, jsonify, request, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from utils import mqtt_connect, mqtt_client, send_notification_and_image

last_notification_time = 0
notification_interval = 300

# Load the model and weights
# net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
net = cv2.dnn.readNet("yolov7-tiny.weights", "yolov7-tiny.cfg")
# net = cv2.dnn.readNet("yolov7.weights", "yolov7.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Define classes using coco.names file
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define the confidence and NMS thresholds for object detection
conf_threshold = 0.6
nms_threshold = 0.4

# Starten der Kamera
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('Birds_12___4K_res.mp4')

# Define the classes to detect
# animal_classes = ["bird", "cat", "dog", "horse", "sheep", "cow", "bear", "person"]
animal_classes = ["bird", "person"]
animal_class_ids = [classes.index(animal_class) for animal_class in animal_classes]
detected_objects = []

# Initialize the output frame and a lock for thread-safe frame exchange
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)
app.secret_key = "your-secret-key-here"


@app.route("/get_detected_objects", methods=["GET"])
def get_detected_objects():
    # Return a list of detected objects as a JSON response
    global detected_objects
    with lock:
        data = detected_objects.copy()
    return jsonify(data)


@app.route("/video_feed")
def video_feed():
    # Return the video feed as a Response object
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/manual_control", methods=["POST"])
def manual_control():
    # Publish the manual control direction to MQTT
    direction = request.form.get("direction")
    mqtt_client.publish("manual_control", direction)
    return "OK"


@app.route("/automatic_control", methods=["POST"])
def automatic_control():
    # Publish the selected object's coordinates for automatic control to MQTT
    object_index = int(request.form.get("object_index"))
    with lock:
        selected_object = detected_objects[object_index]
    coordinates = json.dumps(selected_object["Position"])
    mqtt_client.publish("automatic_control", coordinates)
    return "OK"


def generate():
    # Generate the video feed
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

            # Encode the current frame as JPEG (saves bandwidth and is faster)
            (flag, encodedImage) = cv2.imencode(".jpg", img)

            if not flag:
                continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


# Initialize the LoginManager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


# Create a User class that inherits from UserMixin
class User(UserMixin):
    def __init__(self, id):
        self.id = id


# A dictionary to store users with their corresponding passwords (for demo purposes only)
# In a production environment, use a secure database to store user credentials
users = {'admin': {'password': 'password'}}


@login_manager.user_loader
def load_user(user_id):
    if user_id not in users:
        return
    return User(user_id)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form.get('username')
        password = request.form.get('password')

        if user_id in users and users[user_id]['password'] == password:
            user = User(user_id)
            login_user(user)
            return redirect(url_for('index'))

        return render_template('login.html', error='Invalid username or password.')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/')
@login_required
def index():
    # return the rendered template
    return render_template("index.html")


def object_detection():
    # Perform object detection on the input frames
    global net, classes, conf_threshold, nms_threshold, outputFrame, lock, detected_objects, horse_detected, last_notification_time, notification_interval
    while True:
        # Screenshot des Monitors machen
        screen = pyautogui.screenshot()

        # Record the time when the screenshot is taken
        timestamp = time.time()

        # Convert the screenshot object to a NumPy array
        img = np.array(screen)
        # Convert the color format from RGB to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        resize_scale = 1  # Adjust this value to change the image resolution
        new_width = int(img.shape[1] * resize_scale)
        new_height = int(img.shape[0] * resize_scale)
        img = cv2.resize(img, (new_width, new_height))

        # Create a blob object from the input image
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Set the blob object as the input for the model
        net.setInput(blob)

        # Perform the forward pass operations
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(output_layers)

        # Detect animals in the frame
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

        # Apply Non-Maximum Suppressio
        indices = cv2.dnn.NMSBoxes(animal_boxes, animal_confidences, conf_threshold, nms_threshold)

        # Draw the detected animals on the frame
        species_array = []
        animalId = 0
        horse_detected = False

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = animal_boxes[i]
                label = f"{animal_classes_detected[i]} #id:{animalId}: {animal_confidences[i]:.2f}"
                cv2.rectangle(img, (x, y), (x + w, y + h), [255, 0, 0], 4)
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 0], 2)
                species_array.append({"Species": animal_classes_detected[i], "id": animalId, "Position": [x, y, w, h]})
                animalId = animalId + 1

                if animal_classes_detected[i] == "horse":
                    horse_detected = True

        if horse_detected and (time.time() - last_notification_time) >= notification_interval:
            last_notification_time = time.time()
            send_notification_and_image(img, "A wild horse has been detected!")

        # Print the detected objects
        # print(species_array)

        ## Update the current frame while avoiding race condition with the generate() thread
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
