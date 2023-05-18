"""""""""""""""""""""""""""""""""""
    Imports
"""""""""""""""""""""""""""""""""""
import cv2
import numpy as np
import pyautogui
import threading
import json
import time
from flask import Flask, Response, render_template, jsonify, request, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from utils import mqtt_connect, mqtt_client, send_notification_and_image
import base64
import socket
from sort.sort import Sort

last_notification_time = 0
notification_interval = 300

""""""""""""""""""""""""""""""""""" 
    Object Tracking Initializations 
"""""""""""""""""""""""""""""""""""
# Load the model and weights
weights_path = 'yolov7-tiny.weights'
config_path  = 'yolov7-tiny.cfg'
classes_file = 'coco.names'

# read class names
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# initialize network
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# set tracking target classes
animals = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'person', 'giraffe']
animals_ids = [classes.index(animal) for animal in animals if animal in classes]

# initialize SORT tracker
tracker = Sort()
detected_objects = []

# output layer handling
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers_indices = net.getUnconnectedOutLayers().flatten()
    output_layers = [layer_names[i - 1] for i in output_layers_indices]
    return output_layers

"""""""""""""""""""""""""""""""""""
    Threading Initializations / Flask / Socket
"""""""""""""""""""""""""""""""""""
# Initialize the output frame and a output_frame_lock for thread-safe frame exchange
outputFrame = None
output_frame_lock = threading.Lock()

current_frame = None
input_frame_lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)
app.secret_key = "your-secret-key-here"

# Festlegen der Größe des Empfangspuffers
BUFF_SIZE = 65536

# Erstellen des Sockets für die Kommunikation
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Erhöhen der Größe des Empfangspuffers
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)

# Abrufen des Hostnamens und der IP-Adresse des Hosts
host_name = socket.gethostname()
host_ip = '192.168.188.29'

# Drucken der IP-Adresse des Hosts
print(host_ip)

# Festlegen des Ports und Senden einer Nachricht an den Server
port = 9999
message = b'PC'
client_socket.sendto(message, (host_ip, port))


def receive_frames():
    global current_frame, output_frame_lock
    while True:
        packet, _ = client_socket.recvfrom(BUFF_SIZE)
        data = base64.b64decode(packet, ' /')
        npdata = np.fromstring(data, dtype=np.uint8)
        frame = cv2.imdecode(npdata, 1)

        with output_frame_lock:
            current_frame = frame


@app.route("/get_detected_objects", methods=["GET"])
def get_detected_objects():
    # Return a list of detected objects as a JSON response
    global detected_objects
    with output_frame_lock:
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
    mqtt_client.publish("iot/dhbw/leopard3/manual_control", direction)
    return "OK"


@app.route("/automatic_control", methods=["POST"])
def automatic_control():
    # Publish the selected object's coordinates for automatic control to MQTT
    object_index = int(request.form.get("object_index"))
    with output_frame_lock:
        selected_object = detected_objects[object_index]
    coordinates = json.dumps(selected_object["Position"])
    mqtt_client.publish("iot/dhbw/leopard3/automatic_control", coordinates)
    return "OK"


def generate():
    # Generate the video feed
    global outputFrame, output_frame_lock
    # display_interval = 1  # Update the display every second
    # last_display_time = 0
    while True:
        with output_frame_lock:
            if outputFrame is None:
                continue

            img, timestamp = outputFrame

            # current_time = time.time()

            # # Update the time difference display every second
            # if current_time - last_display_time >= display_interval:
            #     # Calculate the time difference
            #     time_difference = current_time - timestamp
            #
            #     # Display the time difference on the frame
            #     time_text = f"Time_lag {time_difference:.2f}s"
            #     last_display_time = current_time
            #
            # cv2.putText(img, time_text, (img.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
    global net, classes, conf_threshold, nms_threshold, outputFrame, output_frame_lock, detected_objects, last_notification_time, notification_interval, current_frame
    while True:
        # Screenshot des Monitors machen
        # frame = pyautogui.screenshot()

        # Get the current frame from client.py
        with output_frame_lock:
            if current_frame is None:
                continue
            frame = current_frame.copy()

        # Record the time when the screenshot is taken
        timestamp = time.time()

        # Convert the screenshot object to a NumPy array
        img = np.array(frame)
        # Convert the color format from RGB to BGR
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # resize_scale = 1  # Adjust this value to change the image resolution
        # new_width = int(img.shape[1] * resize_scale)
        # new_height = int(img.shape[0] * resize_scale)
        # img = cv2.resize(img, (new_width, new_height))

        height, width, channels = img.shape

        # create blob from image for processing
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        # extract detections from net output
        detections = []
        for output in outs:
            for detection in output.reshape(-1, 85):
                scores = detection[5:]
                class_id = np.argmax(scores)
                if class_id in animals_ids:
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # object img calculation
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = center_x - w // 2
                        y = center_y - h // 2

                        # add detection to list
                        detections.append([x, y, x + w, y + h, confidence, class_id])

        # apply NMS to detections
        if len(detections) > 0:
            detections = np.array(detections)
            nms_threshold = 0.4
            keep_indices = cv2.dnn.NMSBoxes(detections[:, :4].tolist(), detections[:, 4].tolist(), 0.5, nms_threshold)
            nms_detections = np.array([detections[i] for i in keep_indices.flatten()])
        else:
            nms_detections = np.empty(shape=(0, 6))

        # track objects with SORT
        tracked_objects = tracker.update(nms_detections)

        # draw imgs and descriptions in camera feed
        species_array = []
        person_detected = False
        for tracked_object in tracked_objects:   
            x, y, x2, y2, obj_id, class_id = map(int, tracked_object)
            obj_name = classes[class_id]
            label = f"{obj_name} ID: {obj_id}"
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x2, y2), color, 2)
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            species_array.append({"Species": obj_name, "id": obj_id, "Position": [(x+x2)//2, (y+y2)//2]})

            if obj_name == "person":
                person_detected = True

        if person_detected and (time.time() - last_notification_time) >= notification_interval:
            last_notification_time = time.time()
            send_notification_and_image(img, "An intruder has been detected!")

        ## Update the current frame while avoiding race condition with the generate() thread
        with output_frame_lock:
            detected_objects = species_array
            outputFrame = (img.copy(), timestamp)


"""""""""""""""""""""""""""""""""""
    Main program
"""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
    mqtt_connect()

    # start a thread that will perform object detection
    t = threading.Thread(target=object_detection)
    t.daemon = True
    t.start()

    # start a thread that will receive frames from the client
    t_receive = threading.Thread(target=receive_frames)
    t_receive.daemon = True
    t_receive.start()

    # start the flask app
    app.run(host="0.0.0.0", port=8000, debug=True,
            threaded=True, use_reloader=False)
