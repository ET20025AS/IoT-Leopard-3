import base64
import cv2
import numpy as np
import socket

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
