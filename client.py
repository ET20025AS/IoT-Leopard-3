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
host_ip = '192.168.0.29'

# Drucken der IP-Adresse des Hosts
print(host_ip)

# Festlegen des Ports und Senden einer Nachricht an den Server
port = 9999
message = b'PC'
client_socket.sendto(message, (host_ip, port))

# Erstellen eines Fensters zur Anzeige des empfangenen Videos
cv2.namedWindow("RECEIVING VIDEO", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RECEIVING VIDEO", 400, 200)

# Endlose Schleife zur Verarbeitung der empfangenen Video-Frames
while True:
    # Empfangen der Video-Datenpakete und Dekodieren der Base64-codierten Daten
    packet, _ = client_socket.recvfrom(BUFF_SIZE)
    data = base64.b64decode(packet, ' /')

    # Umwandeln der Daten in ein Numpy-Array mit uint8-Datentyp
    npdata = np.fromstring(data, dtype=np.uint8)

    # Dekodieren der Numpy-Daten als Bild mit cv2.imdecode()
    frame = cv2.imdecode(npdata, 1)

    # Anpassen der Größe des Bildes auf 800x600
    # frame = cv2.resize(frame, (800, 600))

    # Anzeigen des Bildes im Fenster "RECEIVING VIDEO"
    cv2.imshow("RECEIVING VIDEO", frame)

    # Warten auf eine Tastatureingabe (maximal 1 Millisekunde)
    key = cv2.waitKey(1) & 0xFF

    # Wenn die Taste "q" gedrückt wird, schließen Sie den Socket und brechen Sie die Schleife ab
    if key == ord('q'):
        client_socket.close()
        break

# Schließen des Fensters und Beenden des Programms
cv2.destroyAllWindows()
