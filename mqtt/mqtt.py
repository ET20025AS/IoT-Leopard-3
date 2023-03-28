"""
    Grundgerüst-Code von Internetseite: https://tutorials-raspberrypi.de/datenaustausch-raspberry-pi-mqtt-broker-client/   
    letzter Zugriff: 28.03.2023 15:20 Uhr
"""
import paho.mqtt.client as mqtt
import re

MQTT_SERVER = "localhost"
MQTT_TOPIC1 = "iot/dhbw/leopard3/automatic_control"
MQTT_TOPIC2 = "iot/dhbw/leopard3/manual_control"


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client_raspi, userdata, flags, rc):
    print("Connected with result code " + str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client_raspi.subscribe(MQTT_TOPIC1)
    client_raspi.subscribe(MQTT_TOPIC2)


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    message = str(msg.payload.decode())
    print(message)
    # actual stop mechanism
    # if(message == "stop"):
    #    client.disconnect()
    # filtering  the topics
    if msg.topic == MQTT_TOPIC1:
        # message: "x_pos, ypos"
        numbers = re.findall(r'-?\d+\.?\d*', message)
        x = numbers[0]
        y = numbers[1]
        print(x, y)
    elif msg.topic == MQTT_TOPIC2:
        # message: commands: "up", "down", "left", "right", "shoot"
        if message == "up":
            # Younes part (I2C Communication to Arduino)
            # Platzhalter
            a = 1
        elif message == "down":
            # Younes part (I2C Communication to Arduino)
            # Platzhalter
            a = 2
        elif message == "left":
            # Younes part (I2C Communication to Arduino)
            # Platzhalter
            a = 3
        elif message == "right":
            # Younes part (I2C Communication to Arduino)
            # Platzhalter
            a = 4
        elif message == "shoot":
            # Younes part (I2C Communication to Arduino)
            # Platzhalter
            a = 5


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(MQTT_SERVER, 1883, 60)
client.loop_forever()
