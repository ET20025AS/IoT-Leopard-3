"""
    Grundgerüst-Code von Internetseite: https://tutorials-raspberrypi.de/datenaustausch-raspberry-pi-mqtt-broker-client/   
    letzter Zugriff: 28.03.2023 15:20 Uhr
"""
import paho.mqtt.client as mqtt
import re

#I2C libraries 
import sys 
import smbus2 as smbus
import time 

# Slave Adresse
I2C_SLAVE_ADDRESS = 11
#create the I2C bus 
I2Cbus=smbus.SMBus(1)

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
        
        I2Cbus.write_byte(I2C_SLAVE_ADDRESS,x)
        time.sleep(3) 
        I2Cbus.write_byte(I2C_SLAVE_ADDRESS,y)
        time.sleep(3)
     
    elif msg.topic == MQTT_TOPIC2:
        # message: commands: "up", "down", "left", "right", "shoot"

        if message == "up":
            num=254
            I2Cbus.write_byte(I2C_SLAVE_ADDRESS,num)
            time.sleep(3)
        elif message == "down":
            num=253
            I2Cbus.write_byte(I2C_SLAVE_ADDRESS,num)
            time.sleep(3)
        elif message == "left":
            num=252
            I2Cbus.write_byte(I2C_SLAVE_ADDRESS,num)
            time.sleep(3)
        elif message == "right":
            num=251
            I2Cbus.write_byte(I2C_SLAVE_ADDRESS,num)
            time.sleep(3)
        elif message == "shoot":
            num=250
            I2Cbus.write_byte(I2C_SLAVE_ADDRESS,num)
            time.sleep(3)    

           
              


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(MQTT_SERVER, 1883, 60)
client.loop_forever()
