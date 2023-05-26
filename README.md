# Target-Tracking on a Robotarm:

Trained Model from: yolov7-tiny.cfg, yolov7-tiny.weights
https://github.com/AlexeyAB/darknet/releases

Refer to "requirements.txt" for more information of dependencies

Documentation for this project can be found in our Hackster project: https://www.hackster.io/die-wilden-kerle/target-tracking-on-a-robotarm-6c48c0

There are currently 5 branches that are structured as follows:

1. Arduino Code Robotansteuerung: This includes the C code to control the Arduino and receive the commands over I2C
2. MQTT_Communications: Includes the MQTT Receiver that should run on the Raspberry Pi
3. The feature branches cover the development of the main python script responsible for the telegram bot, object tracking and website.
