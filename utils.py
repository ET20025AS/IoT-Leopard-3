import cv2
from telegram import InputFile, Bot
import paho.mqtt.client as mqtt
import io
import asyncio

# Telegram Bot config
TOKEN = '5909099367:AAHyVpl-KBTBxAFWezoTIajDTX-cB0Ng-7M'
CHAT_ID = '5332989880'

mqtt_client = mqtt.Client()


async def send_telegram_notification(image, caption):
    # Send a Telegram message and photo to the specified chat
    bot = Bot(token=TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text=caption)
    await bot.send_photo(chat_id=CHAT_ID, photo=InputFile(io.BytesIO(cv2.imencode('.jpg', image)[1].tobytes()),
                                                          filename="detected_object.jpg"))


def send_notification_and_image(image, caption):
    # Run the send_telegram_notification() function asynchronously
    asyncio.run(send_telegram_notification(image, caption))


def mqtt_connect():
    # Connect to the MQTT broker
    global mqtt_client
    try:
        mqtt_client.connect("localhost", 1883, 60)
        mqtt_client.publish("Status", "Animal Recognition Python Script running")
        print("connection with mqtt successfull")
    except:
        print("connection with mqtt not successfull")
