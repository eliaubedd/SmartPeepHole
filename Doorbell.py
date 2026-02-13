from gpiozero import Button, LED
from signal import pause
import requests
import socket
from dotenv import load_dotenv
import os

load_dotenv()
DEVICE_CODE = os.getenv("DEVICE_ID")

if not DEVICE_CODE:
    raise ValueError("DEVICE_ID not set in environment variables")

button = Button(26, bounce_time=0.1)
led = LED(16)

def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

def on_press():
    print("Doorbell pressed!")
    led.on()
    response = requests.post(f"http://{get_ip_address()}:8080/verify", headers={"authorization": DEVICE_CODE})
    print("Verification response:", response.text)

button.when_pressed = on_press
button.when_released = led.off
pause()
