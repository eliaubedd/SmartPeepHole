from gpiozero import Button, LED
from signal import pause
import requests

button = Button(26, bounce_time=0.1)
led = LED(16)

def on_press():
    print("Doorbell pressed!")
    led.on()
    response = requests.post("http://localhost:8000/verify")
    print("Verification response:", response.text)

button.when_pressed = on_press
button.when_released = led.off
pause()
