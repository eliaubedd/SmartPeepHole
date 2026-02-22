"""
Doorbell Trigger Module

This module listens to a physical doorbell button connected to a Raspberry Pi GPIO pin.
When the button is pressed, it sends an authenticated HTTP request to the local
verification server and activates a status LED during processing.

Environment Variables:
    DEVICE_ID (str): Secret device authentication token used to authorize
                     requests to the verification server.

Hardware Configuration:
    - Button connected to GPIO26
    - LED connected to GPIO16
"""

from gpiozero import Button, LED
from signal import pause
import requests
import socket
from dotenv import load_dotenv
import os



# Environment configuration
load_dotenv()
DEVICE_CODE = os.getenv("DEVICE_ID")

if not DEVICE_CODE:
    raise ValueError("DEVICE_ID not set in environment variables")



# GPIO Initialization
button = Button(26, bounce_time=0.1)
led = LED(16)



# Networking Utilities
def get_ip_address() -> str:
    """
    Retrieve the local IPv4 address of the Raspberry Pi.

    Returns:
        str: Local IPv4 address assigned to the primary network interface.

    Raises:
        OSError: If socket creation or connection fails.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip



# GPIO Event Handlers
def on_press() -> None:
    """
    Handle doorbell button press event.

    When triggered:
        1. Prints a status message.
        2. Turns on the status LED.
        3. Sends an authenticated HTTP POST request to the local
           verification server endpoint (/verify).
        4. Prints the server response.

    The request includes an authentication header containing the device code.

    Raises:
        requests.RequestException: If the HTTP request fails.
    """
    print("Doorbell pressed!")
    led.on()

    response = requests.post(
        f"http://{get_ip_address()}:8080/verify",
        headers={"authorization": DEVICE_CODE}
    )

    print("Verification response:", response.text)


# Bind GPIO events to handlers
button.when_pressed = on_press
button.when_released = led.off


# Keep the script running to listen for events
pause()
