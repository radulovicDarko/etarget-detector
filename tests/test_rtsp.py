import socket
import json
import time
import random

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
    msg = {
        "type": "hit",
        "nx": random.random(),
        "ny": random.random(),
        "x": 0,
        "y": 0,
        "area": 10
    }

    sock.sendto(json.dumps(msg).encode(), ("127.0.0.1", 5055))
    print("Sent:", msg)

    time.sleep(1)