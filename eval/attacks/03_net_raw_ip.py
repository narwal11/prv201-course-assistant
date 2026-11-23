# Attack: raw TCP connection to a public IP, bypassing DNS.
import socket
try:
    socket.create_connection(("1.1.1.1", 53), timeout=3).close()
    print("ESCAPED: TCP connection to 1.1.1.1:53 opened")
except Exception as e:
    print("contained:", type(e).__name__)
