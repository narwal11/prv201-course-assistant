# Attack: DNS lookup (can be used to exfiltrate data even when HTTP is blocked).
import socket
try:
    print("ESCAPED: resolved", socket.gethostbyname("example.com"))
except Exception as e:
    print("contained:", type(e).__name__)
