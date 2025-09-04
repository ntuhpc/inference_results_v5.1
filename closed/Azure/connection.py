import socket

try:
    s = socket.socket()
    s.settimeout(2)
    s.connect(("ccwcus-gpu-17", 30000))
    print("✅ Connected")
except socket.error as e:
    print(f"❌ Not connected: {e}")
finally:
    s.close()
