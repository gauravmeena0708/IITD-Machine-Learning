import http.server
import socketserver
import os
import socket
import urllib.request

PORT = 8000
WEB_DIR = os.path.join(os.path.dirname(__file__), 'webapp')

class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

Handler = MyHttpRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    # Get local IP address
    local_ip = socket.gethostbyname(socket.gethostname())
    
    # Get global IP address
    try:
        global_ip = urllib.request.urlopen('https://ifconfig.me/ip').read().decode('utf8')
    except Exception as e:
        global_ip = f"Could not get global IP: {e}"

    print(f"Server running on port {PORT}")
    print(f"Local IP: http://{local_ip}:{PORT}")
    print(f"Global IP: http://{global_ip}:{PORT}")
    
    httpd.serve_forever()
