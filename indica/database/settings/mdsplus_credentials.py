import platform

computer_name = platform.node()
IP_ADDRESS = ""
PORT = "8000"

if computer_name == "COMPUTER_NAME":
    mdsplus_credentials = {"url": f"tcp://{IP_ADDRESS}:{PORT}"}
elif computer_name == "...":
    mdsplus_credentials = {"url": "..."}
else:  # Default
    mdsplus_credentials = {"url": f"tcp://{IP_ADDRESS}:{PORT}"}
