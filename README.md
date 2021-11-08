# echo-ai-HCM-detection
Detect HCM from ECG and echocardiogram

# Requirements
1. Each client should have 'client.py', 'utils.py' and 'model.py' residing in their own respective machines. 
2. Server and Client should be able to connect to same IP and port.

# Training Procedure
1. Run 'server.py' and wait till it connects to provided IP and port.
2. Run 'client.py' for each client. After enough clients are connected to server, training will start automatically and the rest is handled by Flower framework.
