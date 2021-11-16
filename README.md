# echo-ai-HCM-detection
Detect HCM from ECG and echocardiogram

# Setup
- Create conda environment using 'conda env create -f environment.yml'
- Both 'ECG' and 'Echo' folder has 'server.py', 'client.py', 'utils.py' and 'model.py'. 'server.py' file should reside on machine that needs to operate as central body to aggregate and distrubute weights. 'client.py', 'utils.py' and 'model.py' files should be copied to each client machine that needs to participate in the training. 


# Requirements
1. Each client should have 'client.py', 'utils.py' and 'model.py' residing in their own respective machines.
2. Server and Client should be able to connect to same IP and port.

# Training Procedure
1. Run 'server.py' and wait till it connects to provided IP and port.
2. Run 'client.py' for each client. After enough clients are connected to server, training will start automatically and the rest is handled by Flower framework.
