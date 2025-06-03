# ALONE

### The environment setup
Anaconda is suggested to be installed to manage the test environments.

### Prerequisites
- Linux or macOS
- Python >=3.6
- numpy, pandas
- CPU or NVIDIA GPU + CUDA CuDNN
- tensorflow >=1.15

### Overview
The main training loop for ALONE can be found in ```src/simulator/master_QoE_4G/ALONE/train_master_gener.py```,
The corresponding reinforcement learning methods for ALONE are located in the ```src/simulator/master_QoE_4G/ALONE/``` directory.
In addition, the directory ```src/simulator/master_QoE_4G/``` also includes the ABR virtual environment code ```env.py```, 
the ALONE public parameter settings file ```constants.py```, and other files.

### Usage
When running ALONE, you can execute it using the ```src/driver/abr/ALONE.sh``` script, 
where you can specify the directory for saving results using ```--save-dir```, 
the training set using ```--train-trace-dir```, the validation set using ```--val-trace-dir```, and so on.