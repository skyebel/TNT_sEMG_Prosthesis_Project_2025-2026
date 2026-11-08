# sEMG Prosthetic Hand: Iron Man Gauntlet

**Triton NeuroTech, University of California San Diego | Academic Year 2025-2026**


**Iron Man Gauntlet** is an open-source, low-cost prosthetic hand system developed by Triton NeuroTech at UC San Diego for the [2026 California Neurotechnology Conference (CNTC)](https://caneurotech.vercel.app) at UC Berkeley. The system uses surface electromyography (sEMG) signals to classify hand gestures in real time with a bidirectional LSTM (BiLSTM), enabling intuitive control of a 3D-printed prosthetic arm. Our system prioritizes accessibility, using readily available components at roughly 50x lower cost than a comparable commercial prosthetic arm. Future work will focus on expanding the gesture set to 8+ gestures, adding proportional control, and improving wrist movement and wearability.

**Project Leads:** Skye Belsher, Thejo Tattala

**Project Members:** Enrique Aranda, Peter Little, Byron Chen, Ziqing Zhu, Bora Vanli, Dhruv Sehgal, Shivani Rajanala, Abhay Korlapati, Siddhant Gulati, Rishab Kolan, Tristan Lee

## Pipeline

- **Gesture classification** using 4-channel sEMG (97% accuracy across users, 250 ms execution)
- **BiLSTM model** with automatic feature extraction, no hand-crafted features
- **Stabilized predictions** via rolling vote and softmax averaging across consecutive windows
- **Live prosthetic control** via Arduino over WiFi
- **Interactive GUI** for data collection, training, and inference

## Poster

Presented at the 2026 California Neurotechnology Conference (CNTC) @UC Berkeley.

[Click here to view the research poster (PDF)](https://drive.google.com/file/d/1DYNPa0U_blKSx-6KFc1fAIsUNBEod6rd/view), also available in this repository as `sEMG Prosthetic Hand - Iron Man TNT.pdf`.

For the team's ongoing development repository, including exploratory notebooks and initial project direction, see [TritonNeuroTech-EMG-Prosthetics](https://github.com/earanda4/TritonNeuroTech-EMG-Prosthetics).

## Hardware Components

- Arduino Uno R3
- MindRove 8-channel sEMG armband (four channels used in the current pipeline)
- [PCA9685 16-channel Servo Driver](https://www.adafruit.com/product/815)
- 5 servos, one per digit, with tensioned cable-and-spring actuation
- Silicone fingertips and palm
- Modular arm component with wrist rotation
- *(Exploratory, proof of concept)* 4 force-sensitive resistors (FSRs) for future tactile feedback

## Software Overview

- Real-time signal acquisition via the MindRove SDK with BoardShim
- 60 Hz IIR notch filter (Q = 30) and 4th-order Butterworth bandpass filter (10-200 Hz), applied per channel as cascaded second-order sections (`scipy.signal.sosfiltfilt`)
- Overlapping sliding windows with per-channel normalization (StandardScaler); windows spanning a gesture transition are discarded
- Classification using a 2-layer BiLSTM (hidden size 128) with a batch-normalized, dropout-regularized classification head, trained with AdamW, label-smoothed cross-entropy, and a one-cycle learning rate schedule over 15 epochs (80/20 stratified split)
- Scaler parameters saved during training and reapplied during live inference
- GUI for live demo of calibration, training, and inference
- Servo commands sent from the GUI to the Arduino for finger control

## Repository Contents

- `gui.py`: main Tkinter application for device connection, guided data collection, training, and live inference, with a real-time EMG plot
- `Motor_Execute.ino`: firmware that receives predicted gestures from `gui.py` and drives the servos in real time
- `gesture_tests.ino`: firmware that cycles the hand through the gesture set, for testing actuation independently of the classifier
- `fsr_circuit.ino`: proof-of-concept FSR reader (Serial and LCD output); not yet integrated into the gesture pipeline
- `data/`: recorded sEMG sessions (CSV)
- `requirements.txt`: Python dependencies

## Getting Started

### 1. Install Dependencies

Requires Python 3.9 or later. Clone the repository and install the requirements:

```bash
git clone https://github.com/skyebel/TNT_sEMG_Prosthesis_Project_2025-2026.git
cd TNT_sEMG_Prosthesis_Project_2025-2026
pip install -r requirements.txt
```

The default `requirements.txt` installs a CUDA 11.8 build of PyTorch. For a different CUDA version or a CPU-only build, substitute the appropriate index URL. An NVIDIA GPU is optional but speeds up training.

### 2. Setup Hardware

In order to actually control a hand you must have a MindRove armband, a PCA9685, 5 servos, and an Arduino. Once the Arduino is connected to your computer, install the `Adafruit_PWMServoDriver` library in the Arduino IDE and flash one of the following:

- `gesture_tests.ino` for initial testing of the hand on its own
- `Motor_Execute.ino` for live gesture execution from `gui.py`

### 3. Connect Hardware

Make sure the armband is powered on and the Arduino is connected. While `gui.py` is running, the Arduino IDE's Serial Monitor should be closed so it does not hold the port.

### 4. Run the System

From the repository root, run the main script:

```bash
python gui.py
```

This will start the GUI for data collection, training, and inference. Follow the on-screen instructions:

1. **Connect Device** to start a session with the armband.
2. **Start Sequence** to run guided data collection, saving labeled samples to `collected_emg_data.csv`.
3. **Train** to filter, window, and train the BiLSTM. Weights (`bilstm_vader.pt`) and scaler parameters (`scaler_mean.npy`, `scaler_scale.npy`) are saved for reuse.
4. **Live Predict** to run continuous inference and control the prosthetic hand.

## References

Aviles-Mendoza, Karla, et al. "A 3D Printed, Bionic Hand Powered by EMG Signals and Controlled by an Online Neural Network." *Biomimetics*, vol. 8, no. 2, 2023, p. 255. https://doi.org/10.3390/biomimetics8020255.

Karnam, Naveen Kumar, et al. "EMGHandNet: A Hybrid CNN and Bi-LSTM Architecture for Hand Activity Classification Using Surface EMG Signals." *Biomedical Signal Processing and Control*, vol. 42, no. 1, 2022, https://doi.org/10.1016/j.bbe.2022.02.005.
