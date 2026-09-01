# Audio Packet Loss Concealment (PLC)

This repository contains the dataset for analyzing network packet loss and repairing damaged audio files. It is designed to be easily imported and run directly inside Google Colab.

## What This Task Does
* **Network Analysis:** Reads network trace files (`.txt`) to calculate packet loss rates, burst lengths, and markov transitions.
* **Digital Signal Processing:** Uses Linear Predictive Coding (LPC) to analyze clean audio history and mathematically guess missing audio samples.
* **Audio Repair:** Patches the silent gaps in degraded `.wav` files to restore audio quality.

## Folder Structure
* **`Audio_Project_Data_VUT/lossy/`**: Contains the degraded audio files (`.wav`).
* **`Audio_Project_Data_VUT/traces/`**: Contains the network checklists (`.txt`). A `0` means the packet arrived safely, and a `1` means the packet was lost.

