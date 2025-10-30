# 🖐️ 3D Hand Angles Plotting — Right Index Finger Kinematics

This project extracts and visualizes **3D right-index-finger joint movements** during typing using motion-capture data. It computes joint angles and angular velocities, then visualizes how different keystrokes create distinct motion clusters.

This work supports research in **motion-based text entry**, **biomechanics**, and **predictive typing for AR/VR environments**.

---

## 🎯 Purpose

Typing generates unique **biomechanical patterns**.  
This code explores whether finger joint motion alone can reveal **which key was pressed** — a step toward:

- Motion-based predictive typing (VR/AR, XR)
- Assistive & hands-free communication systems
- Gaze-plus-hand multimodal typing
- Motor intention decoding
- HCI / biomechanics research

---

## ✅ Key Features

- Right-hand index-finger kinematic extraction
- Computes:
  - MCP abduction
  - MCP / PIP / DIP flexion
  - Angular velocity
- Detects and samples frames at keypress time
- Handles NaNs, unwrapping, and velocity stability
- Parallel batch processing for many CSV files
- Built for **Qualisys/QTM motion-capture data**
- Visualizes 3D clusters for each key

Supported right-index keys:

`y, h, n, u, j, m`

---

## 📊 Visualizations

| Plot Type | Description |
|----------|-------------|
3D MCP Angle Space | Abduction vs Flexion vs Velocity cluster plot  
3D Flexion Clusters | MCP vs PIP vs DIP joint angle mapping  
Angle–Velocity Space | Displays dynamic movement profiles  
Key-Based Clusters | Shows how finger motion separates keys  

---

## 🧾 Input Requirements

Your CSV files should include:

- 3D marker positions for index finger joints  
- Palm reference markers  
- Timestamps  
- Keypress flag and pressed character  

Example columns:

QTMdc_R_Index_Prox_GLOBAL_X, Y, Z
QTMdc_R_Index_Inter_GLOBAL_X, Y, Z
QTMdc_R_Index_Distal_GLOBAL_X, Y, Z
QTMdc_R_Index_End_GLOBAL_X, Y, Z
Pressed_Letter
KeyPressFlag
TimeStamp

---

## 🧵 Processing Pipeline

1. Load .csv motion-capture files
2. Extract finger and palm marker positions
3. Compute MCP / PIP / DIP joint angles
4. Compute angular velocities
5. Detect keypress frames
6. Extract kinematic values at event frames
7. Plot 2D & 3D cluster visualizations

---

## ⚙️ Dependencies

numpy
pandas
matplotlib
seaborn
concurrent.futures

---

## ▶️ Run the Script

```bash
python PROVE.py
Make sure to update the input data path inside the script.
📎 Project Goal
To evaluate whether hand biomechanics alone can predict typing intention, enabling natural, hands-in-air text entry for:
AR/VR typing interfaces
Neural & kinematic decoding systems
Accessibility & assistive input devices
Cognitive-motor research
👩‍💻 Author
Fateme Eslami
AI & Human-Motion Interaction Research
University of Birmingham
