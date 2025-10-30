# 🖐️ 3D Hand Angles Plotting — Right Index Finger Kinematics

This project computes and visualizes **3D right-index-finger joint angles** from motion-capture typing data.  
It extracts **MCP abduction**, **MCP/PIP/DIP flexion**, and **angular velocity**, then plots 3D kinematic clusters per keypress to analyze biomechanical typing patterns for predictive text and motor-intention research.

---

## 🧠 Motivation

Typing generates **distinct biomechanical signatures**.  
This project studies whether finger-joint motion can reveal **which key is pressed** — a step toward:

- 👁️+✋ Gaze-and-hand predictive typing  
- 🧠 Human motor-intention decoding  
- 🕶️ VR/AR natural text entry  
- 🧵 Biomechanics + HCI research  

---

## 📐 Mathematical Formulation

### ✅ 3D Joint Vector
$$
\vec{v} = \mathbf{p}_2 - \mathbf{p}_1
$$

### ✅ Palm Plane Normal
$$
\mathbf{n} =
\frac{
(\mathbf{p}_2 - \mathbf{p}_1) \times (\mathbf{p}_3 - \mathbf{p}_1)
}{
\lVert (\mathbf{p}_2 - \mathbf{p}_1) \times (\mathbf{p}_3 - \mathbf{p}_1) \rVert
}
$$

### ✅ Projection onto Palm Plane
$$
\vec{v}_{proj}
=
\vec{v}
-
\left( (\vec{v} \cdot \hat{n}) \right)\hat{n}
$$

### ✅ Angle Between Vectors
$$
\theta =
\arccos\left(
\frac{
\vec{v}_1 \cdot \vec{v}_2
}{
\lVert \vec{v}_1 \rVert \, \lVert \vec{v}_2 \rVert
}
\right)
$$

### ✅ MCP Abduction Angle (Signed)
$$
\theta_{\text{abd}}
=
\text{sign}\left((\vec{v}_{mcpBar} \times \vec{v}_{proj}) \cdot \hat{n}\right)
\cdot \theta
$$

---

### ✅ MCP / PIP / DIP Flexion

**General MCP flexion**
$$
\theta_{\text{flexion}}
=
\frac{\pi}{2}
-
\arccos\left(
\frac{
\vec{v}_{segment} \cdot \hat{n}
}{
\lVert \vec{v}_{segment} \rVert
}
\right)
$$

**PIP flexion**
$$
\theta_{\text{PIP}}
=
\arccos\left(
\frac{
\vec{v}_{MCP \rightarrow PIP}
\cdot
\vec{v}_{PIP \rightarrow DIP}
}{
\lVert \vec{v}_{MCP \rightarrow PIP} \rVert
\;
\lVert \vec{v}_{PIP \rightarrow DIP} \rVert
}
\right)
$$

### ✅ DIP Flexion

$$
\theta_{\text{DIP}}
=
\arccos\left(
\frac{
\vec{v}_{PIP \rightarrow DIP}
\cdot
\vec{v}_{DIP \rightarrow TIP}
}{
\lVert \vec{v}_{PIP \rightarrow DIP} \rVert
\;
\lVert \vec{v}_{DIP \rightarrow TIP} \rVert
}
\right)
$$


---

### ✅ Angle Unwrapping

$$
\theta_{\text{unwrap}} = \text{unwrap}(\theta)
$$

### ✅ Angular Velocity (deg/s)

$$
\omega = \frac{d(\theta_{unwrap})}{dt}
$$

---

## ✨ Features

- Full right-hand index finger kinematic extraction  
- MCP **abduction**, MCP/PIP/DIP **flexion**  
- NaN-safe processing & angle unwrapping  
- Per-keypress event-based sampling  
- 3D biomechanical cluster plots  
- Parallel batch processing for CSV datasets  
- Designed for **QTM motion-capture marker format**

Right-index keys analyzed:

y h n u j m

---

## 📊 Visualizations

| Plot | Description |
|------|-------------|
3D MCP space | Abduction × Flexion × Velocity  
3D Flexion clusters | MCP vs PIP vs DIP  
Flexion–Velocity Grid | Joint angle dynamics  
Keystroke separation | Character-wise clusters  

---

## 📂 Input Format

CSV files with:

- Finger marker XYZ positions  
- Palm markers  
- `Pressed_Letter`  
- `KeyPressFlag`  
- `TimeStamp`  

---

## 🧵 Processing Pipeline

1. Load QTM marker data  
2. Compute vectors & palm plane  
3. Compute MCP/PIP/DIP angles  
4. Unwrap + convert to degrees  
5. Compute angular velocity  
6. Extract frames at key events  
7. Plot 3D kinematic clusters  

---

## ⚙️ Dependencies

numpy
pandas
matplotlib
seaborn
concurrent.futures

---

## ▶️ Run

python PROVE.py
