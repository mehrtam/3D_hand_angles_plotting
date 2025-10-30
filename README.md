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
\left\lVert (\mathbf{p}_2 - \mathbf{p}_1) \times (\mathbf{p}_3 - \mathbf{p}_1) \right\rVert
}
$$

### ✅ Projection onto Palm Plane
$$
\vec{v}_{proj}
=
\vec{v}
-
(\vec{v} \cdot \hat{n})\hat{n}
$$

### ✅ Angle Between Vectors
$$
\theta =
\arccos\left(
\frac{\vec{v_1} \cdot \vec{v_2}}
{\lVert\vec{v_1}\rVert \lVert\vec{v_2}\rVert}
\right)
$$

### ✅ MCP Abduction Angle (Signed)
$$
\theta_{abd}
=
\operatorname{sign}((\vec{v_{mcpBar}} \times \vec{v_{proj}})\cdot \hat{n})
\cdot
\theta
$$

### ✅ MCP / PIP / DIP Flexion
General flexion:

$$
\theta_{flexion}
=
\frac{\pi}{2}
-
\arccos
\left(
\frac{\vec{v_{segment}}\cdot \hat{n}}
{\|\vec{v_{segment}}\|}
\right)
$$

PIP/DIP angles:

$$
\theta_{PIP} =
\arccos \left(
\frac{\vec{v_{MCP\to PIP}} \cdot \vec{v_{PIP\to DIP}}}
{\lVert \vec{v_{MCP\to PIP}} \rVert \lVert \vec{v_{PIP\to DIP}} \rVert}
\right)
$$

$$
\theta_{DIP} =
\arccos \left(
\frac{\vec{v_{PIP\to DIP}} \cdot \vec{v_{DIP\to TIP}}}
{\lVert \vec{v_{PIP\to DIP}} \rVert \lVert \vec{v_{DIP\to TIP}} \rVert}
\right)
$$

### ✅ Angle Unwrapping
$$
\theta_{unwrap} = \operatorname{unwrap}(\theta)
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

```bash
python PROVE.py
