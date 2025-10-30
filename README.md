# 🖐️ 3D Hand Angles Plotting — Right Index Finger Kinematics

This project computes and visualizes **3D right-index-finger joint angles** from motion-capture typing data.  
It extracts **MCP abduction**, **MCP/PIP/DIP flexion**, and **angular velocity**, then plots 3D kinematic clusters per keypress to analyze biomechanical typing patterns for predictive text and motor-intention research.

---

## 🧠 Motivation

Typing generates **distinct biomechanical signatures**.  
This project explores whether finger-joint motion can reveal **which key is pressed**, enabling:

- 👁️+✋ Gaze-and-hand predictive typing  
- 🧠 Motor-intention decoding  
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

**DIP flexion**

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

### ✅ Angle Unwrapping

$$
\theta_{\text{unwrap}} = \text{unwrap}(\theta)
$$

### ✅ Angular Velocity (deg/s)

$$
\omega = \frac{d(\theta_{\text{unwrap}})}{dt}
$$

---

## ✨ Features

- Right-hand index finger kinematic extraction  
- MCP **abduction**, MCP/PIP/DIP **flexion**  
- Angle unwrapping & NaN-safe velocity  
- Per-keypress event sampling  
- 3D kinematic cluster visualization  
- Parallel CSV processing  
- Supports **QTM motion-capture marker format**

**Keys analyzed (right index finger):**
y h n u j m

---

## 📊 Visualizations

| Plot | Description |
|------|-------------|
3D MCP space | Abduction × Flexion × Velocity  
3D Flexion clusters | MCP vs PIP vs DIP  
Flexion–Velocity Grid | Dynamic joint motion  
Key separation | Character-wise kinematics  

---

## 📂 Input Format

CSV files with:

- Finger marker XYZ coordinates  
- Palm markers  
- `Pressed_Letter`  
- `KeyPressFlag`  
- `TimeStamp`  

---

## 🧵 Processing Pipeline

1. Load QTM motion-capture data  
2. Compute palm plane + joint vectors  
3. Calculate MCP/PIP/DIP angles  
4. Unwrap angles → convert to degrees  
5. Compute angular velocity  
6. Sample angles at keystrokes  
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
📎 Author
Fateme Eslami — AI & Motion Interaction Research
University of Birmingham
