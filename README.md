#  3D Hand Angles Plotting — Right Index Finger Kinematics

Computes and visualizes **3D right-index-finger joint angles** from motion-capture typing data.  
Extracts **MCP abduction**, **MCP/PIP/DIP flexion**, and **angular velocity**, then plots 3D kinematic clusters per keypress.

---

##  Mathematical Formulation

###  3D Joint Vectors
Finger segment vectors are computed as:

\[
\vec{v} = \mathbf{p}_2 - \mathbf{p}_1
\]

---

###  Palm Plane Normal

Palm plane normal derived from three 3D markers:

\[
\mathbf{n} = \frac{(\mathbf{p}_2 - \mathbf{p}_1) \times (\mathbf{p}_3 - \mathbf{p}_1)}
{\left\lVert (\mathbf{p}_2 - \mathbf{p}_1) \times (\mathbf{p}_3 - \mathbf{p}_1) \right\rVert}
\]

---

###  Projection of Finger Vector onto Palm Plane

\[
\vec{v}_{proj} = \vec{v} - (\vec{v} \cdot \hat{n})\hat{n}
\]

---

###  Angle Between Two 3D vectors

\[
\theta = \arccos \left( 
\frac{\vec{v_1} \cdot \vec{v_2}}{\lVert \vec{v_1} \rVert \lVert \vec{v_2} \rVert}
\right)
\]

---

###  MCP Abduction Angle (Signed)

\[
\theta_{abd} = \operatorname{sign}((\vec{v_{mcpBar}} \times \vec{v_{proj}})\cdot \hat{n}) \cdot \theta
\]

---

###  MCP / PIP / DIP Flexion

\[
\theta_{flexion} = \frac{\pi}{2} - \arccos 
\left(
\frac{\vec{v_{segment}} \cdot \hat{n}}{\lVert \vec{v_{segment}} \rVert}
\right)
\]

For DIP/PIP:

\[
\theta_{PIP} = \arccos
\left(
\frac{\vec{v_{MCP→PIP}} \cdot \vec{v_{PIP→DIP}}}
{\lVert \vec{v_{MCP→PIP}} \rVert\lVert \vec{v_{PIP→DIP}} \rVert}
\right)
\]

\[
\theta_{DIP} = \arccos
\left(
\frac{\vec{v_{PIP→DIP}} \cdot \vec{v_{DIP→TIP}}}
{\lVert \vec{v_{PIP→DIP}} \rVert\lVert \vec{v_{DIP→TIP}} \rVert}
\right)
\]

---

### ✅ Angle Unwrapping

\[
\theta_{unwrap} = \operatorname{unwrap}(\theta)
\]

---

###  Angular Velocity (Degrees/sec)

\[
\omega = \frac{d\,(\theta_{unwrap})}{dt}
\]

Converted to degrees/sec after differentiation.

---

##  Why This Matters

This pipeline studies whether **finger kinematics uniquely encode typing intention**, enabling:

- Gaze-and-hand predictive typing
- Motor-intention decoding
- VR text entry research
- Neural/biomechanical HCI modeling

---

##  Plots

-  MCP angle space (abduction vs flexion vs velocity)
-  3D flexion cluster space (MCP / PIP / DIP)
-  Velocity vs angle scatter
-  MCP-Abduction / MCP-Flexion / PIP-Flexion 3D clusters

---

##  Run

```bash
python PROVE.py
