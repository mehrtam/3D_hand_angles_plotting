🖐️ PROVE: 3D Hand Angles Plotting — Right Index Finger Kinematics
This project computes, analyzes, and visualizes the 3D joint angles of the right index finger derived from motion-capture typing data. By extracting MCP abduction, MCP/PIP/DIP flexion, and angular velocity, PROVE (Project for Verification of Kinematic Events) analyzes biomechanical typing patterns to reveal motor intent and keypress identity.

🧠 Motivation: Decoding Biomechanical Signatures
Typing generates distinct biomechanical signatures in finger movement. This project investigates whether fine-grained finger-joint kinematics can reliably predict which key is being pressed, opening doors for advanced Human-Computer Interaction (HCI) applications:

👁️+✋ Gaze-and-Hand Predictive Typing: Enhancing predictive text models by incorporating sub-conscious finger movements.

🧠 Motor-Intention Decoding: Translating pre-press movement into input for control or accessibility.

🕶️ VR/AR Natural Text Entry: Developing robust, intuitive input systems for immersive environments.

🔬 Biomechanics & HCI Research: Advancing the fundamental understanding of human motor control during interaction.

✨ Features
Kinematic Extraction: Computation of right-index-finger MCP abduction and MCP/PIP/DIP flexion.

Dynamic Analysis: Calculation of angular velocity (ω) using angle unwrapping for continuous data.

Event Sampling: Precise sampling of angles and velocities at per-keypress events.

3D Visualization: Plotting of kinematic clusters for y,h,n,u,j,m keys to demonstrate key separation.

High-Throughput Processing: Parallel CSV processing and support for the QTM motion-capture marker format.

📐 Mathematical Methodology
The project relies on standard vector geometry to calculate joint angles relative to the palm plane.

Key Formulas

Kinematic Measure	Description	Formula
Palm Plane Normal ( 
n
^
 )	Unit vector perpendicular to the palm's surface.	
n
^
 = 
∥…∥
(p 
2
​	
 −p 
1
​	
 )×(p 
3
​	
 −p 
1
​	
 )
​	
 
MCP Abduction Angle	Signed angle between the proximal phalanx projection and a reference vector on the palm plane.	
θ 
abd
​	
 =sign(( 
v

  
ref
​	
 × 
v

  
proj
​	
 )⋅ 
n
^
 )⋅θ
PIP / DIP Flexion	Angle between adjacent bone segment vectors ( 
v

  
1
​	
 , 
v

  
2
​	
 ).	
θ 
flex
​	
 =arccos( 
∥ 
v

  
1
​	
 ∥∥ 
v

  
2
​	
 ∥
v

  
1
​	
 ⋅ 
v

  
2
​	
 
​	
 )
Angular Velocity (ω)	Rate of change of the unwrapped joint angle.	
ω= 
dt
d(θ 
unwrap
​	
 )
​	
 
📊 Visualizations
The output focuses on 3D clustering and dynamic phase space analysis to reveal distinct typing patterns.

Plot Type	Axes/Dimensions	Purpose
3D MCP Space	Abduction × Flexion × Velocity	Visualize the complete dynamic motion space of the MCP joint for each key.
3D Flexion Clusters	MCP vs PIP vs DIP	Analyze the coordination and inter-dependency between the three main finger joints.
Flexion–Velocity Grid	Dynamic Phase Plots	Show the rate of change of flexion against the angle itself.
📂 Input Data Format
The system expects motion-capture data structured in CSV files, including:

XYZ Coordinates for all index finger and palm markers.

TimeStamp (Frame time).

Pressed_Letter (The character associated with the event).

KeyPressFlag (A binary marker for keystroke contact).

⚙️ Processing Pipeline
Data Loading: Ingest QTM motion-capture data.

Vector Calculation: Compute segment vectors and the Palm Plane Normal ( 
n
^
 ).

Angle Calculation: Determine raw MCP, PIP, and DIP joint angles.

Signal Processing: Apply angle unwrapping and conversion to degrees.

Velocity Calculation: Compute angular velocity (ω).

Event Sampling: Sample angle/velocity data precisely at keystrokes.

Visualization: Generate 3D kinematic cluster plots.

▶️ Run Instructions
To run the analysis script:

Bash
# Ensure dependencies are installed
pip install numpy pandas matplotlib seaborn concurrent.futures

# Execute the main script
python PROVE.py
Dependencies

numpy

pandas

matplotlib

seaborn

concurrent.futures

📎 Author
Fateme Eslami AI & Motion Interaction Research University of Birmingham
