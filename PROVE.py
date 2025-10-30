# ============================================
#  FULL VECTORIZED KINEMATIC ANALYSIS FOR RIGHT INDEX FINGER
# ============================================
# - Computes Flexion, Abduction, and Velocity for the R Index finger.
# - Handles angle unwrapping, missing columns, and NaN-safe velocity.
# ============================================

import glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures
import time
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------
# CONFIGURATION
# -------------------------------
PATH_PATTERN = "/Users/fateme/Desktop/data/**/*.csv"
DEG = True
EPS = 1e-12
MAX_WORKERS = 8
RIGHT_INDEX_KEYS = ['y', 'h', 'n', 'u', 'j', 'm']

# -------------------------------
# MARKER MAPPING (Right Hand)
# -------------------------------
MARKER_MAP = {
    'I1': 'QTMdc_R_Index_Prox_GLOBAL',
    'I2': 'QTMdc_R_Index_Inter_GLOBAL',
    'I3': 'QTMdc_R_Index_Distal_GLOBAL',
    'I4': 'QTMdc_R_Index_End_GLOBAL',

    'M1': 'QTMdc_R_Middle_Prox_GLOBAL',
    'M2': 'QTMdc_R_Middle_Inter_GLOBAL',
    'M3': 'QTMdc_R_Middle_Distal_GLOBAL',
    'M4': 'QTMdc_R_Middle_End_GLOBAL',

    'R1': 'QTMdc_R_Ring_Prox_GLOBAL',
    'R2': 'QTMdc_R_Ring_Inter_GLOBAL',
    'R3': 'QTMdc_R_Ring_Distal_GLOBAL',
    'R4': 'QTMdc_R_Ring_End_GLOBAL',

    'L1': 'QTMdc_R_Little_Prox_GLOBAL',
    'L2': 'QTMdc_R_Little_Inter_GLOBAL',
    'L3': 'QTMdc_R_Little_Distal_GLOBAL',
    'L4': 'QTMdc_R_Little_End_GLOBAL',

    'Cin': 'QTMdc_R_Cin',
    'Cout': 'QTMdc_R_Cout',
}

# -------------------------------
# GEOMETRY HELPERS
# -------------------------------
def midpoint(p1, p2):
    return (p1 + p2) / 2

def line_vector(p1, p2):
    return p2 - p1

def angle_between_vectors(v1, v2):
    epsilon = 1e-8
    v1_u = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + epsilon)
    v2_u = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + epsilon)
    cos_theta = np.sum(v1_u * v2_u, axis=1)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

def project_onto_plane(v, n):
    dot_product = np.sum(v * n, axis=1, keepdims=True)
    return v - dot_product * n

def plane_normal(p1, p2, p3):
    n = np.cross(p2 - p1, p3 - p1)
    epsilon = 1e-12
    norm_n = np.linalg.norm(n, axis=1, keepdims=True)
    return n / (norm_n + epsilon)

# -------------------------------
# ANGLE COMPUTATIONS
# -------------------------------
def compute_all_mcp_abduction_angles(L1, R1, M1, I1, L4, R4, M4, I4, Cin, Cout, hand_side='R'):
    palm_mid = midpoint(Cin, Cout)
    mcp_bar_vec = line_vector(I1, L1)
    palm_normal_vec = plane_normal(I1, L1, palm_mid)

    fingers = {'Little': (L1, L4), 'Ring': (R1, R4), 'Middle': (M1, M4), 'Index': (I1, I4)}
    results = {}

    for label, (mcp, tip) in fingers.items():
        v_mcp_tip = line_vector(mcp, tip)
        v_proj = project_onto_plane(v_mcp_tip, palm_normal_vec)
        angle = angle_between_vectors(v_proj, mcp_bar_vec)

        cross_prod = np.cross(mcp_bar_vec, v_proj)
        sign = np.sign(np.sum(cross_prod * palm_normal_vec, axis=1))
        sign[sign == 0] = 1  # avoid 0-sign degeneracy

        results[label] = angle * sign if hand_side == 'R' else angle * -sign
    return results

def compute_mcp_flexion_angles(L1, L2, R1, R2, M1, M2, I1, I2, Cin, Cout):
    palm_mid = midpoint(Cin, Cout)
    palm_normal_vec = plane_normal(I1, L1, palm_mid)
    fingers = {'Little': (L1, L2), 'Ring': (R1, R2), 'Middle': (M1, M2), 'Index': (I1, I2)}
    results = {}
    for label, (mcp, pip) in fingers.items():
        v = line_vector(mcp, pip)
        angle_with_normal = angle_between_vectors(v, palm_normal_vec)
        results[label] = np.pi/2 - angle_with_normal
    return results

def compute_all_finger_segment_angles(L1, L2, L3, L4, R1, R2, R3, R4, M1, M2, M3, M4, I1, I2, I3, I4):
    fingers = {'Little': (L1, L2, L3, L4), 'Ring': (R1, R2, R3, R4),
               'Middle': (M1, M2, M3, M4), 'Index': (I1, I2, I3, I4)}
    results = {}
    for label, (mcp, pip, dip, tip) in fingers.items():
        v1, v2, v3 = line_vector(mcp, pip), line_vector(pip, dip), line_vector(dip, tip)
        results[label] = {
            'PIP_angle': angle_between_vectors(v1, v2),
            'DIP_angle': angle_between_vectors(v2, v3)
        }
    return results

# -------------------------------
# ANGLE VELOCITY
# -------------------------------
def calculate_angle_velocity(angles_df, df_full):
    if 'TimeStamp' not in df_full.columns:
        print("Error: 'TimeStamp' column missing for velocity calculation.")
        return pd.DataFrame()

    t = df_full['TimeStamp'].to_numpy()
    velocity_features = {}

    for col in angles_df.columns:
        a_deg = angles_df[col].to_numpy()
        mask = ~np.isnan(a_deg)
        if mask.sum() < 3:
            velocity_features[f'VELOCITY_{col}'] = np.full_like(a_deg, np.nan)
            continue

        a_rad = np.radians(a_deg[mask])
        a_rad_unwrapped = np.unwrap(a_rad)
        v = np.gradient(a_rad_unwrapped, t[mask])  # rad/s
        v_full = np.full_like(a_deg, np.nan)
        v_full[mask] = np.degrees(v)
        velocity_features[f'VELOCITY_{col}'] = v_full

    return pd.DataFrame(velocity_features, index=angles_df.index)

# -------------------------------
# DATA HANDLING
# -------------------------------
def get_3d_cols(prefix):
    return [f'{prefix}_X', f'{prefix}_Y', f'{prefix}_Z']

def load_marker_coords(df, marker_name):
    cols = get_3d_cols(marker_name)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[WARN] Missing cols for {marker_name}: {missing}")
        return np.full((len(df), 3), np.nan)
    return df[cols].replace([np.inf, -np.inf], np.nan).to_numpy()

def calculate_kinematics_for_file(fpath):
    df = pd.read_csv(fpath)
    df['Pressed_Letter'] = df['Pressed_Letter'].astype(str).str.lower()

    if 'KeyPressFlag' not in df.columns or 'Pressed_Letter' not in df.columns or 'TimeStamp' not in df.columns:
        return []

    coords = {k: load_marker_coords(df, v) for k, v in MARKER_MAP.items()}

    L1, L2, L3, L4 = coords['L1'], coords['L2'], coords['L3'], coords['L4']
    R1, R2, R3, R4 = coords['R1'], coords['R2'], coords['R3'], coords['R4']
    M1, M2, M3, M4 = coords['M1'], coords['M2'], coords['M3'], coords['M4']
    I1, I2, I3, I4 = coords['I1'], coords['I2'], coords['I3'], coords['I4']
    Cin, Cout = coords['Cin'], coords['Cout']

    mcp_abd = compute_all_mcp_abduction_angles(L1, R1, M1, I1, L4, R4, M4, I4, Cin, Cout)
    mcp_flex = compute_mcp_flexion_angles(L1, L2, R1, R2, M1, M2, I1, I2, Cin, Cout)
    seg_flex = compute_all_finger_segment_angles(L1, L2, L3, L4, R1, R2, R3, R4, M1, M2, M3, M4, I1, I2, I3, I4)

    final_angles = {}
    for f, a in mcp_abd.items(): final_angles[f'{f}_MCP_Abduction'] = a
    for f, a in mcp_flex.items(): final_angles[f'{f}_MCP_Flexion'] = a
    for f, ang in seg_flex.items():
        final_angles[f'{f}_PIP_Flexion'] = ang['PIP_angle']
        final_angles[f'{f}_DIP_Flexion'] = ang['DIP_angle']

    angles_df = pd.DataFrame(final_angles, index=df.index)
    if DEG:
        angles_df = angles_df.apply(np.degrees)
        angles_df = angles_df.clip(lower=-180, upper=180)

    velocity_df = calculate_angle_velocity(angles_df, df)
    kinematics_df = angles_df.join(velocity_df)

    keystroke_df = df[df['KeyPressFlag'] == 1].copy()
    out = []

    for eidx in keystroke_df.index:
        key = df.loc[eidx, 'Pressed_Letter']
        t = df.loc[eidx, 'TimeStamp']
        if key not in RIGHT_INDEX_KEYS:
            continue

        frame_kin = kinematics_df.loc[eidx].to_dict()
        for col, val in frame_kin.items():
            if col.startswith('VELOCITY_'):
                parts = col.split('_')
                finger, joint, angle_type = parts[1], parts[2], parts[3]
                measure = f'VELOCITY_{angle_type}'
            else:
                finger, joint, angle_type = col.split('_')
                measure = angle_type

            if finger != 'Index':
                continue

            out.append([key, 'R', eidx, t, f'{joint}_{measure}', val])
    return out

# -------------------------------
# PLOTTING
# -------------------------------
def plot_3d_mcp_kinematics(angles_df, target_keys):
    x_measure, y_measure, z_measure = 'MCP_Abduction', 'MCP_Flexion', 'MCP_VELOCITY_Abduction'
    df_filtered = angles_df[
        (angles_df['Pressed_Letter'].isin(target_keys)) &
        (angles_df['Joint_Angle_Type'].isin([x_measure, y_measure, z_measure]))
    ].copy()

    if df_filtered.empty:
        print("3D MCP Plot skipped: missing data.")
        return

    VIS_DF = df_filtered.pivot_table(
        index=['Pressed_Letter', 'Frame_Index'],
        columns='Joint_Angle_Type',
        values='Angle_Value'
    ).reset_index().dropna()

    VIS_DF = VIS_DF.rename(columns={
        x_measure: 'Abduction_Angle',
        y_measure: 'Flexion_Angle',
        z_measure: 'Abduction_Velocity'
    })

    keys = sorted(VIS_DF['Pressed_Letter'].unique())
    key_map = {k: i for i, k in enumerate(keys)}
    VIS_DF['Key_Numeric'] = VIS_DF['Pressed_Letter'].map(key_map)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        VIS_DF['Abduction_Angle'], VIS_DF['Flexion_Angle'],
        VIS_DF['Abduction_Velocity'], c=VIS_DF['Key_Numeric'],
        cmap='viridis', s=50, alpha=0.6
    )
    ax.set_title("3D MCP Keystroke Space (R Index)", fontsize=16)
    ax.set_xlabel("MCP Abduction Angle (°)")
    ax.set_ylabel("MCP Flexion Angle (°)")
    ax.set_zlabel("MCP Abduction Velocity (°/s)")
    cbar = fig.colorbar(scatter, ticks=list(key_map.values()))
    cbar.ax.set_yticklabels(keys)
    cbar.set_label('Pressed Character', rotation=270, labelpad=20)
    plt.show()

def plot_flexion_velocity_grid(angles_df, target_keys):
    flexion_measures = ['MCP_Flexion', 'PIP_Flexion', 'DIP_Flexion']
    velocity_measures = ['MCP_VELOCITY_Flexion', 'PIP_VELOCITY_Flexion', 'DIP_VELOCITY_Flexion']

    df_flexion = angles_df[
        (angles_df['Pressed_Letter'].isin(target_keys)) &
        (angles_df['Joint_Angle_Type'].isin(flexion_measures))
    ].rename(columns={'Angle_Value': 'Angle'})

    df_velocity = angles_df[
        (angles_df['Pressed_Letter'].isin(target_keys)) &
        (angles_df['Joint_Angle_Type'].isin(velocity_measures))
    ].rename(columns={'Angle_Value': 'Velocity'})

    df_flexion['Joint'] = df_flexion['Joint_Angle_Type'].str.replace('_Flexion', '')
    df_velocity['Joint'] = df_velocity['Joint_Angle_Type'].str.replace('_VELOCITY_Flexion', '')

    df_merged = pd.merge(
        df_flexion[['Pressed_Letter', 'Frame_Index', 'Joint', 'Angle']],
        df_velocity[['Pressed_Letter', 'Frame_Index', 'Joint', 'Velocity']],
        on=['Pressed_Letter', 'Frame_Index', 'Joint'],
        how='inner'
    ).dropna()

    if df_merged.empty:
        print("2D Grid plot skipped: no data.")
        return

    g = sns.FacetGrid(df_merged, col="Joint", hue="Pressed_Letter", palette="viridis",
                      col_wrap=3, height=5)
    g.map(sns.scatterplot, "Angle", "Velocity", alpha=0.5, s=20)
    g.add_legend(title="Keystroke")
    g.set_axis_labels("Flexion Angle (°)", "Flexion Velocity (°/s)")
    g.fig.suptitle("Flexion Angle vs Velocity (R Index)", y=1.02, fontsize=16)
    plt.show()
def plot_3d_flexion_clusters(angles_df, target_keys):
    """
    3D cluster plot: X=MCP_Flexion, Y=PIP_Flexion, Z=DIP_Flexion.
    Each point represents a keypress, colored by the pressed character.
    """

    # Required joint measures
    joints = ['MCP_Flexion', 'PIP_Flexion', 'DIP_Flexion']

    # Filter only Index finger flexion measures
    df_filtered = angles_df[
        (angles_df['Pressed_Letter'].isin(target_keys)) &
        (angles_df['Joint_Angle_Type'].isin(joints))
    ].copy()

    if df_filtered.empty:
        print("3D Flexion Cluster Plot skipped: Missing flexion data.")
        return

    # Pivot to wide format: each row = frame, each col = joint angle
    VIS_DF = df_filtered.pivot_table(
        index=['Pressed_Letter', 'Frame_Index'],
        columns='Joint_Angle_Type',
        values='Angle_Value'
    ).reset_index().dropna(subset=joints)

    if VIS_DF.empty:
        print("3D Flexion Cluster Plot skipped: Empty after pivot/dropna.")
        return

    # Rename columns for clarity
    VIS_DF = VIS_DF.rename(columns={
        'MCP_Flexion': 'MCP',
        'PIP_Flexion': 'PIP',
        'DIP_Flexion': 'DIP'
    })

    # Assign numeric colors for each key
    keys = sorted(VIS_DF['Pressed_Letter'].unique())
    key_map = {k: i for i, k in enumerate(keys)}
    VIS_DF['Key_Numeric'] = VIS_DF['Pressed_Letter'].map(key_map)

    # Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        VIS_DF['MCP'], VIS_DF['PIP'], VIS_DF['DIP'],
        c=VIS_DF['Key_Numeric'], cmap='plasma', s=60, alpha=0.7
    )

    # Labels and aesthetics
    ax.set_title("3D Flexion Angle Clusters (R Index)", fontsize=16)
    ax.set_xlabel("MCP Flexion (°)", fontsize=12)
    ax.set_ylabel("PIP Flexion (°)", fontsize=12)
    ax.set_zlabel("DIP Flexion (°)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)

    cbar = fig.colorbar(scatter, ticks=list(key_map.values()))
    cbar.ax.set_yticklabels(keys)
    cbar.set_label('Pressed Character', rotation=270, labelpad=20)

    plt.tight_layout()
    plt.show()
def plot_3d_mcp_abd_flex_pip_clusters(angles_df, target_keys):
    """
    3D cluster plot based on MCP Abduction, MCP Flexion, and PIP Flexion.
    Each point = keystroke, colored by pressed character.
    """

    measures = ['MCP_Abduction', 'MCP_Flexion', 'PIP_Flexion']

    # Filter relevant data
    df_filtered = angles_df[
        (angles_df['Pressed_Letter'].isin(target_keys)) &
        (angles_df['Joint_Angle_Type'].isin(measures))
    ].copy()

    if df_filtered.empty:
        print("3D MCP-Abd/Flex/PIP plot skipped: Missing data.")
        return

    # Pivot to wide format: each row = frame, columns = measures
    VIS_DF = df_filtered.pivot_table(
        index=['Pressed_Letter', 'Frame_Index'],
        columns='Joint_Angle_Type',
        values='Angle_Value'
    ).reset_index().dropna(subset=measures)

    if VIS_DF.empty:
        print("3D MCP-Abd/Flex/PIP plot skipped: Empty after pivot/dropna.")
        return

    # Rename for clarity
    VIS_DF = VIS_DF.rename(columns={
        'MCP_Abduction': 'MCP_Abd',
        'MCP_Flexion': 'MCP_Flex',
        'PIP_Flexion': 'PIP_Flex'
    })

    # Assign numeric colors per character
    keys = sorted(VIS_DF['Pressed_Letter'].unique())
    key_map = {k: i for i, k in enumerate(keys)}
    VIS_DF['Key_Numeric'] = VIS_DF['Pressed_Letter'].map(key_map)

    # --- 3D Scatter ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        VIS_DF['MCP_Abd'],
        VIS_DF['MCP_Flex'],
        VIS_DF['PIP_Flex'],
        c=VIS_DF['Key_Numeric'],
        cmap='plasma',
        s=60,
        alpha=0.7
    )

    ax.set_title("3D MCP–PIP Angle Space (R Index)", fontsize=16)
    ax.set_xlabel("MCP Abduction (°)", fontsize=12)
    ax.set_ylabel("MCP Flexion (°)", fontsize=12)
    ax.set_zlabel("PIP Flexion (°)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Add colorbar with key labels
    cbar = fig.colorbar(scatter, ticks=list(key_map.values()))
    cbar.ax.set_yticklabels(keys)
    cbar.set_label('Pressed Character', rotation=270, labelpad=20)

    plt.tight_layout()
    plt.show()

# -------------------------------
# MAIN
# -------------------------------
if __name__ == '__main__':
    start = time.time()
    OUTPUT_HEADERS = ['Pressed_Letter', 'Hand', 'Frame_Index', 'TimeStamp',
                      'Joint_Angle_Type', 'Angle_Value']

    files = glob.glob(PATH_PATTERN, recursive=True)
    print(f"Found {len(files)} CSV files.")

    all_rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        print(f"Processing with {MAX_WORKERS} workers...")
        for rows in ex.map(calculate_kinematics_for_file, files):
            all_rows.extend(rows)

    angles_df = pd.DataFrame(all_rows, columns=OUTPUT_HEADERS)
    print("=" * 50)
    print(f"Kinematic extraction complete in {time.time() - start:.2f}s")
    print(f"Total Rows: {angles_df.shape[0]}")
    print("=" * 50)

    # print("\nGenerating 3D MCP Plot...")
    # plot_3d_mcp_kinematics(angles_df, RIGHT_INDEX_KEYS)

    # print("\nenerating 2D Flexion-Velocity Grid...")
    # plot_flexion_velocity_grid(angles_df, RIGHT_INDEX_KEYS)

    # print("\nangles_df preview:")
    # print(angles_df.head(10))
    # print("\nGenerating 3D MCP/PIP/DIP Flexion Cluster Plot...")
    # plot_3d_flexion_clusters(angles_df, RIGHT_INDEX_KEYS)
    print("\Generating 3D MCP-Abduction/Flexion/PIP-Flexion Cluster Plot...")
    plot_3d_mcp_abd_flex_pip_clusters(angles_df, RIGHT_INDEX_KEYS)


