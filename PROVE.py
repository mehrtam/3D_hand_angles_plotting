"""
Right-Hand Index Finger Kinematic Analysis
==========================================

Extracts 3D joint kinematics (MCP abduction, MCP/PIP/DIP flexion, and angular
velocities) for the right index finger from Qualisys motion-capture CSVs, and
visualizes how distinct keystrokes form separate clusters in joint-angle space.

Geometric pipeline
------------------
  1. Load 3D marker positions for index, middle, ring, little fingers + palm.
  2. Build a palm-normal vector from the proximal-MCP triangle (I1, L1, palm midpoint).
  3. Project per-finger MCP-tip vectors onto the palm plane to compute *signed*
     MCP abduction angles, with sign determined by cross-product agreement
     with the palm normal.
  4. Compute MCP flexion as the complement of the angle between the proximal
     phalanx and the palm normal.
  5. Compute PIP and DIP flexion as the inter-segment angle between consecutive
     phalanx vectors.
  6. Differentiate unwrapped angles w.r.t. timestamps for angular velocity.
  7. Sample at keypress events (KeyPressFlag == 1) and plot 3D clusters in the
     joint-angle space, colored by pressed character.

Targets the QWERTY right-index keys: y, h, n, u, j, m.

Usage
-----
    python kinematics_analysis.py --data /path/to/csvs --plot mcp_pip_flex
    python kinematics_analysis.py --data /path/to/csvs --plot all
"""

import argparse
import concurrent.futures
import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEG = True
EPS = 1e-12
MAX_WORKERS = 8
RIGHT_INDEX_KEYS = ["y", "h", "n", "u", "j", "m"]

MARKER_MAP = {
    "I1": "QTMdc_R_Index_Prox_GLOBAL",
    "I2": "QTMdc_R_Index_Inter_GLOBAL",
    "I3": "QTMdc_R_Index_Distal_GLOBAL",
    "I4": "QTMdc_R_Index_End_GLOBAL",
    "M1": "QTMdc_R_Middle_Prox_GLOBAL",
    "M2": "QTMdc_R_Middle_Inter_GLOBAL",
    "M3": "QTMdc_R_Middle_Distal_GLOBAL",
    "M4": "QTMdc_R_Middle_End_GLOBAL",
    "R1": "QTMdc_R_Ring_Prox_GLOBAL",
    "R2": "QTMdc_R_Ring_Inter_GLOBAL",
    "R3": "QTMdc_R_Ring_Distal_GLOBAL",
    "R4": "QTMdc_R_Ring_End_GLOBAL",
    "L1": "QTMdc_R_Little_Prox_GLOBAL",
    "L2": "QTMdc_R_Little_Inter_GLOBAL",
    "L3": "QTMdc_R_Little_Distal_GLOBAL",
    "L4": "QTMdc_R_Little_End_GLOBAL",
    "Cin": "QTMdc_R_Cin",
    "Cout": "QTMdc_R_Cout",
}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def midpoint(p1, p2):
    return (p1 + p2) / 2


def line_vector(p1, p2):
    return p2 - p1


def angle_between_vectors(v1, v2, eps=1e-8):
    v1_u = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + eps)
    v2_u = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + eps)
    cos_theta = np.sum(v1_u * v2_u, axis=1)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))


def project_onto_plane(v, n):
    dot = np.sum(v * n, axis=1, keepdims=True)
    return v - dot * n


def plane_normal(p1, p2, p3, eps=EPS):
    n = np.cross(p2 - p1, p3 - p1)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + eps)


# ---------------------------------------------------------------------------
# Angle computations
# ---------------------------------------------------------------------------

def compute_all_mcp_abduction_angles(L1, R1, M1, I1, L4, R4, M4, I4, Cin, Cout, hand_side="R"):
    """
    Signed MCP abduction (deviation from the index–little MCP bar in the palm plane).
    Sign comes from cross-product agreement with the palm normal — positive sign
    indicates one side of the bar, negative the other.
    """
    palm_mid = midpoint(Cin, Cout)
    mcp_bar = line_vector(I1, L1)
    palm_n = plane_normal(I1, L1, palm_mid)

    fingers = {"Little": (L1, L4), "Ring": (R1, R4), "Middle": (M1, M4), "Index": (I1, I4)}
    out = {}
    for label, (mcp, tip) in fingers.items():
        v = line_vector(mcp, tip)
        v_proj = project_onto_plane(v, palm_n)
        angle = angle_between_vectors(v_proj, mcp_bar)

        sign = np.sign(np.sum(np.cross(mcp_bar, v_proj) * palm_n, axis=1))
        sign[sign == 0] = 1
        out[label] = angle * sign if hand_side == "R" else angle * -sign
    return out


def compute_mcp_flexion_angles(L1, L2, R1, R2, M1, M2, I1, I2, Cin, Cout):
    """MCP flexion = π/2 - angle between proximal phalanx and palm normal."""
    palm_mid = midpoint(Cin, Cout)
    palm_n = plane_normal(I1, L1, palm_mid)
    fingers = {"Little": (L1, L2), "Ring": (R1, R2), "Middle": (M1, M2), "Index": (I1, I2)}
    return {
        label: (np.pi / 2 - angle_between_vectors(line_vector(mcp, pip), palm_n))
        for label, (mcp, pip) in fingers.items()
    }


def compute_all_finger_segment_angles(L1, L2, L3, L4, R1, R2, R3, R4, M1, M2, M3, M4, I1, I2, I3, I4):
    """Inter-segment PIP and DIP flexion angles for each finger."""
    fingers = {
        "Little": (L1, L2, L3, L4),
        "Ring": (R1, R2, R3, R4),
        "Middle": (M1, M2, M3, M4),
        "Index": (I1, I2, I3, I4),
    }
    out = {}
    for label, (mcp, pip, dip, tip) in fingers.items():
        v1 = line_vector(mcp, pip)
        v2 = line_vector(pip, dip)
        v3 = line_vector(dip, tip)
        out[label] = {
            "PIP_angle": angle_between_vectors(v1, v2),
            "DIP_angle": angle_between_vectors(v2, v3),
        }
    return out


# ---------------------------------------------------------------------------
# Angular velocity
# ---------------------------------------------------------------------------

def calculate_angle_velocity(angles_df, df_full):
    """NaN-safe angular velocity via gradient of unwrapped angles w.r.t. TimeStamp."""
    if "TimeStamp" not in df_full.columns:
        print("Error: 'TimeStamp' column missing for velocity calculation.")
        return pd.DataFrame()

    t = df_full["TimeStamp"].to_numpy()
    out = {}

    for col in angles_df.columns:
        a_deg = angles_df[col].to_numpy()
        mask = ~np.isnan(a_deg)
        if mask.sum() < 3:
            out[f"VELOCITY_{col}"] = np.full_like(a_deg, np.nan)
            continue

        a_rad_unwrapped = np.unwrap(np.radians(a_deg[mask]))
        v_full = np.full_like(a_deg, np.nan)
        v_full[mask] = np.degrees(np.gradient(a_rad_unwrapped, t[mask]))
        out[f"VELOCITY_{col}"] = v_full

    return pd.DataFrame(out, index=angles_df.index)


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def get_3d_cols(prefix):
    return [f"{prefix}_X", f"{prefix}_Y", f"{prefix}_Z"]


def load_marker_coords(df, marker_name):
    cols = get_3d_cols(marker_name)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[WARN] Missing cols for {marker_name}: {missing}")
        return np.full((len(df), 3), np.nan)
    return df[cols].replace([np.inf, -np.inf], np.nan).to_numpy()


def calculate_kinematics_for_file(fpath):
    """Load one CSV, compute all joint angles + velocities, sample at keypress events."""
    df = pd.read_csv(fpath)
    df["Pressed_Letter"] = df["Pressed_Letter"].astype(str).str.lower()

    required = {"KeyPressFlag", "Pressed_Letter", "TimeStamp"}
    if not required.issubset(df.columns):
        return []

    coords = {k: load_marker_coords(df, v) for k, v in MARKER_MAP.items()}
    L1, L2, L3, L4 = coords["L1"], coords["L2"], coords["L3"], coords["L4"]
    R1, R2, R3, R4 = coords["R1"], coords["R2"], coords["R3"], coords["R4"]
    M1, M2, M3, M4 = coords["M1"], coords["M2"], coords["M3"], coords["M4"]
    I1, I2, I3, I4 = coords["I1"], coords["I2"], coords["I3"], coords["I4"]
    Cin, Cout = coords["Cin"], coords["Cout"]

    mcp_abd = compute_all_mcp_abduction_angles(L1, R1, M1, I1, L4, R4, M4, I4, Cin, Cout)
    mcp_flex = compute_mcp_flexion_angles(L1, L2, R1, R2, M1, M2, I1, I2, Cin, Cout)
    seg_flex = compute_all_finger_segment_angles(L1, L2, L3, L4, R1, R2, R3, R4, M1, M2, M3, M4, I1, I2, I3, I4)

    final_angles = {}
    for f, a in mcp_abd.items():
        final_angles[f"{f}_MCP_Abduction"] = a
    for f, a in mcp_flex.items():
        final_angles[f"{f}_MCP_Flexion"] = a
    for f, ang in seg_flex.items():
        final_angles[f"{f}_PIP_Flexion"] = ang["PIP_angle"]
        final_angles[f"{f}_DIP_Flexion"] = ang["DIP_angle"]

    angles_df = pd.DataFrame(final_angles, index=df.index)
    if DEG:
        angles_df = angles_df.apply(np.degrees).clip(lower=-180, upper=180)

    velocity_df = calculate_angle_velocity(angles_df, df)
    kinematics_df = angles_df.join(velocity_df)

    keystroke_idx = df.index[df["KeyPressFlag"] == 1]
    out = []
    for eidx in keystroke_idx:
        key = df.loc[eidx, "Pressed_Letter"]
        if key not in RIGHT_INDEX_KEYS:
            continue
        t = df.loc[eidx, "TimeStamp"]
        frame_kin = kinematics_df.loc[eidx].to_dict()

        for col, val in frame_kin.items():
            if col.startswith("VELOCITY_"):
                _, finger, joint, angle_type = col.split("_", 3)
                measure = f"VELOCITY_{angle_type}"
            else:
                finger, joint, angle_type = col.split("_")
                measure = angle_type

            if finger != "Index":
                continue
            out.append([key, "R", eidx, t, f"{joint}_{measure}", val])

    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _pivot_for_plot(angles_df, target_keys, measures):
    df_filt = angles_df[
        (angles_df["Pressed_Letter"].isin(target_keys))
        & (angles_df["Joint_Angle_Type"].isin(measures))
    ].copy()
    if df_filt.empty:
        return None
    pivoted = (
        df_filt.pivot_table(
            index=["Pressed_Letter", "Frame_Index"],
            columns="Joint_Angle_Type",
            values="Angle_Value",
        )
        .reset_index()
        .dropna(subset=measures)
    )
    if pivoted.empty:
        return None
    keys = sorted(pivoted["Pressed_Letter"].unique())
    pivoted["Key_Numeric"] = pivoted["Pressed_Letter"].map({k: i for i, k in enumerate(keys)})
    return pivoted, keys


def plot_3d_mcp_kinematics(angles_df, target_keys):
    """3D MCP kinematic space: Abduction × Flexion × Abduction Velocity."""
    measures = ["MCP_Abduction", "MCP_Flexion", "MCP_VELOCITY_Abduction"]
    result = _pivot_for_plot(angles_df, target_keys, measures)
    if result is None:
        print("3D MCP plot skipped: missing data.")
        return
    df, keys = result
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        df["MCP_Abduction"], df["MCP_Flexion"], df["MCP_VELOCITY_Abduction"],
        c=df["Key_Numeric"], cmap="viridis", s=50, alpha=0.6,
    )
    ax.set_title("3D MCP Keystroke Space (R Index)", fontsize=16)
    ax.set_xlabel("MCP Abduction (°)")
    ax.set_ylabel("MCP Flexion (°)")
    ax.set_zlabel("MCP Abduction Velocity (°/s)")
    cbar = fig.colorbar(sc, ticks=range(len(keys)))
    cbar.ax.set_yticklabels(keys)
    cbar.set_label("Pressed Character", rotation=270, labelpad=20)
    plt.tight_layout()
    plt.show()


def plot_3d_flexion_clusters(angles_df, target_keys):
    """3D flexion space: MCP × PIP × DIP flexion."""
    measures = ["MCP_Flexion", "PIP_Flexion", "DIP_Flexion"]
    result = _pivot_for_plot(angles_df, target_keys, measures)
    if result is None:
        print("3D flexion plot skipped: missing data.")
        return
    df, keys = result
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        df["MCP_Flexion"], df["PIP_Flexion"], df["DIP_Flexion"],
        c=df["Key_Numeric"], cmap="plasma", s=60, alpha=0.7,
    )
    ax.set_title("3D Flexion Angle Clusters (R Index)", fontsize=16)
    ax.set_xlabel("MCP Flexion (°)")
    ax.set_ylabel("PIP Flexion (°)")
    ax.set_zlabel("DIP Flexion (°)")
    ax.grid(True, linestyle="--", alpha=0.3)
    cbar = fig.colorbar(sc, ticks=range(len(keys)))
    cbar.ax.set_yticklabels(keys)
    cbar.set_label("Pressed Character", rotation=270, labelpad=20)
    plt.tight_layout()
    plt.show()


def plot_3d_mcp_abd_flex_pip_clusters(angles_df, target_keys):
    """3D mixed space: MCP Abduction × MCP Flexion × PIP Flexion."""
    measures = ["MCP_Abduction", "MCP_Flexion", "PIP_Flexion"]
    result = _pivot_for_plot(angles_df, target_keys, measures)
    if result is None:
        print("3D MCP-Abd/Flex/PIP plot skipped: missing data.")
        return
    df, keys = result
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        df["MCP_Abduction"], df["MCP_Flexion"], df["PIP_Flexion"],
        c=df["Key_Numeric"], cmap="plasma", s=60, alpha=0.7,
    )
    ax.set_title("3D MCP–PIP Angle Space (R Index)", fontsize=16)
    ax.set_xlabel("MCP Abduction (°)")
    ax.set_ylabel("MCP Flexion (°)")
    ax.set_zlabel("PIP Flexion (°)")
    ax.grid(True, linestyle="--", alpha=0.3)
    cbar = fig.colorbar(sc, ticks=range(len(keys)))
    cbar.ax.set_yticklabels(keys)
    cbar.set_label("Pressed Character", rotation=270, labelpad=20)
    plt.tight_layout()
    plt.show()


def plot_flexion_velocity_grid(angles_df, target_keys):
    """2D facet grid: flexion angle vs flexion velocity, one panel per joint (MCP/PIP/DIP)."""
    flex_meas = ["MCP_Flexion", "PIP_Flexion", "DIP_Flexion"]
    vel_meas = ["MCP_VELOCITY_Flexion", "PIP_VELOCITY_Flexion", "DIP_VELOCITY_Flexion"]

    df_flex = angles_df[
        (angles_df["Pressed_Letter"].isin(target_keys))
        & (angles_df["Joint_Angle_Type"].isin(flex_meas))
    ].rename(columns={"Angle_Value": "Angle"})
    df_vel = angles_df[
        (angles_df["Pressed_Letter"].isin(target_keys))
        & (angles_df["Joint_Angle_Type"].isin(vel_meas))
    ].rename(columns={"Angle_Value": "Velocity"})

    df_flex["Joint"] = df_flex["Joint_Angle_Type"].str.replace("_Flexion", "", regex=False)
    df_vel["Joint"] = df_vel["Joint_Angle_Type"].str.replace("_VELOCITY_Flexion", "", regex=False)

    df_merged = pd.merge(
        df_flex[["Pressed_Letter", "Frame_Index", "Joint", "Angle"]],
        df_vel[["Pressed_Letter", "Frame_Index", "Joint", "Velocity"]],
        on=["Pressed_Letter", "Frame_Index", "Joint"],
        how="inner",
    ).dropna()

    if df_merged.empty:
        print("2D grid plot skipped: no data.")
        return

    g = sns.FacetGrid(
        df_merged, col="Joint", hue="Pressed_Letter", palette="viridis", col_wrap=3, height=5
    )
    g.map(sns.scatterplot, "Angle", "Velocity", alpha=0.5, s=20)
    g.add_legend(title="Keystroke")
    g.set_axis_labels("Flexion Angle (°)", "Flexion Velocity (°/s)")
    g.fig.suptitle("Flexion Angle vs Velocity (R Index)", y=1.02, fontsize=16)
    plt.show()


PLOT_REGISTRY = {
    "mcp_kinematics": plot_3d_mcp_kinematics,
    "flexion_clusters": plot_3d_flexion_clusters,
    "mcp_pip_flex": plot_3d_mcp_abd_flex_pip_clusters,
    "flex_vel_grid": plot_flexion_velocity_grid,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract_kinematics(data_path, max_workers=MAX_WORKERS):
    """Run kinematic extraction across all CSVs in `data_path` (recursive)."""
    files = sorted(glob.glob(os.path.join(data_path, "**", "*.csv"), recursive=True))
    print(f"Found {len(files)} CSV files under {data_path}.")
    if not files:
        return pd.DataFrame()

    headers = [
        "Pressed_Letter", "Hand", "Frame_Index", "TimeStamp",
        "Joint_Angle_Type", "Angle_Value",
    ]
    all_rows = []
    print(f"Processing with {max_workers} workers...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        for rows in ex.map(calculate_kinematics_for_file, files):
            all_rows.extend(rows)

    return pd.DataFrame(all_rows, columns=headers)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--data", required=True,
        help="Directory containing motion-capture CSVs (searched recursively).",
    )
    parser.add_argument(
        "--plot", default="mcp_pip_flex",
        choices=list(PLOT_REGISTRY.keys()) + ["all", "none"],
        help="Which plot to render after extraction.",
    )
    parser.add_argument(
        "--workers", type=int, default=MAX_WORKERS,
        help="Number of parallel processes for CSV processing.",
    )
    parser.add_argument(
        "--save-csv", default=None,
        help="Optional path to save the extracted kinematics dataframe as CSV.",
    )
    args = parser.parse_args()

    start = time.time()
    angles_df = extract_kinematics(args.data, max_workers=args.workers)
    elapsed = time.time() - start

    print("=" * 50)
    print(f"Kinematic extraction complete in {elapsed:.2f}s")
    print(f"Total rows: {len(angles_df)}")
    print("=" * 50)

    if angles_df.empty:
        return

    if args.save_csv:
        angles_df.to_csv(args.save_csv, index=False)
        print(f"Saved kinematics to {args.save_csv}")

    if args.plot == "none":
        return
    plots_to_run = list(PLOT_REGISTRY.values()) if args.plot == "all" else [PLOT_REGISTRY[args.plot]]
    for plot_fn in plots_to_run:
        print(f"\nRendering: {plot_fn.__name__}")
        plot_fn(angles_df, RIGHT_INDEX_KEYS)


if __name__ == "__main__":
    main()
