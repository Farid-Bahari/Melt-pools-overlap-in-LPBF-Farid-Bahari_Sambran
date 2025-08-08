# streamlit_app.py
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit page configuration
st.set_page_config(page_title="Melt Pool Simulation", layout="wide")

st.title("Melt Pool Cross-Section Simulation")

# Sidebar for user input
st.sidebar.header("Simulation Parameters (µm unless noted)")
width = st.sidebar.number_input("Melt Pool Width", value=130, min_value=10, max_value=1000, step=5)
depth = st.sidebar.number_input("Melt Pool Depth", value=65, min_value=5, max_value=1000, step=5)
layer_thickness = st.sidebar.number_input("Layer Thickness", value=25, min_value=1, max_value=500, step=1)
hatch_distance = st.sidebar.number_input("Hatch Distance", value=130, min_value=10, max_value=1000, step=5)
rotation_angle_deg = st.sidebar.number_input("Rotation Angle (degrees)", value=67, min_value=0, max_value=180, step=1)
cut_plane_depth = st.sidebar.number_input("Cut Plane Depth", value=350, min_value=0, max_value=5000, step=10)

# Fixed parameters
theta = np.linspace(0, np.pi, 100)
extrusion_depth = 1300
rotation_center = np.array([350, 650])

num_layers = 40
num_paths = 10

# Create plot
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

for j in range(num_layers):
    angle_rad = np.radians(j * rotation_angle_deg)
    Rz = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    extrude_dir = Rz @ np.array([0, 1])

    for i in range(num_paths):
        center_x = width + i * hatch_distance - 300
        center_y = 0
        center_z = j * layer_thickness

        # Half ellipse
        x_arc = width/2 * np.cos(theta)
        z_arc = -depth * np.sin(theta)
        x = np.concatenate([x_arc, x_arc[::-1]])
        z = np.concatenate([z_arc, np.full_like(z_arc, z_arc[-1])])

        ellipse_xy = np.vstack([x + center_x, np.zeros_like(x) + center_y])
        ellipse_xy_centered = ellipse_xy - rotation_center[:, np.newaxis]
        ellipse_xy_rot = Rz @ ellipse_xy_centered + rotation_center[:, np.newaxis]

        x_rot = ellipse_xy_rot[0]
        y_rot = ellipse_xy_rot[1]
        z_rot = z + center_z

        # Extrude
        X0, Y0, Z0 = x_rot, y_rot, z_rot
        X1 = x_rot + extrusion_depth * extrude_dir[0]
        Y1 = y_rot + extrusion_depth * extrude_dir[1]
        Z1 = z_rot

        x_cross, z_cross = [], []

        # Front face intersection
        for k in range(len(X0) - 1):
            if (Y0[k] - cut_plane_depth) * (Y0[k+1] - cut_plane_depth) <= 0:
                if Y0[k] != Y0[k+1]:
                    t = (cut_plane_depth - Y0[k]) / (Y0[k+1] - Y0[k])
                    x_cross.append(X0[k] + t * (X0[k+1] - X0[k]))
                    z_cross.append(Z0[k] + t * (Z0[k+1] - Z0[k]))

        # Back face intersection
        for k in range(len(X1) - 1):
            if (Y1[k] - cut_plane_depth) * (Y1[k+1] - cut_plane_depth) <= 0:
                if Y1[k] != Y1[k+1]:
                    t = (cut_plane_depth - Y1[k]) / (Y1[k+1] - Y1[k])
                    x_cross.append(X1[k] + t * (X1[k+1] - X1[k]))
                    z_cross.append(Z1[k] + t * (Z1[k+1] - Z1[k]))

        # Side walls intersection
        for k in range(len(X0)):
            if (Y0[k] - cut_plane_depth) * (Y1[k] - cut_plane_depth) <= 0:
                if Y0[k] != Y1[k]:
                    t = (cut_plane_depth - Y0[k]) / (Y1[k] - Y0[k])
                    x_cross.append(X0[k] + t * (X1[k] - X0[k]))
                    z_cross.append(Z0[k] + t * (Z1[k] - Z0[k]))

        if x_cross:
            x_cross = np.array(x_cross)
            z_cross = np.array(z_cross)
            center_x_cross = np.mean(x_cross)
            center_z_cross = np.mean(z_cross)
            angles = np.arctan2(z_cross - center_z_cross, x_cross - center_x_cross)
            sorted_indices = np.argsort(angles)
            ax.fill(
                x_cross[sorted_indices],
                z_cross[sorted_indices],
                facecolor=(0, 0, 0, 0.3),
                edgecolor='black',
                linewidth=0.5
            )

ax.set_xlabel('X-coordinate (µm)')
ax.set_ylabel('Z-coordinate (µm)')
ax.set_title(f"MPD = {depth} µm, MPW = {width} µm, HD = {hatch_distance} µm, Rotation = {rotation_angle_deg}°")
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0, 750)
ax.set_ylim(0, 750)
plt.tight_layout()

# Display plot in Streamlit
st.pyplot(fig)
