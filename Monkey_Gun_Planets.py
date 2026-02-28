import streamlit as st
import numpy as np
import cv2
import time
import os

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Monkey Gun Physics Simulation", layout="wide")
st.title("🐵 Monkey and Hunter Simulation – Planetary Gravity")

# Narrative / explanation
st.markdown("""
### Monkey Gun Problem
A hunter aims directly at a monkey sitting on a tree branch.  
As soon as the hunter fires, the monkey lets go and falls. Will the projectile hit the monkey?

- **Projectile speed**: set how fast the bullet travels  
- **Distance & height**: place the monkey  
- **Planet**: select gravity to see how physics changes  
- **Playback speed**: slow motion for better observation  
- **Aim offset**: try firing above or below to see the paradox in action
""")

# -----------------------------
# Constants
# -----------------------------
width, height = 600, 400
scale = 25               # pixels per meter
hit_radius = 0.5         # meters

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Simulation Parameters")
v0 = st.sidebar.slider("Projectile speed (m/s)", 5, 50, 20)
target_height = st.sidebar.slider("Target height (m)", 1, 15, 10)
distance = st.sidebar.slider("Distance to target (m)", 5, 20, 15)  # reduced max to fit canvas
fps = st.sidebar.slider("Frames per second", 10, 60, 30)

playback_speed = st.sidebar.select_slider(
    "Playback speed",
    options=[0.25, 0.5, 1.0, 2.0],
    value=1.0,
    format_func=lambda x: f"{int(x*100)}%"
)

# Optional angle override to illustrate missing the monkey
angle_offset_deg = st.sidebar.slider("Aim offset (degrees)", -20, 20, 0)

# Planetary gravities
gravities = {
    "Mercury": 3.7, "Venus": 8.87, "Earth": 9.8, "Moon": 1.62,
    "Mars": 3.71, "Jupiter": 24.79, "Saturn": 10.44,
    "Uranus": 8.69, "Neptune": 11.15, "Pluto": 0.62
}
planet = st.sidebar.selectbox("Planet / Moon", list(gravities.keys()))
gravity = gravities[planet]

# Buttons
fire = st.button("Fire!")
replay = st.button("Replay")  # New replay button

# -----------------------------
# Load monkey image
# -----------------------------
monkey_path = os.path.join("assets", "monkey.png")
monkey_img = None

if os.path.exists(monkey_path):
    monkey_img = cv2.imread(monkey_path, cv2.IMREAD_UNCHANGED)
    monkey_h, monkey_w = monkey_img.shape[:2]
else:
    st.sidebar.warning("Monkey image not found in assets/monkey.png")

# -----------------------------
# Drawing helpers
# -----------------------------
def draw_shooter(frame, x, y):
    cv2.line(frame, (x, y), (x, y-30), (0,0,0), 2)      # body
    cv2.circle(frame, (x, y-40), 8, (0,0,0), 2)         # head
    cv2.line(frame, (x, y-25), (x+20, y-35), (0,0,0), 3) # gun

def simulate_positions(v0, target_height, distance, g, dt, angle_offset_deg=0, t_max=5):
    theta = np.arctan2(target_height, distance) + np.deg2rad(angle_offset_deg)
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)

    t_vals = np.arange(0, t_max, dt)
    px = vx * t_vals
    py = vy * t_vals - 0.5 * g * t_vals**2
    tx = np.full_like(t_vals, distance)
    ty = target_height - 0.5 * g * t_vals**2
    return t_vals, px, py, tx, ty

# -----------------------------
# Function to run simulation
# -----------------------------
def run_simulation():
    dt = 1 / fps
    t_vals, px, py, tx, ty = simulate_positions(
        v0, target_height, distance, gravity, dt, angle_offset_deg
    )

    canvas = st.empty()
    hit = False

    shooter_x = int(0 * scale)
    shooter_y = height

    # Resize monkey to reasonable size
    if monkey_img is not None:
        desired_height = 40
        aspect = monkey_w / monkey_h
        new_w = int(desired_height * aspect)
        monkey_resized = cv2.resize(monkey_img, (new_w, desired_height))
        monkey_h2, monkey_w2 = monkey_resized.shape[:2]
    else:
        monkey_resized = None

    for i in range(len(t_vals)):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255

        proj_x = int(px[i] * scale)
        proj_y = int(height - py[i] * scale)
        targ_x = int(tx[i] * scale)
        targ_y = int(height - ty[i] * scale)

        # Shooter
        draw_shooter(frame, shooter_x, shooter_y)

        # Trail
        for j in range(i):
            trail_x = int(px[j] * scale)
            trail_y = int(height - py[j] * scale)
            cv2.circle(frame, (trail_x, trail_y), 2, (150,150,150), -1)

        # Projectile
        cv2.circle(frame, (proj_x, proj_y), 6, (0,0,255), -1)

        # Tree trunk
        trunk_x = targ_x - 15
        trunk_y = targ_y
        trunk_height = 60
        cv2.rectangle(frame, (trunk_x, trunk_y), (trunk_x+10, trunk_y+trunk_height), (101,67,33), -1)

        # Branch
        branch_length = 50
        branch_y = targ_y + 10
        cv2.line(frame, (trunk_x+5, branch_y), (trunk_x+5+branch_length, branch_y), (101,67,33), 5)

        # Monkey
        if monkey_resized is not None:
            y1 = branch_y - monkey_h2
            x1 = trunk_x + 5 + branch_length - monkey_w2 // 2
            y2 = y1 + monkey_h2
            x2 = x1 + monkey_w2
            if 0 <= x1 < width and 0 <= y1 < height and x2 < width and y2 < height:
                alpha_s = monkey_resized[:,:,3] / 255.0
                alpha_l = 1.0 - alpha_s
                for c in range(3):
                    frame[y1:y2, x1:x2, c] = (
                        alpha_s * monkey_resized[:,:,c] +
                        alpha_l * frame[y1:y2, x1:x2, c]
                    )

        # Aim line
        cv2.line(frame, (shooter_x + 20, shooter_y - 25),
                 (trunk_x + 5 + branch_length, branch_y - monkey_h2//2),
                 (0,0,0), 1, lineType=cv2.LINE_AA)

        # Velocity arrow
        if i < len(t_vals) - 1:
            dx = int((px[i+1]-px[i]) * scale * 5)
            dy = int(-(py[i+1]-py[i]) * scale * 5)
            cv2.arrowedLine(frame, (proj_x, proj_y),
                            (proj_x+dx, proj_y+dy),
                            (255,0,0), 2)

        # Hit detection
        dist = np.hypot(px[i]-tx[i], py[i]-ty[i])
        if dist <= hit_radius:
            hit = True
            hit_time = t_vals[i]
            cv2.putText(frame, f"HIT! t={hit_time:.2f}s", (200,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
            canvas.image(frame)
            break

        # Ground impact
        if py[i] < 0 and not hit:
            cv2.putText(frame, f"MISS! t={t_vals[i]:.2f}s", (200,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
            canvas.image(frame)
            break

        canvas.image(frame)
        time.sleep(dt / playback_speed)

# -----------------------------
# Run simulation
# -----------------------------
if fire or replay:
    run_simulation()
