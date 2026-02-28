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
WIDTH, HEIGHT = 600, 400
SCALE = 25          # pixels per metre
HIT_RADIUS = 0.5   # metres

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Simulation Parameters")
v0             = st.sidebar.slider("Projectile speed (m/s)", 5, 50, 20)
target_height  = st.sidebar.slider("Target height (m)", 1, 15, 10)
distance       = st.sidebar.slider("Distance to target (m)", 5, 20, 15)
fps            = st.sidebar.slider("Frames per second", 10, 60, 30)
playback_speed = st.sidebar.select_slider(
    "Playback speed", options=[0.25, 0.5, 1.0, 2.0], value=1.0,
    format_func=lambda x: f"{int(x * 100)}%"
)
angle_offset_deg = st.sidebar.slider("Aim offset (degrees)", -20, 20, 0)

gravities = {
    "Mercury": 3.7,  "Venus": 8.87,  "Earth": 9.8,   "Moon": 1.62,
    "Mars":    3.71, "Jupiter": 24.79, "Saturn": 10.44,
    "Uranus":  8.69, "Neptune": 11.15, "Pluto": 0.62
}
planet  = st.sidebar.selectbox("Planet / Moon", list(gravities.keys()))
gravity = gravities[planet]

fire   = st.button("Fire!")
replay = st.button("Replay")

# -----------------------------
# Load monkey image (optional)
# -----------------------------
monkey_path = os.path.join("assets", "monkey.png")
monkey_img  = None
MONKEY_W = MONKEY_H = 0

if os.path.exists(monkey_path):
    monkey_img = cv2.imread(monkey_path, cv2.IMREAD_UNCHANGED)
    if monkey_img is not None:
        raw_h, raw_w = monkey_img.shape[:2]
        desired_h    = 40
        new_w        = int(desired_h * raw_w / raw_h)
        monkey_img   = cv2.resize(monkey_img, (new_w, desired_h))
        MONKEY_H, MONKEY_W = monkey_img.shape[:2]
else:
    st.sidebar.info("No assets/monkey.png found – using a coloured dot for the monkey.")

# -----------------------------
# Drawing helpers
# -----------------------------
def draw_shooter(frame, x, y):
    """Simple stick-figure hunter."""
    cv2.line(frame,   (x, y),      (x, y - 30),      (0, 0, 0), 2)
    cv2.circle(frame, (x, y - 40), 8,                 (0, 0, 0), 2)
    cv2.line(frame,   (x, y - 25), (x + 20, y - 35), (0, 0, 0), 3)


def overlay_monkey(frame, cx, cy):
    """
    Draw monkey centred at pixel (cx, cy).
    Falls back to a coloured circle if no image is loaded.
    """
    if monkey_img is not None:
        x1 = cx - MONKEY_W // 2
        y1 = cy - MONKEY_H // 2
        x2 = x1 + MONKEY_W
        y2 = y1 + MONKEY_H

        # Clamp to canvas bounds
        fx1 = max(x1, 0); fy1 = max(y1, 0)
        fx2 = min(x2, WIDTH); fy2 = min(y2, HEIGHT)
        if fx2 <= fx1 or fy2 <= fy1:
            return  # entirely off-screen

        # Corresponding crop in the sprite
        sx1 = fx1 - x1; sy1 = fy1 - y1
        sx2 = sx1 + (fx2 - fx1); sy2 = sy1 + (fy2 - fy1)

        sprite = monkey_img[sy1:sy2, sx1:sx2]
        if sprite.shape[2] == 4:                      # RGBA
            alpha = sprite[:, :, 3:4] / 255.0
            frame[fy1:fy2, fx1:fx2] = (
                alpha * sprite[:, :, :3] +
                (1 - alpha) * frame[fy1:fy2, fx1:fx2]
            ).astype(np.uint8)
        else:
            frame[fy1:fy2, fx1:fx2] = sprite[:, :, :3]
    else:
        # Fallback: orange circle
        cv2.circle(frame, (cx, cy), 10, (0, 140, 255), -1)
        cv2.circle(frame, (cx, cy), 10, (0, 0, 0), 1)


# -----------------------------
# Physics
# -----------------------------
def simulate_positions(v0, target_height, distance, g, dt,
                        angle_offset_deg=0, t_max=5):
    """
    Returns arrays of time, projectile (px,py) and monkey (tx,ty).

    Key physics insight
    ------------------
    Both the bullet and the monkey fall under the SAME gravitational
    acceleration from t=0.  The bullet is aimed at the monkey's *initial*
    position.  Therefore they always meet (assuming the bullet reaches
    the horizontal distance before hitting the ground), regardless of g.

    Projectile:
        px(t) = vx * t
        py(t) = vy * t  -  ½ g t²

    Monkey (free-fall from rest at (distance, target_height)):
        tx(t) = distance           (fixed horizontal)
        ty(t) = target_height  -  ½ g t²
    """
    theta = np.arctan2(target_height, distance) + np.deg2rad(angle_offset_deg)
    vx    = v0 * np.cos(theta)
    vy    = v0 * np.sin(theta)

    t_vals = np.arange(0, t_max, dt)

    px = vx * t_vals
    py = vy * t_vals - 0.5 * g * t_vals ** 2   # bullet y

    tx = np.full_like(t_vals, distance)          # monkey x stays fixed
    ty = target_height - 0.5 * g * t_vals ** 2  # monkey y – independent fall

    return t_vals, px, py, tx, ty


# -----------------------------
# Simulation renderer
# -----------------------------
def run_simulation():
    dt = 1.0 / fps
    t_vals, px, py, tx, ty = simulate_positions(
        v0, target_height, distance, gravity, dt, angle_offset_deg
    )

    canvas = st.empty()
    hit    = False

    shooter_px = 0        # hunter starts at left edge
    shooter_py = HEIGHT

    # Branch geometry (static – where the monkey *started*)
    branch_y_px    = HEIGHT - int(target_height * SCALE)
    branch_x_start = int(distance * SCALE) - 30
    branch_x_end   = int(distance * SCALE) + 30

    # Motion-blur trail accumulator
    trail = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255

    for i in range(len(t_vals)):
        # ── Pixel positions ────────────────────────────────────────────
        proj_px = int(px[i] * SCALE)
        proj_py = HEIGHT - int(py[i] * SCALE)   # flip Y (screen coords)

        monk_cx = int(tx[i] * SCALE)            # always at target x
        monk_cy = HEIGHT - int(ty[i] * SCALE)   # falls independently

        # ── Motion-blur fade ───────────────────────────────────────────
        trail = (trail * 0.85).astype(np.uint8)
        cv2.circle(trail, (proj_px, proj_py), 3, (0, 0, 200), -1)

        frame = trail.copy()

        # ── Static scene elements ──────────────────────────────────────
        draw_shooter(frame, shooter_px, shooter_py)

        # Branch (stays at original height)
        cv2.line(frame,
                 (branch_x_start, branch_y_px),
                 (branch_x_end,   branch_y_px),
                 (101, 67, 33), 3)

        # Aim-line from gun tip to monkey's *initial* position
        cv2.line(frame,
                 (shooter_px + 20, shooter_py - 35),
                 (branch_x_end,    branch_y_px),
                 (80, 80, 80), 1, cv2.LINE_AA)

        # ── Monkey ─────────────────────────────────────────────────────
        overlay_monkey(frame, monk_cx, monk_cy)

        # ── Velocity arrow on bullet ───────────────────────────────────
        if i < len(t_vals) - 1:
            dx = int((px[i + 1] - px[i]) * SCALE * 2)
            dy = int(-(py[i + 1] - py[i]) * SCALE * 2)
            cv2.arrowedLine(frame,
                            (proj_px, proj_py),
                            (proj_px + dx, proj_py + dy),
                            (200, 0, 0), 2)

        # ── Hit / miss detection ───────────────────────────────────────
        dist_m = np.hypot(px[i] - tx[i], py[i] - ty[i])  # in metres

        if dist_m <= HIT_RADIUS:
            hit = True
            label = f"HIT!  t = {t_vals[i]:.2f} s"
            cv2.putText(frame, label, (160, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 180, 0), 3)
            canvas.image(frame, channels="BGR")
            break

        # Monkey hits the ground
        if ty[i] <= 0 and not hit:
            label = f"MISS – monkey hit ground at t = {t_vals[i]:.2f} s"
            cv2.putText(frame, label, (60, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
            canvas.image(frame, channels="BGR")
            break

        # Bullet hits the ground without reaching monkey
        # Use < -0.01 (not <= 0) so the starting position y=0 doesn't trigger this
        if py[i] < -0.01 and not hit:
            label = f"MISS – bullet hit ground at t = {t_vals[i]:.2f} s"
            cv2.putText(frame, label, (60, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
            canvas.image(frame, channels="BGR")
            break

        canvas.image(frame, channels="BGR")
        time.sleep(dt / playback_speed)

# -----------------------------
# Entry point
# -----------------------------
if fire or replay:
    run_simulation()
