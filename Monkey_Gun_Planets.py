import streamlit as st
import numpy as np
import cv2
import time
import urllib.request

st.set_page_config(page_title="Monkey Gun Physics Simulation", layout="wide")
st.title("🐵 Monkey and Hunter Simulation – Planetary Gravity (URL Image)")

# ----- Sidebar Controls -----
st.sidebar.header("Simulation Parameters")
v0 = st.sidebar.slider("Projectile speed (m/s)", 5, 50, 20)
target_height = st.sidebar.slider("Target height (m)", 1, 15, 10)
distance = st.sidebar.slider("Distance to target (m)", 5, 30, 15)
fps = st.sidebar.slider("Frames per second", 10, 60, 30)
hit_radius = 0.5  # meters

# Planetary gravities (m/s²)
gravities = {
    "Mercury": 3.7, "Venus": 8.87, "Earth": 9.8, "Moon": 1.62,
    "Mars": 3.71, "Jupiter": 24.79, "Saturn": 10.44,
    "Uranus": 8.69, "Neptune": 11.15, "Pluto": 0.62
}
planet = st.sidebar.selectbox("Planet / Moon", list(gravities.keys()))
gravity = gravities[planet]

fire = st.button("Fire!")

# Canvas setup
width, height = 600, 400
scale = 25  # pixels per meter

# ----- Load monkey image from URL -----
# Example: a small transparent PNG hosted online
monkey_url = "https://i.imgur.com/4AiXzf8.png"  # Replace with your preferred image
try:
    resp = urllib.request.urlopen(monkey_url)
    monkey_img = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    monkey_img = cv2.imdecode(monkey_img, cv2.IMREAD_UNCHANGED)
    if monkey_img is not None:
        monkey_h, monkey_w = monkey_img.shape[:2]
    else:
        st.warning("Monkey image could not be loaded.")
except:
    st.warning("Monkey image could not be loaded.")
    monkey_img = None

# ----- Physics Function -----
def simulate_positions(v0, target_height, distance, g, dt, t_max=5):
    theta = np.arctan2(target_height, distance)
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    t_vals = np.arange(0, t_max, dt)
    px = vx * t_vals
    py = vy * t_vals - 0.5 * g * t_vals**2
    tx = np.full_like(t_vals, distance)
    ty = target_height - 0.5 * g * t_vals**2
    return t_vals, px, py, tx, ty

# ----- Fire Simulation -----
if fire:
    dt = 1 / fps
    t_vals, px, py, tx, ty = simulate_positions(v0, target_height, distance, gravity, dt)

    canvas_placeholder = st.empty()
    hit = False

    # Trail image for motion blur
    trail_frame = np.ones((height, width, 3), dtype=np.uint8) * 255
    alpha_decay = 0.85  # motion blur fading factor

    projectile_color = (0,0,255)
    target_color = (0,255,0)
    arrow_color = (255,0,0)

    for i in range(len(t_vals)):
        trail_frame = (trail_frame * alpha_decay).astype(np.uint8)

        proj_x = int(px[i] * scale)
        proj_y = int(height - py[i] * scale)
        targ_x = int(tx[i] * scale)
        targ_y = int(height - ty[i] * scale)

        cv2.circle(trail_frame, (proj_x, proj_y), 6, projectile_color, -1)

        frame = trail_frame.copy()
        cv2.circle(frame, (targ_x, targ_y), 8, target_color, -1)

        if monkey_img is not None:
            y1 = targ_y - monkey_h
            y2 = y1 + monkey_h
            x1 = targ_x - monkey_w//2
            x2 = x1 + monkey_w
            if y1 >= 0 and x1 >= 0 and x2 < width:
                alpha_s = monkey_img[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                for c in range(3):
                    frame[y1:y2, x1:x2, c] = (alpha_s * monkey_img[:, :, c] +
                                              alpha_l * frame[y1:y2, x1:x2, c])

        if i < len(t_vals)-1:
            vx_pixel = int((px[i+1]-px[i])*scale*5)
            vy_pixel = int(-(py[i+1]-py[i])*scale*5)
            cv2.arrowedLine(frame, (proj_x, proj_y),
                            (proj_x + vx_pixel, proj_y + vy_pixel),
                            arrow_color, 2, tipLength=0.3)

        distance_to_target = np.sqrt((px[i]-tx[i])**2 + (py[i]-ty[i])**2)
        if distance_to_target <= hit_radius:
            hit = True
            cv2.putText(frame, "HIT!", (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0,0,255), 3)
            st.sidebar.success(f"Hit on {planet} at t={t_vals[i]:.2f}s")
            canvas_placeholder.image(frame)
            break

        if py[i] < 0:
            if not hit:
                st.sidebar.warning(f"Projectile hit the ground on {planet}!")
            break

        canvas_placeholder.image(frame)
        time.sleep(dt)