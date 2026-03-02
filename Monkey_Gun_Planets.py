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
### Monkey and Hunter Problem
A monkey is swinging through the jungle, when he sees a hunter aiming directly at him.  
He thinks to himself - if I let go as soon as the hunter fires, I'll drop safely out of the way.
Will his plan work?
""")

with st.expander("ℹ️ How to use this simulation"):
    st.markdown("""
    Use the **sidebar** (tap **>** on mobile) to adjust the parameters, then press **Fire!**

    - **Projectile speed** — how fast the bullet travels
    - **Target height** — how high up the monkey is
    - **Distance to target** — how far away the monkey is
    - **Frames per second** — animation smoothness
    - **Playback speed** — slow down to see the physics in action (try 10–25%)
    - **Aim offset** — fire above or below the monkey to see what happens
    - **Planet** — change gravity to see how it affects the outcome

    *Tip: the hunter always aims directly at the monkey. Try different aim offsets
    and planets to explore the physics.*
    """)

# -----------------------------
# Constants
# -----------------------------
WIDTH, HEIGHT = 600, 400
SCALE = 25          # pixels per metre
HIT_RADIUS = 0.05  # metres (tight so any aim offset causes a miss)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Simulation Parameters")
v0             = st.sidebar.slider("Projectile speed (m/s)", 5, 50, 20)
target_height  = st.sidebar.slider("Target height (m)", 1, 15, 10)
distance       = st.sidebar.slider("Distance to target (m)", 5, 20, 15)
fps            = st.sidebar.slider("Frames per second", 10, 60, 30)
playback_pct   = st.sidebar.slider(
    "Playback speed (%)", min_value=5, max_value=100, value=100, step=5,
    format="%d%%"
)
playback_speed = playback_pct / 100.0
angle_offset_deg = st.sidebar.slider("Aim offset (degrees)", -20, 20, 0)

gravities = {
    "Mercury (38%)":    3.7,
    "Venus (90.4%)":    8.87,
    "Earth (100%)":     9.8,
    "Moon (16.5%)":     1.62,
    "Mars (38%)":       3.71,
    "Jupiter (252.8%)": 24.79,
    "Saturn (106.5%)":  10.44,
    "Uranus (88.6%)":   8.69,
    "Neptune (114%)":   11.15,
    "Pluto (6%)":       0.62,
}
planet  = st.sidebar.selectbox("Planet / Moon", list(gravities.keys()), index=2)
gravity = gravities[planet]

st.sidebar.markdown("""
---
*On mobile, tap **>** at the top left to open the sidebar.*
""")

fire   = st.button("Fire!")
replay = st.button("Replay")

# -----------------------------
# Worked math expander (live values)
# -----------------------------
gun_tip_x_m = 20 / SCALE        # 0.80 m
gun_tip_y_m = 35 / SCALE        # 1.40 m
dx = distance - gun_tip_x_m
dy = target_height - gun_tip_y_m
theta_rad = np.arctan2(dy, dx) + np.deg2rad(angle_offset_deg)
theta_deg = np.rad2deg(theta_rad)
vx = v0 * np.cos(theta_rad)
vy = v0 * np.sin(theta_rad)
t_meet = dx / vx if vx > 0 else float('inf')
monkey_y_at_meet = target_height - 0.5 * gravity * t_meet**2
t_monkey_ground = np.sqrt(2 * target_height / gravity)
min_speed = np.hypot(dx, dy) / t_monkey_ground

with st.expander("📐 Show the math — how would you solve this on paper?"):
    st.markdown(f"""
#### Step 1 — Find the aim angle

The hunter aims directly at the monkey, so we draw a right triangle:
- Horizontal leg = distance − gun height offset = **{distance} − {gun_tip_x_m:.1f} = {dx:.1f} m**
- Vertical leg = monkey height − gun height = **{target_height} − {gun_tip_y_m:.1f} = {dy:.1f} m**

$$\\theta = \\arctan\\!\\left(\\frac{{{dy:.1f}}}{{{dx:.1f}}}\\right) = {theta_deg:.1f}°$$

---

#### Step 2 — Break the muzzle velocity into components

$$v_x = v_0 \\cos\\theta = {v0} \\times \\cos({theta_deg:.1f}°) = {vx:.2f} \\text{{ m/s}}$$

$$v_y = v_0 \\sin\\theta = {v0} \\times \\sin({theta_deg:.1f}°) = {vy:.2f} \\text{{ m/s}}$$

---

#### Step 3 — Write equations of motion

**Bullet** (launched from gun tip at height {gun_tip_y_m:.1f} m):

$$x_{{\\text{{bullet}}}}(t) = {gun_tip_x_m:.1f} + {vx:.2f}\\,t$$

$$y_{{\\text{{bullet}}}}(t) = {gun_tip_y_m:.1f} + {vy:.2f}\\,t - \\tfrac{{1}}{{2}} \\times {gravity:.1f} \\times t^2$$

**Monkey** (drops from rest at height {target_height} m):

$$x_{{\\text{{monkey}}}}(t) = {distance} \\text{{ m (fixed)}}$$

$$y_{{\\text{{monkey}}}}(t) = {target_height} - \\tfrac{{1}}{{2}} \\times {gravity:.1f} \\times t^2$$

---

#### Step 4 — Find when the bullet reaches the monkey's x position

$$t_{{\\text{{meet}}}} = \\frac{{{dx:.1f}}}{{{vx:.2f}}} = {t_meet:.3f} \\text{{ s}}$$

---

#### Step 5 — Check both heights at t = {t_meet:.3f} s

$$y_{{\\text{{bullet}}}} = {gun_tip_y_m:.1f} + {vy:.2f} \\times {t_meet:.3f} - \\tfrac{{1}}{{2}} \\times {gravity:.1f} \\times {t_meet:.3f}^2 = {gun_tip_y_m + vy*t_meet - 0.5*gravity*t_meet**2:.2f} \\text{{ m}}$$

$$y_{{\\text{{monkey}}}} = {target_height} - \\tfrac{{1}}{{2}} \\times {gravity:.1f} \\times {t_meet:.3f}^2 = {monkey_y_at_meet:.2f} \\text{{ m}}$$

{'✅ **Both heights match — the bullet hits the monkey!**' if abs(angle_offset_deg) == 0 else '⚠️ **Aim offset applied — heights no longer match, so this is a miss.**'}

---

#### Why does gravity cancel out?

Notice that both equations contain **−½ × g × t²**. Since both objects experience
the same gravitational acceleration from the same moment (t = 0), gravity affects
them equally — it simply cancels out of the comparison. The bullet always hits,
regardless of which planet you're on, as long as it reaches the monkey before
either hits the ground.

---

#### Minimum speed to guarantee a hit

The monkey hits the ground at:

$$t_{{\\text{{ground}}}} = \\sqrt{{\\frac{{2 \\times {target_height}}}{{{gravity:.1f}}}}} = {t_monkey_ground:.3f} \\text{{ s}}$$

The bullet must travel **{np.hypot(dx,dy):.1f} m** (straight-line distance) in less than that time:

$$v_{{\\text{{min}}}} = \\frac{{{np.hypot(dx,dy):.1f}}}{{{t_monkey_ground:.3f}}} = {min_speed:.1f} \\text{{ m/s}}$$

{'✅ Current speed **' + str(v0) + ' m/s** is fast enough.' if v0 >= min_speed else '⚠️ Current speed **' + str(v0) + ' m/s** is too slow — the monkey will hit the ground first!'}
    """)

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
        desired_h    = 20
        new_w        = int(desired_h * raw_w / raw_h)
        monkey_img   = cv2.resize(monkey_img, (new_w, desired_h))
        MONKEY_H, MONKEY_W = monkey_img.shape[:2]
else:
    st.sidebar.info("No assets/monkey.png found - using a coloured dot for the monkey.")

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

    # Always draw a small green centre dot so hit point is unambiguous
    cv2.circle(frame, (cx, cy), 3, (0, 200, 0), -1)


# -----------------------------
# Physics
# -----------------------------
def simulate_positions(v0, target_height, distance, g, dt,
                        angle_offset_deg=0, t_max=5,
                        origin_x_m=0.0, origin_y_m=0.0):
    """
    Returns arrays of time, projectile (px,py) and monkey (tx,ty).
    px/py are relative to origin_x_m / origin_y_m.
    The aim angle is computed from the gun tip to the monkey's initial position.

    Key physics insight
    ------------------
    Both the bullet and the monkey experience the same gravitational acceleration
    from t=0. The bullet is aimed at the monkey's *initial* position. Therefore
    they always meet (provided the bullet reaches the target x before hitting the
    ground), regardless of g. Any aim offset breaks this and produces a miss.

    Projectile (relative to gun tip):
        px(t) = vx * t
        py(t) = vy * t  -  half * g * t^2

    Monkey (absolute coords, free-fall from rest):
        tx(t) = distance           (fixed horizontal)
        ty(t) = target_height  -  half * g * t^2
    """
    dx_to_monkey = distance      - origin_x_m
    dy_to_monkey = target_height - origin_y_m
    theta = np.arctan2(dy_to_monkey, dx_to_monkey) + np.deg2rad(angle_offset_deg)
    vx    = v0 * np.cos(theta)
    vy    = v0 * np.sin(theta)

    t_vals = np.arange(0, t_max, dt)

    px = vx * t_vals
    py = vy * t_vals - 0.5 * g * t_vals ** 2   # bullet y, relative to gun tip

    tx = np.full_like(t_vals, distance)          # monkey x stays fixed
    ty = target_height - 0.5 * g * t_vals ** 2  # monkey y, absolute

    return t_vals, px, py, tx, ty


# -----------------------------
# Simulation renderer
# -----------------------------
def run_simulation():
    dt = 1.0 / fps

    shooter_px = 0
    shooter_py = HEIGHT

    # Gun tip in pixel space - projectile launches from here
    gun_tip_px = shooter_px + 20
    gun_tip_py = shooter_py - 35

    # Gun tip in physics/metres space (bullet's true origin)
    gun_tip_x_m = gun_tip_px / SCALE          # 0.80 m
    gun_tip_y_m = (HEIGHT - gun_tip_py) / SCALE  # 1.40 m

    t_vals, px, py, tx, ty = simulate_positions(
        v0, target_height, distance, gravity, dt, angle_offset_deg,
        origin_x_m=gun_tip_x_m, origin_y_m=gun_tip_y_m
    )

    canvas = st.empty()
    hit    = False

    # Branch geometry (static - where the monkey started)
    branch_y_px    = HEIGHT - int(target_height * SCALE)
    branch_x_start = int(distance * SCALE) - 30
    branch_x_end   = int(distance * SCALE) + 30

    # Monkey's initial centre pixel (for aim line target)
    monkey_start_cx = int(distance * SCALE)
    monkey_start_cy = HEIGHT - int(target_height * SCALE)

    # Aim-line fade setup (currently always visible — see comment below)
    aim_fade_frames = int(0.5 * fps / playback_speed)

    # Persistent trajectory layer - dots stamped here are never faded
    traj_layer = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255

    # Motion-blur trail - fades toward white each frame
    trail = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255

    # One trajectory dot every 0.1 s
    dot_interval = max(1, int(0.1 * fps))

    # Track previous bullet/monkey positions for between-frame hit detection
    prev_bx_m = gun_tip_x_m
    prev_by_m = gun_tip_y_m
    prev_mx_m = float(distance)
    prev_my_m = float(target_height)

    min_dist_overall = 999.0   # track closest approach for close-call message
    min_dist_time    = 0.0     # time at which closest approach occurred

    # Render a static first frame (scene before firing) so Streamlit is ready
    first_frame = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
    cv2.line(first_frame, (0, HEIGHT - 1), (WIDTH - 1, HEIGHT - 1), (80, 120, 40), 2)
    cv2.line(first_frame,
             (int(distance * SCALE) - 30, HEIGHT - int(target_height * SCALE)),
             (int(distance * SCALE) + 30, HEIGHT - int(target_height * SCALE)),
             (101, 67, 33), 3)
    draw_shooter(first_frame, shooter_px, shooter_py)
    overlay_monkey(first_frame, int(distance * SCALE), HEIGHT - int(target_height * SCALE))
    canvas.image(first_frame, channels="BGR")
    time.sleep(0.5)  # hold the static frame so Streamlit is fully ready before animating

    for i in range(len(t_vals)):
        frame_start = time.time()   # wall-clock start for this frame

        # ── Pixel positions ────────────────────────────────────────────
        proj_px = gun_tip_px + int(px[i] * SCALE)
        proj_py = gun_tip_py - int(py[i] * SCALE)   # screen Y is inverted

        monk_cx = int(tx[i] * SCALE)
        monk_cy = HEIGHT - int(ty[i] * SCALE)

        # ── Stamp persistent trajectory dot every 0.1 s ───────────────
        if i % dot_interval == 0 and 0 <= proj_px < WIDTH and 0 <= proj_py < HEIGHT:
            cv2.circle(traj_layer, (proj_px, proj_py), 2, (160, 160, 160), -1)

        # ── Motion-blur fade (decays toward white) ────────────────────
        trail = (255 - ((255 - trail) * 0.75)).astype(np.uint8)
        if 0 <= proj_px < WIDTH and 0 <= proj_py < HEIGHT:
            cv2.circle(trail, (proj_px, proj_py), 3, (0, 0, 200), -1)

        # Compose: persistent dots underlaid by fading motion-blur
        frame = np.minimum(traj_layer, trail)

        # ── Ground line ───────────────────────────────────────────────
        cv2.line(frame, (0, HEIGHT - 1), (WIDTH - 1, HEIGHT - 1), (80, 120, 40), 2)

        # ── Branch ────────────────────────────────────────────────────
        cv2.line(frame,
                 (branch_x_start, branch_y_px),
                 (branch_x_end,   branch_y_px),
                 (101, 67, 33), 3)

        # ── Shooter ───────────────────────────────────────────────────
        draw_shooter(frame, shooter_px, shooter_py)

        # ── Aim line: dashed, always visible
        # To restore fade: change 'if True' to 'if i < aim_fade_frames'
        #                  and change 'alpha = 1.0' to 'alpha = 1.0 - (i / aim_fade_frames)'
        if True:
            alpha    = 1.0
            colour   = (int(180 * alpha),) * 3
            x0, y0   = gun_tip_px, gun_tip_py
            x1d, y1d = monkey_start_cx, monkey_start_cy
            total    = np.hypot(x1d - x0, y1d - y0)
            dash, gap = 10, 6
            steps = int(total / (dash + gap))
            for s in range(steps):
                t0 = s * (dash + gap) / total
                t1 = min((s * (dash + gap) + dash) / total, 1.0)
                p0 = (int(x0 + t0 * (x1d - x0)), int(y0 + t0 * (y1d - y0)))
                p1 = (int(x0 + t1 * (x1d - x0)), int(y0 + t1 * (y1d - y0)))
                cv2.line(frame, p0, p1, colour, 1, cv2.LINE_AA)

        # ── Monkey ────────────────────────────────────────────────────
        overlay_monkey(frame, monk_cx, monk_cy)

        # ── Velocity arrow on bullet ──────────────────────────────────
        if i < len(t_vals) - 1:
            adx = int((px[i + 1] - px[i]) * SCALE * 2)
            ady = int(-(py[i + 1] - py[i]) * SCALE * 2)
            cv2.arrowedLine(frame,
                            (proj_px, proj_py),
                            (proj_px + adx, proj_py + ady),
                            (200, 0, 0), 2)

        # ── Hit / miss detection (absolute metres) ────────────────────
        # Check minimum distance along the segment from previous frame to current
        # frame, so fast-moving bullets don't slip through the gap between frames.
        bullet_x_m = gun_tip_x_m + px[i]
        bullet_y_m = gun_tip_y_m + py[i]
        rx0 = prev_bx_m - prev_mx_m; ry0 = prev_by_m - prev_my_m
        rx1 = bullet_x_m - tx[i];    ry1 = bullet_y_m - ty[i]
        drx = rx1 - rx0; dry = ry1 - ry0
        denom = drx**2 + dry**2
        if denom > 0:
            s_star = np.clip(-(rx0*drx + ry0*dry) / denom, 0, 1)
            min_dist = np.hypot(rx0 + s_star*drx, ry0 + s_star*dry)
        else:
            min_dist = np.hypot(rx0, ry0)
        prev_bx_m = bullet_x_m; prev_by_m = bullet_y_m
        prev_mx_m = tx[i];      prev_my_m = ty[i]
        if min_dist < min_dist_overall:
            min_dist_overall = min_dist
            min_dist_time    = t_vals[i]

        if min_dist <= HIT_RADIUS:
            cv2.putText(frame, f"HIT!  t = {t_vals[i]:.2f} s",
                        (160, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 180, 0), 3)
            canvas.image(frame, channels="BGR")
            break

        if ty[i] <= 0 and not hit:
            close = min_dist_overall < 0.35
            label = (f"CLOSE CALL!  t = {min_dist_time:.2f} s  (missed by {min_dist_overall:.2f} m)"
                     if close else
                     f"MISS - monkey hit ground at t = {t_vals[i]:.2f} s")
            colour = (0, 140, 255) if close else (0, 0, 200)
            cv2.putText(frame, label, (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
            canvas.image(frame, channels="BGR")
            break

        # Use absolute bullet y (gun_tip_y_m + py[i]) so we track all the way to ground
        if (gun_tip_y_m + py[i]) < 0 and not hit:
            close = min_dist_overall < 0.35
            label = (f"CLOSE CALL!  t = {min_dist_time:.2f} s  (missed by {min_dist_overall:.2f} m)"
                     if close else
                     f"MISS - bullet hit ground at t = {t_vals[i]:.2f} s")
            colour = (0, 140, 255) if close else (0, 0, 200)
            cv2.putText(frame, label, (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
            canvas.image(frame, channels="BGR")
            break

        # Wall-clock timing: sleep only what remains after render overhead
        # Enforce minimum 80ms per frame so Streamlit always renders visibly
        frame_deadline = frame_start + max(dt / playback_speed, 0.08)
        canvas.image(frame, channels="BGR")
        remaining = frame_deadline - time.time()
        if remaining > 0:
            time.sleep(remaining)


# -----------------------------
# Entry point
# -----------------------------
if fire or replay:
    run_simulation()

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
---
*This was inspired by my favorite high school physics demonstration (h/t Mr. John Balaban; AMDG).
I always wanted to build one, but not having a big room and a complicated electromagnetic setup,
I'm excited to be able to recreate and share it virtually — and enhance it to show the
differences that running the experiment on different planets (if it were possible to do so)
would have.*
""")
