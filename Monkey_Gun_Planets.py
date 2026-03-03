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

    - **Projectile speed** — how fast the projectile travels
    - **Target height** — how high up the monkey is
    - **Distance to target** — how far away the monkey is
    - **Frames per second** — animation smoothness
    - **Playback speed** — slow down to see the physics in action (try 10–25%)
    - **Aim offset** — fire above or below the monkey to see what happens
    - **Planet** — change gravity to see how it affects the outcome
    - **Challenge mode 🎯** — give the monkey a "head start" or reaction delay and figure out
      the correct aim offset to still make the hit!

    *Tip: the hunter always aims directly at the monkey. Try different aim offsets
    and planets to explore the physics.*
    """)

# -----------------------------
# Constants
# -----------------------------
WIDTH, HEIGHT    = 600, 400
SCALE            = 25     # pixels per metre
HIT_RADIUS_NORMAL    = 0.05   # strict — any offset misses
HIT_RADIUS_CHALLENGE = 0.16   # tight — requires within ~0.5° of correct answer

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Simulation Parameters")
v0            = st.sidebar.slider("Projectile speed (m/s)", 5, 50, 20)
target_height = st.sidebar.slider("Target height (m)", 1, 15, 10)
distance      = st.sidebar.slider("Distance to target (m)", 5, 20, 15)
fps           = st.sidebar.slider("Frames per second", 10, 60, 24)
playback_pct  = st.sidebar.slider(
    "Playback speed (%)", min_value=5, max_value=100, value=100, step=5,
    format="%d%%"
)
playback_speed   = playback_pct / 100.0
angle_offset_deg = st.sidebar.slider("Aim offset (degrees)", -10.0, 10.0, 0.0, step=0.5)

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

# -----------------------------
# Challenge mode
# -----------------------------
st.sidebar.markdown("---")
challenge_mode = st.sidebar.checkbox("🎯 Challenge mode", value=False)
reaction_delay = 0.0

if challenge_mode:
    reaction_delay = st.sidebar.slider(
        "Monkey reaction (s)   < 0 = lets go early,  > 0 = hesitates",
        min_value=-0.20, max_value=0.30, value=0.20, step=0.05
    )
    if reaction_delay < 0:
        st.sidebar.caption(f"🐵 Monkey lets go **{abs(reaction_delay):.2f}s early** — already falling when the shot is fired!")
    elif reaction_delay == 0.0:
        st.sidebar.caption("🐵 Monkey reacts instantly — same as normal mode")
    else:
        st.sidebar.caption(f"🐵 Monkey hesitates **{reaction_delay:.2f}s** before letting go")

    # Compute compensating offset
    _gun_x = 20 / SCALE; _gun_y = 35 / SCALE
    _dx = distance - _gun_x; _dy = target_height - _gun_y
    _theta0 = np.arctan2(_dy, _dx)
    _vx0    = v0 * np.cos(_theta0)
    _t_meet = _dx / _vx0 if _vx0 > 0 else float('inf')

    if np.isfinite(_t_meet) and _t_meet > abs(reaction_delay):
        _vy_needed = (_dy / _t_meet
                      + gravity * reaction_delay
                      - 0.5 * gravity * reaction_delay**2 / _t_meet)
        _theta_needed   = np.arctan2(_vy_needed, _vx0)
        _offset_needed  = np.rad2deg(_theta_needed - _theta0)

        st.sidebar.markdown("---")
        st.sidebar.markdown("**🧩 Challenge:** adjust the *Aim offset* slider to compensate for the reaction delay, then press Fire!")
        if st.sidebar.checkbox("💡 Reveal answer"):
            st.sidebar.success(f"Compensating aim offset: **{_offset_needed:+.1f}°**")
        else:
            st.sidebar.caption("Check the box above to reveal the answer.")
    else:
        st.sidebar.warning("Bullet can't reach monkey in time — try increasing speed or reducing distance.")

hit_radius = HIT_RADIUS_CHALLENGE if challenge_mode else HIT_RADIUS_NORMAL

st.sidebar.markdown("""
---
*On mobile, tap **>** at the top left to open the sidebar.*
""")

if 'run_sim' not in st.session_state:
    st.session_state.run_sim = False

if st.button("🔫 Fire! / Replay", use_container_width=False):
    st.session_state.run_sim = True

# -----------------------------
# Worked math expander (live values)
# -----------------------------
_gx = 20 / SCALE   # gun tip x in metres
_gy = 35 / SCALE   # gun tip y in metres
_dx  = distance - _gx
_dy  = target_height - _gy
_theta_direct = np.arctan2(_dy, _dx)
_theta_fired  = _theta_direct + np.deg2rad(angle_offset_deg)
_vx   = v0 * np.cos(_theta_fired)
_vy   = v0 * np.sin(_theta_fired)
_t_meet = _dx / _vx if _vx > 0 else float('inf')
_t_ground = np.sqrt(2 * target_height / gravity)
_min_speed = (_dx / _t_ground) / np.cos(_theta_direct)
_bullet_y_meet = _gy + _vy * _t_meet - 0.5 * gravity * _t_meet**2

# Monkey height at meeting time — accounts for delay
if np.isfinite(_t_meet):
    if reaction_delay >= 0:
        _fall = max(_t_meet - reaction_delay, 0)
        _monkey_y_meet = target_height - 0.5 * gravity * _fall**2
    else:
        _fall = _t_meet - reaction_delay   # reaction_delay negative → larger fall
        _monkey_y_meet = target_height - 0.5 * gravity * _fall**2
else:
    _monkey_y_meet = target_height

# Pre-build monkey equation to avoid LaTeX escaping issues inside f-string
if reaction_delay == 0:
    _monkey_eq = (
        f"$$y_{{\\text{{monkey}}}}(t) = {target_height} - "
        f"\\tfrac{{1}}{{2}} \\times {gravity:.1f}\\,t^2$$"
    )
elif reaction_delay > 0:
    _monkey_eq = (
        f"$$y_{{\\text{{monkey}}}}(t) = "
        f"\\begin{{cases}} "
        f"{target_height} & t < {reaction_delay:.2f} \\\\\\\\ "
        f"{target_height} - \\tfrac{{1}}{{2}} \\times {gravity:.1f}(t - {reaction_delay:.2f})^2 "
        f"& t \\geq {reaction_delay:.2f} "
        f"\\end{{cases}}$$"
    )
else:
    _monkey_eq = (
        f"$$y_{{\\text{{monkey}}}}(t) = {target_height} - "
        f"\\tfrac{{1}}{{2}} \\times {gravity:.1f}\\,"
        f"(t + {abs(reaction_delay):.2f})^2$$"
    )

with st.expander("📐 Show your work - how would you solve this on paper?"):
    _delay_label = (
        f"jumps **{abs(reaction_delay):.2f}s early**" if reaction_delay < 0
        else ("reacts **instantly**" if reaction_delay == 0
              else f"hesitates **{reaction_delay:.2f}s**")
    )
    st.markdown(f"""
#### Step 1 — Find the aim angle

The hunter aims directly at the monkey — draw a right triangle:
- Horizontal leg = **{distance} − {_gx:.1f} = {_dx:.1f} m**
- Vertical leg = **{target_height} − {_gy:.1f} = {_dy:.1f} m**

$$\\theta_{{\\text{{direct}}}} = \\arctan\\!\\left(\\frac{{{_dy:.1f}}}{{{_dx:.1f}}}\\right) = {np.rad2deg(_theta_direct):.1f}°$$

{"*(Aim offset of **" + f"{angle_offset_deg:+.1f}°**" + f" applied → actual firing angle = **{np.rad2deg(_theta_fired):.1f}°**)*" if angle_offset_deg != 0 else ""}

---

#### Step 2 — Break the muzzle velocity into components

$$v_x = {v0} \\times \\cos({np.rad2deg(_theta_fired):.1f}°) = {_vx:.2f} \\text{{ m/s}}$$

$$v_y = {v0} \\times \\sin({np.rad2deg(_theta_fired):.1f}°) = {_vy:.2f} \\text{{ m/s}}$$

---

#### Step 3 — Equations of motion

**Bullet** (from gun tip at height {_gy:.1f} m):

$$y_{{\\text{{bullet}}}}(t) = {_gy:.1f} + {_vy:.2f}\\,t - \\tfrac{{1}}{{2}} \\times {gravity:.1f}\\,t^2$$

**Monkey** ({_delay_label}, starting at {target_height} m):

{_monkey_eq}

---

#### Step 4 — When does the bullet reach the monkey's x?

$$t_{{\\text{{meet}}}} = \\frac{{{_dx:.1f}}}{{{_vx:.2f}}} = {_t_meet:.3f}\\text{{ s}}$$

---

#### Step 5 — Check heights at t = {_t_meet:.3f} s

$$y_{{\\text{{bullet}}}} = {_bullet_y_meet:.2f}\\text{{ m}}$$

$$y_{{\\text{{monkey}}}} = {_monkey_y_meet:.2f}\\text{{ m}}$$

{("✅ **Both heights match — gravity cancels perfectly!**"
  if reaction_delay == 0 and abs(angle_offset_deg) < 0.1
  else "⚠️ **Reaction delay breaks the cancellation** — the monkey's −½g(t−Δt)² no longer matches the bullet's −½gt², so they diverge unless you compensate with an aim offset."
  if reaction_delay != 0 and abs(angle_offset_deg) < 0.1
  else f"{'✅ Heights close — this should be a hit!' if abs(_bullet_y_meet - _monkey_y_meet) < 0.35 else '⚠️ Heights differ by ' + f'{abs(_bullet_y_meet-_monkey_y_meet):.2f}m' + ' — keep adjusting.'}"
  if challenge_mode
  else "⚠️ **Aim offset applied** — heights no longer match.")}

---

#### Why does gravity cancel out? (and when doesn't it?)

In the standard case both equations contain **−½ × g × t²** — gravity cancels.
{"**With a reaction delay**, the monkey's term becomes **−½g(t−Δt)²**, which expands differently. The delay breaks the cancellation: the hunter must aim **lower** (hesitation) or **more steeply** (early jump) to compensate." if reaction_delay != 0 else "This holds on any planet — try changing gravity to verify!"}

---

#### Minimum speed to guarantee a hit

$$t_{{\\text{{ground}}}} = \\sqrt{{\\frac{{2 \\times {target_height}}}{{{gravity:.1f}}}}} = {_t_ground:.3f}\\text{{ s}}$$

$$v_{{\\text{{min}}}} = \\frac{{{_dx:.1f}\\,/\\,{_t_ground:.3f}}}{{\\cos({np.rad2deg(_theta_direct):.1f}°)}} = {_min_speed:.1f}\\text{{ m/s}}$$

{"✅ Current speed **" + str(v0) + " m/s** is fast enough." if v0 >= _min_speed else "⚠️ Current speed **" + str(v0) + " m/s** is too slow — the monkey hits the ground first!"}
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
    st.sidebar.info("No assets/monkey.png found — using a coloured dot for the monkey.")

# -----------------------------
# Drawing helpers
# -----------------------------
def draw_shooter(frame, x, y):
    cv2.line(frame,   (x, y),      (x, y - 30),      (0, 0, 0), 2)
    cv2.circle(frame, (x, y - 40), 8,                 (0, 0, 0), 2)
    cv2.line(frame,   (x, y - 25), (x + 20, y - 35), (0, 0, 0), 3)


def overlay_monkey(frame, cx, cy):
    if monkey_img is not None:
        x1 = cx - MONKEY_W // 2;  y1 = cy - MONKEY_H // 2
        x2 = x1 + MONKEY_W;       y2 = y1 + MONKEY_H
        fx1 = max(x1, 0); fy1 = max(y1, 0)
        fx2 = min(x2, WIDTH); fy2 = min(y2, HEIGHT)
        if fx2 <= fx1 or fy2 <= fy1:
            return
        sx1 = fx1 - x1; sy1 = fy1 - y1
        sx2 = sx1 + (fx2 - fx1); sy2 = sy1 + (fy2 - fy1)
        sprite = monkey_img[sy1:sy2, sx1:sx2]
        if sprite.shape[2] == 4:
            alpha = sprite[:, :, 3:4] / 255.0
            frame[fy1:fy2, fx1:fx2] = (
                alpha * sprite[:, :, :3] +
                (1 - alpha) * frame[fy1:fy2, fx1:fx2]
            ).astype(np.uint8)
        else:
            frame[fy1:fy2, fx1:fx2] = sprite[:, :, :3]
    else:
        cv2.circle(frame, (cx, cy), 10, (0, 140, 255), -1)
        cv2.circle(frame, (cx, cy), 10, (0, 0, 0), 1)
    cv2.circle(frame, (cx, cy), 3, (0, 200, 0), -1)


# -----------------------------
# Physics
# -----------------------------
def simulate_positions(v0, target_height, distance, g, dt,
                        angle_offset_deg=0.0, t_max=5.0,
                        origin_x_m=0.0, origin_y_m=0.0,
                        reaction_delay=0.0):
    """
    reaction_delay > 0 : monkey hesitates, starts falling at t = delay
    reaction_delay < 0 : monkey jumps early, already falling |delay| s before t=0
    reaction_delay = 0 : instant reaction (standard case)
    """
    dx_to_monkey = distance      - origin_x_m
    dy_to_monkey = target_height - origin_y_m
    theta = np.arctan2(dy_to_monkey, dx_to_monkey) + np.deg2rad(angle_offset_deg)
    vx    = v0 * np.cos(theta)
    vy    = v0 * np.sin(theta)

    t_vals = np.arange(0, t_max, dt)

    px = vx * t_vals
    py = vy * t_vals - 0.5 * g * t_vals**2

    tx = np.full_like(t_vals, distance)

    if reaction_delay > 0:
        ty = np.where(
            t_vals < reaction_delay,
            float(target_height),
            target_height - 0.5 * g * (t_vals - reaction_delay)**2
        )
    elif reaction_delay < 0:
        # already in free-fall: effective fall time = t - reaction_delay (negative → adds)
        ty = target_height - 0.5 * g * (t_vals - reaction_delay)**2
    else:
        ty = target_height - 0.5 * g * t_vals**2

    return t_vals, px, py, tx, ty


# -----------------------------
# Simulation renderer
# -----------------------------
def run_simulation(canvas=None):
    dt = 1.0 / fps

    shooter_px = 0
    shooter_py = HEIGHT
    gun_tip_px = shooter_px + 20
    gun_tip_py = shooter_py - 35
    gun_tip_x_m = gun_tip_px / SCALE
    gun_tip_y_m = (HEIGHT - gun_tip_py) / SCALE

    t_vals, px, py, tx, ty = simulate_positions(
        v0, target_height, distance, gravity, dt, angle_offset_deg,
        origin_x_m=gun_tip_x_m, origin_y_m=gun_tip_y_m,
        reaction_delay=reaction_delay
    )

    if canvas is None:
        canvas = st.empty()
    hit    = False

    branch_y_px    = HEIGHT - int(target_height * SCALE)
    branch_x_start = int(distance * SCALE) - 30
    branch_x_end   = int(distance * SCALE) + 30
    monkey_start_cx = int(distance * SCALE)
    monkey_start_cy = HEIGHT - int(target_height * SCALE)

    aim_fade_frames = int(0.5 * fps / playback_speed)
    traj_layer      = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
    trail           = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
    dot_interval    = max(1, int(0.1 * fps))

    prev_bx_m = gun_tip_x_m;  prev_by_m = gun_tip_y_m
    prev_mx_m = float(distance); prev_my_m = float(target_height)
    min_dist_overall = 999.0
    min_dist_time    = 0.0

    # Static first frame
    first_frame = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
    cv2.line(first_frame, (0, HEIGHT-1), (WIDTH-1, HEIGHT-1), (80, 120, 40), 2)
    cv2.line(first_frame, (branch_x_start, branch_y_px),
             (branch_x_end, branch_y_px), (101, 67, 33), 3)
    draw_shooter(first_frame, shooter_px, shooter_py)
    overlay_monkey(first_frame, monkey_start_cx, monkey_start_cy)
    canvas.image(first_frame, channels="BGR")
    time.sleep(1.0)  # hold static frame — gives Streamlit time to settle

    for i in range(len(t_vals)):
        frame_start = time.time()

        proj_px = gun_tip_px + int(px[i] * SCALE)
        proj_py = gun_tip_py - int(py[i] * SCALE)
        monk_cx = int(tx[i] * SCALE)
        monk_cy = HEIGHT - int(ty[i] * SCALE)

        # Persistent trajectory dot
        if i % dot_interval == 0 and 0 <= proj_px < WIDTH and 0 <= proj_py < HEIGHT:
            cv2.circle(traj_layer, (proj_px, proj_py), 2, (160, 160, 160), -1)

        # Motion-blur trail
        trail = (255 - ((255 - trail) * 0.75)).astype(np.uint8)
        if 0 <= proj_px < WIDTH and 0 <= proj_py < HEIGHT:
            cv2.circle(trail, (proj_px, proj_py), 3, (0, 0, 200), -1)

        frame = np.minimum(traj_layer, trail)

        # Ground
        cv2.line(frame, (0, HEIGHT-1), (WIDTH-1, HEIGHT-1), (80, 120, 40), 2)

        # Branch
        cv2.line(frame, (branch_x_start, branch_y_px),
                 (branch_x_end, branch_y_px), (101, 67, 33), 3)

        # Shooter
        draw_shooter(frame, shooter_px, shooter_py)

        # Aim line: dashed, always visible
        # To restore fade: change 'if True' to 'if i < aim_fade_frames'
        #                  and 'alpha = 1.0' to 'alpha = 1.0 - (i / aim_fade_frames)'
        if True:
            alpha  = 1.0
            colour = (int(180 * alpha),) * 3
            x0, y0 = gun_tip_px, gun_tip_py
            x1d, y1d = monkey_start_cx, monkey_start_cy
            total  = np.hypot(x1d - x0, y1d - y0)
            dash, gap = 10, 6
            steps  = int(total / (dash + gap))
            for s in range(steps):
                t0 = s * (dash + gap) / total
                t1 = min((s * (dash + gap) + dash) / total, 1.0)
                p0 = (int(x0 + t0*(x1d-x0)), int(y0 + t0*(y1d-y0)))
                p1 = (int(x0 + t1*(x1d-x0)), int(y0 + t1*(y1d-y0)))
                cv2.line(frame, p0, p1, colour, 1, cv2.LINE_AA)

        # Monkey
        overlay_monkey(frame, monk_cx, monk_cy)

        # Velocity arrow on bullet
        if i < len(t_vals) - 1:
            adx = int((px[i+1] - px[i]) * SCALE * 2)
            ady = int(-(py[i+1] - py[i]) * SCALE * 2)
            cv2.arrowedLine(frame, (proj_px, proj_py),
                            (proj_px+adx, proj_py+ady), (200, 0, 0), 2)

        # Hit / miss detection — segment-based for accuracy
        bullet_x_m = gun_tip_x_m + px[i]
        bullet_y_m = gun_tip_y_m + py[i]
        rx0 = prev_bx_m - prev_mx_m;  ry0 = prev_by_m - prev_my_m
        rx1 = bullet_x_m - tx[i];     ry1 = bullet_y_m - ty[i]
        drx = rx1 - rx0; dry = ry1 - ry0
        denom = drx**2 + dry**2
        if denom > 0:
            s_star   = np.clip(-(rx0*drx + ry0*dry) / denom, 0, 1)
            min_dist = np.hypot(rx0 + s_star*drx, ry0 + s_star*dry)
        else:
            min_dist = np.hypot(rx0, ry0)
        prev_bx_m = bullet_x_m; prev_by_m = bullet_y_m
        prev_mx_m = tx[i];      prev_my_m = ty[i]
        if min_dist < min_dist_overall:
            min_dist_overall = min_dist
            min_dist_time    = t_vals[i]

        if min_dist <= hit_radius:
            if challenge_mode:
                cv2.putText(frame, f"HIT!  t = {t_vals[i]:.2f} s",
                            (160, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 0), 3)
                cv2.putText(frame, "Challenge solved!",
                            (170, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 0), 2)
            else:
                cv2.putText(frame, f"HIT!  t = {t_vals[i]:.2f} s",
                            (160, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 180, 0), 3)
            canvas.image(frame, channels="BGR")
            break

        if ty[i] <= 0 and not hit:
            close = min_dist_overall < 0.35
            _idx = min(int(min_dist_time * fps), len(t_vals)-1)
            _by  = gun_tip_y_m + py[_idx]
            _my  = ty[_idx]
            _aim_hint = ("aim lower" if _by > _my else "aim higher") if angle_offset_deg != 0 or challenge_mode else ""
            col = (0, 140, 255) if close else (0, 0, 200)
            if close:
                cv2.putText(frame, f"CLOSE CALL!  t = {min_dist_time:.2f} s  (missed by {min_dist_overall:.2f} m)",
                            (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            else:
                cv2.putText(frame, f"MISS :( - monkey hit ground at t = {t_vals[i]:.2f} s",
                            (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            if _aim_hint:
                cv2.putText(frame, f"^ {_aim_hint}",
                            (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            canvas.image(frame, channels="BGR")
            break

        if (gun_tip_y_m + py[i]) < 0 and not hit:
            close = min_dist_overall < 0.35
            _idx = min(int(min_dist_time * fps), len(t_vals)-1)
            _by  = gun_tip_y_m + py[_idx]
            _my  = ty[_idx]
            _aim_hint = ("aim lower" if _by > _my else "aim higher") if angle_offset_deg != 0 or challenge_mode else ""
            col = (0, 140, 255) if close else (0, 0, 200)
            if close:
                cv2.putText(frame, f"CLOSE CALL!  t = {min_dist_time:.2f} s  (missed by {min_dist_overall:.2f} m)",
                            (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            else:
                cv2.putText(frame, f"MISS :( - bullet hit ground at t = {t_vals[i]:.2f} s",
                            (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            if _aim_hint:
                cv2.putText(frame, f"^ {_aim_hint}",
                            (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            canvas.image(frame, channels="BGR")
            break

        # Wall-clock timing — enforce 80ms minimum so frames are always visible
        frame_deadline = frame_start + max(dt / playback_speed, 0.15)
        canvas.image(frame, channels="BGR")
        remaining = frame_deadline - time.time()
        if remaining > 0:
            time.sleep(remaining)


# -----------------------------
# Entry point
# -----------------------------
# Canvas placeholder created at top level so Streamlit never loses it
canvas_placeholder = st.empty()

if st.session_state.get('run_sim', False):
    st.session_state.run_sim = False
    run_simulation(canvas_placeholder)

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
---
*This was inspired by my favorite high school physics demonstration (h/t Mr. John Balaban; AMDG).
I always wanted to build one, but not having a big room and a complicated electromagnetic setup,
I'm excited to be able to recreate and share it virtually — and (a) enhance it to show the
differences that running the experiment on different planets (if it were possible to do so)
would have and (b) add a challenge mode to add some variability (h/t Mike R. for that idea).*
""")
