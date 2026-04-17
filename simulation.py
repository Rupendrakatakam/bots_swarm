"""
simulation.py — Matplotlib runner for the navigation pipeline.
Imports from integration.py. Keep this file for all visualization/debug.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle

from hybrid_astar import HybridAStar
from apf import ImprovedAPF, ImprovedAPFParams, CircleObstacle, RectObstacle, DynamicObstacle
from apf_orca import (
    FleetManager, DiffDriveConfig, CameraBlob,
    RobotController, WheelCommand
)


def run_simulation(fleet, agent_data, camera_blobs=None, max_steps=3000, dt=0.1):
    """Run pipeline, return histories = {robot_id: [(x,y,yaw), ...]}"""
    camera_blobs = camera_blobs or []
    histories = {rid: [] for rid in fleet.robots}

    for rid, ctrl in fleet.robots.items():
        p = ctrl.state.pose
        histories[rid].append((p.x, p.y, p.yaw))

    circle_center = np.array([8.0, 8.0])
    circle_radius = 4.0
    blob_period = 400
    blob_history = []

    for step in range(max_steps):
        angle = 2.0 * math.pi * (step % blob_period) / float(blob_period)
        blob_pos = circle_center + circle_radius * np.array([math.cos(angle), math.sin(angle)])
        blob_history.append((float(blob_pos[0]), float(blob_pos[1])))

        # Angular velocity ω = 2π / (T_period × dt) rad/s
        # Tangential velocity: d/dt [r·cos(ωt)] = -r·ω·sin(ωt)
        #                      d/dt [r·sin(ωt)] = +r·ω·cos(ωt)
        omega_blob = 2.0 * math.pi / (blob_period * dt)  # rad/s
        blob_vx = -circle_radius * omega_blob * math.sin(angle)
        blob_vy =  circle_radius * omega_blob * math.cos(angle)

        dynamic_camera_blobs = [CameraBlob(
            x=float(blob_pos[0]),
            y=float(blob_pos[1]),
            radius=1.3,
            priority=1.5,
            vx=float(blob_vx),    # velocity for predictive EmergencyBrake
            vy=float(blob_vy),    # velocity for predictive EmergencyBrake
        )]

        commands = fleet.tick_all(camera_blobs=dynamic_camera_blobs)
        all_reached = all(fleet.robots[rid].reached_goal for rid in fleet.robots)

        for rid, cmd in commands.items():
            ctrl = fleet.robots[rid]
            p    = ctrl.state.pose
            new_yaw = p.yaw + cmd.omega * dt
            new_x   = p.x   + cmd.v * math.cos(new_yaw) * dt
            new_y   = p.y   + cmd.v * math.sin(new_yaw) * dt
            fleet.update_state(rid, new_x, new_y, new_yaw,
                               cmd.v * math.cos(new_yaw),
                               cmd.v * math.sin(new_yaw))
            histories[rid].append((new_x, new_y, new_yaw))

        if all_reached:
            print(f"All goals reached at step {step}.")
            break
    else:
        print(f"Max steps ({max_steps}) reached.")

    # Print jitter diagnostics for every robot
    for rid, ctrl in fleet.robots.items():
        ctrl.logger.jitter_report()

    return histories, blob_history


def build_figure(fleet, agent_data, global_paths, histories,
                  world_size=20.0, static_circles=None, static_rects=None,
                  camera_blobs=None, blob_history=None):

    static_circles = static_circles or []
    static_rects = static_rects or []
    camera_blobs = camera_blobs or []
    blob_history = blob_history or []

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, world_size)
    ax.set_ylim(0, world_size)
    ax.set_aspect('equal')
    ax.set_title('Multi-Agent Navigation: APF + NH-ORCA Pipeline', fontsize=13)
    ax.set_xticks(np.arange(0, world_size + 1, 1))
    ax.set_yticks(np.arange(0, world_size + 1, 1))
    ax.grid(True, linestyle=':', alpha=0.5)

    # ── Solid boundary walls ────────────────────────────────────────────
    wall_color = 'dimgray'
    wall_alpha = 0.6
    ax.add_patch(Rectangle((0.0, 0.0), world_size, 0.3,
        color=wall_color, alpha=wall_alpha, zorder=2))
    ax.add_patch(Rectangle((0.0, world_size - 0.3), world_size, 0.3,
        color=wall_color, alpha=wall_alpha, zorder=2))
    ax.add_patch(Rectangle((0.0, 0.0), 0.3, world_size,
        color=wall_color, alpha=wall_alpha, zorder=2))
    ax.add_patch(Rectangle((world_size - 0.3, 0.0), 0.3, world_size,
        color=wall_color, alpha=wall_alpha, zorder=2))

    # ── Static circle obstacles (solid fill, no dashed overlay) ─────────
    for obs in static_circles:
        ax.add_patch(Circle((obs.cx, obs.cy), obs.radius,
            color='red', alpha=0.45, zorder=2))

    # ── Camera blob history trail ────────────────────────────────────────
    if len(blob_history) > 1:
        ax.plot([p[0] for p in blob_history], [p[1] for p in blob_history],
            color='darkviolet', linestyle=':', linewidth=1.0, alpha=0.35, zorder=1)

    # ── Dynamic blob patch ───────────────────────────────────────────────
    blob_patch_container = [None]
    if len(blob_history) > 0:
        bx, by = blob_history[0]
        blob_patch_container[0] = Circle((bx, by), 1.3, color='darkviolet', alpha=0.7, zorder=4)
        ax.add_patch(blob_patch_container[0])

    # ── A* paths ─────────────────────────────────────────────────────────
    for rid, path in global_paths.items():
        color = agent_data[rid]['color']
        ax.plot([p[0] for p in path], [p[1] for p in path],
            color=color, linestyle='--', linewidth=1.5, alpha=0.4, zorder=1)

    # ── Goals ────────────────────────────────────────────────────────────
    for rid, data in agent_data.items():
        gx, gy = data['goal']
        color = data['color']
        ax.scatter(gx, gy, s=220, c=color, marker='*', zorder=5)
        ax.annotate(f"Goal {rid}", (gx, gy),
            textcoords='offset points', xytext=(5, 5), fontsize=8)

    # ── Start markers ────────────────────────────────────────────────────
    for rid, data in agent_data.items():
        sx, sy, _ = data['start']
        ax.scatter(sx, sy, s=120, c=data['color'], marker='s',
            zorder=5, alpha=0.5)

    r = fleet.cfg.robot_radius
    L = fleet.cfg.wheel_base
    wl = L * 0.6

    # ── Social bubble radius ─────────────────────────────────────────────
    social_bubble_factor = getattr(fleet.cfg, 'social_bubble_factor', 0.20)
    social_r = r * (1.0 + social_bubble_factor)

    body_patches = {}
    heading_lines = {}
    trail_lines = {}
    left_wheels = {}
    right_wheels = {}
    bubble_patches = {}

    for rid, data in agent_data.items():
        color = data['color']
        x0, y0, _ = histories[rid][0]
        body_patches[rid] = Circle((x0, y0), r, color=color, alpha=0.6, zorder=4)
        ax.add_patch(body_patches[rid])
        heading_lines[rid], = ax.plot([], [], 'k-', linewidth=2, zorder=5)
        trail_lines[rid], = ax.plot([], [], color=color, linewidth=1.8,
            alpha=0.65, zorder=3)
        left_wheels[rid], = ax.plot([], [], 'k-', linewidth=4,
            solid_capstyle='round', zorder=5)
        right_wheels[rid], = ax.plot([], [], 'k-', linewidth=4,
            solid_capstyle='round', zorder=5)
        bubble_patches[rid] = Circle((x0, y0), social_r, fill=False,
            color=color, linewidth=1.5, linestyle='--', alpha=0.45, zorder=3)
        ax.add_patch(bubble_patches[rid])

    step_text = ax.text(0.02, 0.97, '', transform=ax.transAxes,
        fontsize=10, va='top')
    speed_text = ax.text(0.02, 0.93, '', transform=ax.transAxes,
        fontsize=8, va='top', color='gray')

    def update(frame):
        artists = [step_text, speed_text]
        step_text.set_text(f'Step: {frame}')

        speeds = []
        for rid in agent_data:
            hist = histories[rid]
            f = min(frame, len(hist) - 1)
            x, y, theta = hist[f]

            trail_lines[rid].set_data([p[0] for p in hist[:f+1]],
                                      [p[1] for p in hist[:f+1]])
            body_patches[rid].center = (x, y)
            heading_lines[rid].set_data([x, x + r*math.cos(theta)],
                                        [y, y + r*math.sin(theta)])

            lx = x - (L/2)*math.sin(theta); ly = y + (L/2)*math.cos(theta)
            rx = x + (L/2)*math.sin(theta); ry = y - (L/2)*math.cos(theta)

            left_wheels[rid].set_data(
                [lx - wl*math.cos(theta), lx + wl*math.cos(theta)],
                [ly - wl*math.sin(theta), ly + wl*math.sin(theta)])
            right_wheels[rid].set_data(
                [rx - wl*math.cos(theta), rx + wl*math.cos(theta)],
                [ry - wl*math.sin(theta), ry + wl*math.sin(theta)])

            bubble_patches[rid].center = (x, y)

            debug = fleet.robots[rid].get_debug_info()
            v_path = debug['v_path']
            v_pref = debug['v_pref']
            v_safe = debug['v_safe']

            artists.extend([
                trail_lines[rid], body_patches[rid],
                heading_lines[rid], left_wheels[rid],
                right_wheels[rid], bubble_patches[rid],
            ])

            if f > 0:
                dx = hist[f][0] - hist[f-1][0]
                dy = hist[f][1] - hist[f-1][1]
                spd = math.hypot(dx, dy) / 0.1
                speeds.append(f"{rid}:{spd:.1f}")

        speed_text.set_text(' '.join(speeds) + ' m/s')

        if blob_patch_container[0] is not None and frame < len(blob_history):
            bx, by = blob_history[frame]
            blob_patch_container[0].center = (bx, by)
            artists.append(blob_patch_container[0])

        return artists

    return fig, update, max(len(h) for h in histories.values())


if __name__ == '__main__':

    # =======================================================================
    # 1. CENTRAL TUNING PANEL
    # =======================================================================
    # Tune these parameters to adjust the swarm's behavior.
    
    # --- A. Differential Drive & Physical Limits ---
    cfg = DiffDriveConfig(
        robot_radius      = 1.0,  # [m] Physical radius. Increase: larger safety zones, harder to fit in gaps.
        wheel_base        = 0.8,  # [m] Distance between wheels.
        max_linear_speed  = 2.0,  # [m/s] Max forward speed. Increase: faster arrival. Decrease: safer navigation.
        max_angular_speed = 5.0,  # [rad/s] Max spin. Increase: snappier turns. Decrease: wider, smoother turns.
        max_wheel_speed   = 3.0,  # [m/s] Physical motor limit. Couples linear/angular speeds.
        max_linear_accel  = 2.0,  # [m/s²] Motor acceleration. Increase: reaches top speed faster.
        max_linear_decel  = 3.0,  # [m/s²] Motor braking. Increase: harder stops. Decrease: glides longer.
        max_angular_accel = 4.0,  # [rad/s²] Spin acceleration. 
        tracking_error    = 0.25, # [m] Swarm social bubble (ORCA). Decrease: robots pass closer. Increase: yields earlier.
        orientation_time  = 1.0,  # [s] Heading correction time. Decrease: faster response. Increase: smoother rotation.
        time_horizon      = 3.0,  # [s] ORCA prediction. Increase: dodges swarm earlier, but path deflects more.
        neighbor_dist     = 6.0,  # [m] Swarm sensing range. Increase: coordinates with distant robots.
        sim_dt            = 0.1,  # [s] 10Hz control loop.
    )

    # --- B. Artificial Potential Field (APF) - Emergency Avoidance ---
    # Controls how the robot dodges unknown camera blobs and static walls.
    p = ImprovedAPFParams(
        k_att        = 1.9,   # Pull to A* path. Increase: sticks tighter to path. Decrease: easily pushed off path.
        eta          = 14.0,  # Repulsion strength. Increase: harder push away. Decrease: softer course corrections.
        rho_0        = 3.0,   # [m] Repulsion zone. Increase: starts dodging earlier. Decrease: ignores obstacles until close.
        max_force    = 12.0,  # Force ceiling. Increase: allows violent jerks to escape. Decrease: caps force for smoothness.
        vortex_gain  = 0.15,  # Tangential sliding. Increase: slides perpendicularly around blobs. Decrease: less jitter.
        robot_radius = cfg.robot_radius,
        
        # GNRO and Local Minima Escape (Eq. 4-6)
        n_reg        = 0.5,   # GNRO regulation. Adjusts repulsion curve shape near goal.
        step_length  = 0.2,   # Normal step size expected.
        beta_stuck   = 3.0,   # Stuck detection threshold. Decrease: detects local minima faster.
        stuck_window = 8,     # Ticks to declare stuck. Decrease: breaks out of local minima sooner.
        beta1        = 2.0,   # Virtual target X amplitude. Increase: wider escape arcs.
        beta2        = 2.0,   # Virtual target Y amplitude. Increase: wider escape arcs.
        n_vt         = 1.0,   # Escape oscillation frequency.
        virtual_target_duration = 20, # [ticks] Time to chase virtual target.
    )

    # --- C. Path Following & Smoothing ---
    FILTER_ALPHA     = 0.5  # Final velocity command EMA smoothing. Decrease: smoother commands but more lag.
    GOAL_TOLERANCE   = 0.5   # [m] Goal reach radius. Increase: finishes earlier if A* waypoints are sparse.
    LOOKAHEAD_WINDOW = 25    # A* waypoints to scan. Increase: recovers from large detours better.
    CARROT_STEPS     = 10    # Target waypoint index offset. Increase: smoother following, cuts corners more.

    # =======================================================================

    # ── 2. Obstacles ───────────────────────────────────────────────────────
    # Internal static obstacles for A* path planning AND APF emergency backup
    INTERNAL_CIRCLES = [CircleObstacle(8.0, 8.0, 1.5)]  # Static pillar near diagonal path
    INTERNAL_RECTS   = []  # No rect obstacles as requested

    # Boundary walls for A*
    BOUNDARY_RECTS = [
        RectObstacle(-1.0, -1.0, 21.0,  0.0),
        RectObstacle(-1.0, 20.0, 21.0, 21.0),
        RectObstacle(-1.0,  0.0,  0.0, 20.0),
        RectObstacle(20.0,  0.0, 21.0, 20.0),
    ]
    ALL_RECTS_FOR_ASTAR = INTERNAL_RECTS + BOUNDARY_RECTS

    apf = ImprovedAPF(
        p,
        static_circles=INTERNAL_CIRCLES,
        static_rects=INTERNAL_RECTS,
    )

    # ── 3. Agent definitions ────────────────────────────────────────────────
    # Two robots - Robot A will pass near camera blob at (6,6) before static circle at (8,8)
    agent_data = {
        'A': {'start': (2.0, 5.0, math.pi/4),'goal': (19.0, 18.0), 'color': 'royalblue'},
        'B': {'start': (2.0, 15.0, math.pi/2),'goal': (15.0,  5.0), 'color': 'seagreen'},
        'C': {'start': (2.0, 10.0, 0.0),'goal': (19.0,  14.0), 'color': 'magenta'},
    }

    # ── 4. Build fleet ─────────────────────────────────────────────────────
    fleet = FleetManager(cfg, apf)
    global_paths = {}

    for rid, data in agent_data.items():
        robot = fleet.add_robot(
            rid,
            filter_alpha     = FILTER_ALPHA,
            goal_tolerance   = GOAL_TOLERANCE,
            lookahead_window = LOOKAHEAD_WINDOW,
            carrot_steps     = CARROT_STEPS,
        )
        sx, sy, stheta = data['start']
        gx, gy = data['goal']

        fleet.update_state(rid, sx, sy, stheta, 0.0, 0.0)
        fleet.set_goal(rid, gx, gy)

        print(f"Planning A* for {rid}...")
        h_astar = HybridAStar(
            start_pose = data['start'],
            goal_pose  = (gx, gy, 0.0),
            obstacles  = [(c.cx, c.cy, c.radius) for c in INTERNAL_CIRCLES],
            rect_obstacles = [
                ((r.bounds[0] + r.bounds[2]) / 2.0,
                 (r.bounds[1] + r.bounds[3]) / 2.0,
                 r.bounds[2] - r.bounds[0],
                 r.bounds[3] - r.bounds[1],
                 0.0)
                for r in ALL_RECTS_FOR_ASTAR   # A* sees boundaries, APF does not
            ],
            robot_radius = cfg.robot_radius,
        )
        result = h_astar.find_path()
        if result and result[0]:
            print(f"  Path found ({len(result[0])} waypoints)")
            robot.path.set_path(result[0])
            global_paths[rid] = result[0]
        else:
            print(f"  WARNING: No path found, using straight line")
            robot.path.set_straight_line_path((sx, sy), (gx, gy), n_steps=100)
            global_paths[rid] = robot.path.waypoints

    # ── 5. Camera blobs ─────────────────────────────────────────────────────
    # Dynamic blob moves in circular motion around center (8,8) with radius 4
    camera_blobs = [CameraBlob(x=12.0, y=8.0, radius=1.3, priority=1.5)]  # Starting at right side of circle

    # ── 6. Simulate ─────────────────────────────────────────────────────────
    print("Running simulation...")
    histories, blob_history = run_simulation(
        fleet        = fleet,
        agent_data   = agent_data,
        camera_blobs = camera_blobs,
        max_steps    = 3000,
        dt           = cfg.sim_dt,
    )

    # ── 7. Animate ──────────────────────────────────────────────────────────
    fig, update_fn, n_frames = build_figure(
        fleet          = fleet,
        agent_data     = agent_data,
        global_paths   = global_paths,
        histories      = histories,
        world_size     = 20.0,
        static_circles = INTERNAL_CIRCLES,
        static_rects   = INTERNAL_RECTS,
        camera_blobs   = camera_blobs,
        blob_history   = blob_history,
    )

    ani = animation.FuncAnimation(
        fig, update_fn,
        frames=n_frames, interval=25, blit=True, repeat=False
    )
    plt.tight_layout()
    plt.show()