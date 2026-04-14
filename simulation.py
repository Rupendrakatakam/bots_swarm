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
from apf import APF, CircleObstacle, RectObstacle, DynamicObstacle
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

    for step in range(max_steps):
        commands = fleet.tick_all(camera_blobs=camera_blobs)
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

    return histories


def build_figure(fleet, agent_data, global_paths, histories,
                 world_size=20.0, static_circles=None, static_rects=None,
                 camera_blobs=None):

    static_circles = static_circles or []
    static_rects   = static_rects   or []
    camera_blobs   = camera_blobs   or []

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, world_size)
    ax.set_ylim(0, world_size)
    ax.set_aspect('equal')
    ax.set_title('Multi-Agent Navigation: APF + NH-ORCA Pipeline', fontsize=13)
    ax.set_xticks(np.arange(0, world_size + 1, 1))
    ax.set_yticks(np.arange(0, world_size + 1, 1))
    ax.grid(True, linestyle=':', alpha=0.5)

    eps = fleet.cfg.tracking_error

    # Static obstacles
    for obs in static_circles:
        ax.add_patch(Circle((obs.cx, obs.cy), obs.radius,
                            color='red', alpha=0.35, zorder=2))
        ax.add_patch(Circle((obs.cx, obs.cy), obs.radius + eps,
                            fill=False, color='orange', linestyle='--',
                            linewidth=1.2, zorder=2))

    for obs in static_rects:
        x0, y0, x1, y1 = obs.bounds
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                color='red', alpha=0.35, zorder=2))
        ax.add_patch(Rectangle((x0 - eps, y0 - eps),
                                (x1 - x0) + 2*eps, (y1 - y0) + 2*eps,
                                fill=False, color='orange', linestyle='--',
                                linewidth=1.2, zorder=2))

    # Camera blobs
    # for blob in camera_blobs:
    #     ax.add_patch(Circle((blob.x, blob.y), blob.radius,
    #                         color='purple', alpha=0.25, zorder=2))
    #     # ax.text(blob.x, blob.y + blob.radius + 0.2, 'CAM', fontsize=7,
    #     #         ha='center', color='purple')

    # A* paths
    for rid, path in global_paths.items():
        color = agent_data[rid]['color']
        ax.plot([p[0] for p in path], [p[1] for p in path],
                color=color, linestyle='--', linewidth=1.5, alpha=0.4, zorder=1)

    # Goals
    for rid, data in agent_data.items():
        gx, gy = data['goal']
        ax.scatter(gx, gy, s=220, c=data['color'], marker='*', zorder=5)
        ax.annotate(f"Goal {rid}", (gx, gy),
                    textcoords='offset points', xytext=(5, 5), fontsize=8)

    # Start markers
    for rid, data in agent_data.items():
        sx, sy, _ = data['start']
        ax.scatter(sx, sy, s=120, c=data['color'], marker='s',
                   zorder=5, alpha=0.5)

    r = fleet.cfg.robot_radius
    L = fleet.cfg.wheel_base
    wl = L * 0.6

    body_patches  = {}
    heading_lines = {}
    trail_lines   = {}
    left_wheels   = {}
    right_wheels  = {}

    for rid, data in agent_data.items():
        color = data['color']
        x0, y0, _ = histories[rid][0]
        body_patches[rid]  = Circle((x0, y0), r, color=color, alpha=0.6, zorder=4)
        ax.add_patch(body_patches[rid])
        heading_lines[rid], = ax.plot([], [], 'k-', linewidth=2, zorder=5)
        trail_lines[rid],   = ax.plot([], [], color=color, linewidth=1.8,
                                       alpha=0.65, zorder=3)
        left_wheels[rid],   = ax.plot([], [], 'k-', linewidth=4,
                                       solid_capstyle='round', zorder=5)
        right_wheels[rid],  = ax.plot([], [], 'k-', linewidth=4,
                                       solid_capstyle='round', zorder=5)

    step_text  = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                          fontsize=10, va='top')
    speed_text = ax.text(0.02, 0.93, '', transform=ax.transAxes,
                          fontsize=8, va='top', color='gray')

    def update(frame):
        artists = [step_text, speed_text]
        step_text.set_text(f'Step: {frame}')

        speeds = []
        for rid in agent_data:
            hist = histories[rid]
            f    = min(frame, len(hist) - 1)
            x, y, theta = hist[f]

            trail_lines[rid].set_data([p[0] for p in hist[:f+1]],
                                       [p[1] for p in hist[:f+1]])
            body_patches[rid].center = (x, y)
            heading_lines[rid].set_data([x, x + r*math.cos(theta)],
                                         [y, y + r*math.sin(theta)])

            lx = x - (L/2)*math.sin(theta);  ly = y + (L/2)*math.cos(theta)
            rx = x + (L/2)*math.sin(theta);  ry = y - (L/2)*math.cos(theta)

            left_wheels[rid].set_data(
                [lx - wl*math.cos(theta), lx + wl*math.cos(theta)],
                [ly - wl*math.sin(theta), ly + wl*math.sin(theta)])
            right_wheels[rid].set_data(
                [rx - wl*math.cos(theta), rx + wl*math.cos(theta)],
                [ry - wl*math.sin(theta), ry + wl*math.sin(theta)])

            if f > 0:
                dx = hist[f][0] - hist[f-1][0]
                dy = hist[f][1] - hist[f-1][1]
                spd = math.hypot(dx, dy) / 0.1
                speeds.append(f"{rid}:{spd:.1f}")

            artists.extend([trail_lines[rid], body_patches[rid],
                             heading_lines[rid], left_wheels[rid],
                             right_wheels[rid]])

        speed_text.set_text('  '.join(speeds) + ' m/s')
        return artists

    return fig, update, max(len(h) for h in histories.values())


if __name__ == '__main__':

    # ── 1. Config ──────────────────────────────────────────────────────────
    cfg = DiffDriveConfig(
        robot_radius      = 1.0,
        wheel_base        = 0.8,
        max_linear_speed  = 2.0,
        max_angular_speed = 10.0,   # Increased to allow faster turns, reducing time spent in high heading error
        max_wheel_speed   = 3.0,    # Increased to support higher linear speeds during turns
        # KEY TUNING: balance tracking error and orientation time for better speed
        tracking_error    = 0.40,   # Moderate tracking error tolerance
        orientation_time  = 0.5,    # Moderate time to turn
        time_horizon      = 3.0,
        neighbor_dist     = 6.0,
        sim_dt            = 0.1,
    )

    # ── 2. Obstacles ───────────────────────────────────────────────────────
    INTERNAL_CIRCLES = [CircleObstacle(10.0,10.0,2.0)] #'cx', 'cy', and 'radius'
    INTERNAL_RECTS   = [RectObstacle(0.0, 0.0, 0.0, 0.0)]  #'x_min', 'y_min', 'x_max', and 'y_max'

    # Boundary walls: used ONLY by A* planner, NOT by APF.
    # APF only handles unknown dynamic obstacles (camera blobs).
    # Giving boundary walls to APF creates forces that fight path-following.
    BOUNDARY_RECTS = [
        RectObstacle(-1.0, -1.0, 21.0,  0.0),   # South
        RectObstacle(-1.0, 20.0, 21.0, 21.0),   # North
        RectObstacle(-1.0,  0.0,  0.0, 20.0),   # West
        RectObstacle(20.0,  0.0, 21.0, 20.0),   # East
    ]
    ALL_RECTS_FOR_ASTAR = INTERNAL_RECTS + BOUNDARY_RECTS

    # APF: internal obstacles only — NO boundary walls
    apf = APF(
        k_att          = 1.9,
        k_rep          = 10.0,    # reduced from 16 — less fighting with A* path
        rho_0          = 1.5,     # reduced from 2.5 — tighter influence zone
        max_force      = 7.0,
        vortex_gain    = 0.35,
        static_circles = INTERNAL_CIRCLES,
        static_rects   = INTERNAL_RECTS,   # NO boundaries here
        enable_static_repulsion = False,  # APF only for dynamic obstacles (camera blobs)
    )

    # ── 3. Agent definitions ────────────────────────────────────────────────
    # Increased separation to avoid ORCA deadlock at startup
    agent_data = {
        'A': {'start': (2.0, 2.0, math.pi/4),       'goal': (18.0, 18.0), 'color': 'royalblue'},
        'B': {'start': (12.0, 2.0, 0.0),            'goal': (18.0,  2.0), 'color': 'seagreen'},  # Moved B to y=12 for more separation
        'C': {'start': (2.0, 12.0, 0.0),            'goal': (13.0,  2.0), 'color': 'yellow'},  # Moved B to y=12 for more separation
    }

    # ── 4. Build fleet ─────────────────────────────────────────────────────
    fleet = FleetManager(cfg, apf)
    global_paths = {}

    for rid, data in agent_data.items():
        robot = fleet.add_robot(
            rid,
            filter_alpha     = 0.5,
            goal_tolerance   = 0.3,   # Reduced from 0.8 - prevents premature "goal reached"
            lookahead_window = 20,   # wider scan — better for sparse A* paths
            carrot_steps     = 8,    # further carrot — smoother following in large world
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
    camera_blobs = []  # No dynamic obstacles - APF only responds to dynamic obstacles

    # ── 6. Simulate ─────────────────────────────────────────────────────────
    print("Running simulation...")
    histories = run_simulation(
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
    )

    ani = animation.FuncAnimation(
        fig, update_fn,
        frames=n_frames, interval=25, blit=True, repeat=False
    )
    plt.tight_layout()
    plt.show()