import numpy as np
import habitat_sim
from matplotlib import pyplot as plt
import pathlib
import argparse
from tqdm import tqdm
from typing import Optional, List


def get_camera_pose(center, radius, duration, t):
    velocity = 2 * np.pi * radius / duration
    omega = velocity / radius
    theta = omega * t

    # Position in Habitat world coordinates
    position = np.array([
        radius * np.cos(theta),
        0,
        radius * np.sin(theta)
    ]) + center

    # Forward direction = tangent of circle
    direction = np.array([
        -np.sin(theta),
        0.0,
        np.cos(theta)
    ])

    # Habitat camera forward axis is -Z
    yaw = np.arctan2(-direction[0], -direction[2])

    rotation = np.quaternion(
        np.cos(yaw / 2),
        0.0,
        np.sin(yaw / 2),
        0.0
    )

    return position, rotation


def get_camera_intrinsics(sim: habitat_sim.Simulator, sensor_name: str) -> np.ndarray:
    # Get render camera
    render_camera = sim._sensors[sensor_name]._sensor_object.render_camera

    # Get projection matrix
    projection_matrix = render_camera.projection_matrix

    # Get resolution
    viewport_size = render_camera.viewport

    # Intrinsic calculation
    fx = projection_matrix[0, 0] * viewport_size[0] / 2.0
    fy = projection_matrix[1, 1] * viewport_size[1] / 2.0
    cx = (projection_matrix[2, 0] + 1.0) * viewport_size[0] / 2.0
    cy = (projection_matrix[2, 1] + 1.0) * viewport_size[1] / 2.0

    intrinsics = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])
    return intrinsics


def motion_rotate_dynamic_omega(args: argparse.Namespace) -> None:
    # Parameter setup
    scene_path: pathlib.Path = pathlib.Path(args.scene_path)
    output_folder_path = pathlib.Path(args.output_folder_path)
    (output_folder_path / "rgb").mkdir(parents=True, exist_ok=True)
    (output_folder_path / "semantic").mkdir(parents=True, exist_ok=True)
    duration_one_circle = args.duration_circle
    num_rotations = args.num_rotations
    total_duration = duration_one_circle * num_rotations
    fps = args.fps
    cam_width = args.cam_width
    cam_height = args.cam_height
    timestamp_file: Optional[pathlib.Path] = None
    angular_velocity_file: Optional[pathlib.Path] = None
    cam_intrinsics_file: Optional[pathlib.Path] = None
    gt_pose_file: Optional[pathlib.Path] = None
    imu_file: Optional[pathlib.Path] = None
    center = np.array(args.center, dtype=float)
    radius = args.radius
    if args.timestamp_file is not None:
        timestamp_file = pathlib.Path(args.output_folder_path) / pathlib.Path(args.timestamp_file)
        timestamp_file.parent.mkdir(parents=True, exist_ok=True)
        open(timestamp_file, "w").close()  # Create or clear the timestamp file
    if args.angular_velocity_file is not None:
        angular_velocity_file = pathlib.Path(args.output_folder_path) / pathlib.Path(args.angular_velocity_file)
        angular_velocity_file.parent.mkdir(parents=True, exist_ok=True)
        open(angular_velocity_file, "w").close()  # Create or clear the angular velocity file
    if args.cam_intrinsic_file is not None:
        cam_intrinsics_file = pathlib.Path(args.output_folder_path) / pathlib.Path(args.cam_intrinsic_file)
        cam_intrinsics_file.parent.mkdir(parents=True, exist_ok=True)
        open(cam_intrinsics_file, "w").close()  # Create or clear the camera intrinsics file
    if args.gt_pose_file is not None:
        gt_pose_file = pathlib.Path(args.output_folder_path) / pathlib.Path(args.gt_pose_file)
        gt_pose_file.parent.mkdir(parents=True, exist_ok=True)
        open(gt_pose_file, "w").close()  # Create or clear the ground truth pose file
    if args.imu_file is not None:
        imu_file = pathlib.Path(args.output_folder_path) / pathlib.Path(args.imu_file)
        imu_file.parent.mkdir(parents=True, exist_ok=True)
        open(imu_file, "w").close()  # Create or clear the IMU data file

    # Path generation with angular velocity
    path_positions: List = []
    path_rotations: List = []
    path_angvels: List = []
    for time in np.arange(0, total_duration, 1.0 / fps):
        pos, rot = get_camera_pose(center, radius, duration_one_circle, time)
        path_positions.append(pos)
        path_rotations.append(rot)
        # For simplicity, we assume a constant angular velocity
        ang_vel = 2 * np.pi / duration_one_circle
        path_angvels.append(ang_vel)

    # Simulator configuration
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path.__str__()
    sim_cfg.enable_physics = True
    
    # Setup RGB and Semantic camera sensors
    sensor_specs = []
    rgb_sensor = habitat_sim.bindings.CameraSensorSpec()
    rgb_sensor.uuid = "rgb_camera"
    rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor.resolution = [cam_height, cam_width]
    rgb_sensor.position = [0.0, 0.0, 0.0] # Relative to agent's position
    sensor_specs.append(rgb_sensor)

    semantic_camera_spec = habitat_sim.bindings.CameraSensorSpec()
    semantic_camera_spec.uuid = "semantic_camera"
    semantic_camera_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_camera_spec.resolution = [cam_height, cam_width]
    semantic_camera_spec.position = [0.0, 0.0, 0.0]
    sensor_specs.append(semantic_camera_spec)

    # Agent configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

    # Record camera intrinsics if required
    if cam_intrinsics_file is not None:
        cam_intrinsics = get_camera_intrinsics(sim, "rgb_camera")
        np.savetxt(cam_intrinsics_file, cam_intrinsics, fmt="%.6f")

    # Record images during linear motion 
    dt = 1.0 / fps
    current_time: float = 0.0

    for num_frames, (pos, rot, ang_vel) in enumerate(tqdm(zip(path_positions, path_rotations, path_angvels), desc="Recording frames", total=len(path_positions))):
        # Set current state
        agent = sim.get_agent(0)
        state = agent.get_state()
        state.position = pos.astype(np.float32)
        state.rotation = rot
        agent.set_state(state)
        
        # Get observations
        observation = sim.get_sensor_observations()
        rgb_image = observation["rgb_camera"]
        semantic_image = observation["semantic_camera"]

        # Save image to output folder
        plt.imsave(output_folder_path / "rgb" / f"{num_frames:06d}.png", rgb_image)
        plt.imsave(output_folder_path / "semantic" / f"{num_frames:06d}.png", semantic_image)

        # Save timestamp if required
        if timestamp_file is not None:
            with open(timestamp_file, "a") as f:
                f.write(f"{current_time:.6f}\n")

        # Save angular velocity if required
        if angular_velocity_file is not None:
            with open(angular_velocity_file, "a") as f:
                f.write(f"{current_time:.6f} {0:.6f} {ang_vel:.6f} {0:.6f}\n")

        # Convert ideal IMU measurements from Habitat-Sim axes to the requested
        # Kalibr axes: keep X and invert Y and Z. The camera yaw is pi - theta,
        # so its Habitat-Sim angular velocity is -Y; centripetal acceleration
        # points along +X. For the accelerometer reading (specific force),
        # include gravity: at rest this is +Y in Habitat-Sim axes, hence -Y
        # after the Habitat-Sim-to-Kalibr axis conversion.
        if imu_file is not None:
            habitat_to_kalibr_axes = np.array([1.0, -1.0, -1.0])
            habitat_angular_velocity = np.array([0.0, -ang_vel, 0.0])
            habitat_linear_acceleration = np.array([radius * ang_vel**2, 0.0, 0.0])
            gravity_magnitude = 9.80665  # m/s^2
            habitat_gravity = np.array([0.0, -gravity_magnitude, 0.0])
            # An accelerometer measures specific force: a - g, not a alone.
            habitat_accelerometer = habitat_linear_acceleration - habitat_gravity
            imu_angular_velocity = habitat_angular_velocity * habitat_to_kalibr_axes
            imu_linear_acceleration = habitat_accelerometer * habitat_to_kalibr_axes
            with open(imu_file, "a") as f:
                f.write(
                    f"{current_time:.6f} "
                    f"{imu_angular_velocity[0]:.6f} "
                    f"{imu_angular_velocity[1]:.6f} "
                    f"{imu_angular_velocity[2]:.6f} "
                    f"{imu_linear_acceleration[0]:.6f} "
                    f"{imu_linear_acceleration[1]:.6f} "
                    f"{imu_linear_acceleration[2]:.6f}\n"
                )

        # Update timestamp and frame count
        current_time += dt

        # Save ground truth pose if required
        if gt_pose_file is not None:
            with open(gt_pose_file, "a") as f:
                # Convert the sensor/body axes from Habitat-Sim to Kalibr
                # (X unchanged, Y/Z reversed) by composing a 180-degree
                # rotation about local X. Position remains in world coordinates.
                kalibr_rot = rot * np.quaternion(0.0, 1.0, 0.0, 0.0)
                f.write(
                    f"{current_time:.6f} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                    f"{kalibr_rot.x:.6f} {kalibr_rot.y:.6f} "
                    f"{kalibr_rot.z:.6f} {kalibr_rot.w:.6f}\n"
                )

    sim.close()
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Record RGB images from moving camera sensor in a Habitat-Sim simulator.
                                     This script records a circular motion around a specified center.
                                     The IMU and GT pose data are recorded in Kalibr axes convention (X forward, Y left, Z up).
                                     || Example usage: python examples/motion_rotate_dynamic_omega.py
                                     --scene_path /data/Replica/room_0/habitat/mesh_semantic.ply 
                                     --output_folder_path data/tmp 
                                     --duration_circle 10.0 --fps 30 
                                     --cam_width 640 --cam_height 480
                                     """
    )
    parser.add_argument(
        "-s",
        "--scene_path",
        help="Absolute or relative Path to scene file .ply or .glb",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_folder_path",
        help="Output folder path to record rgb and semantic images",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--duration_circle",
        help="Duration of the circular motion in seconds",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--fps",
        help="Frames per second for recording",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--num_rotations",
        help="Number of rotations to complete during the circular motion",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--center",
        type=float,
        nargs=3,
        default=[3.8866307735443115, -0.12736272811889648, -0.9417892694473267],
        metavar=("X", "Y", "Z"),
        help="Center of circular trajectory [x y z] in Habitat world coordinates"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=1.0,
        help="Radius of circular trajectory [m]"
    )
    parser.add_argument(
        "--cam_width",
        help="Camera width",
        type=int,
        default=640,
    )
    parser.add_argument(
        "--cam_height",
        help="Camera height",
        type=int,
        default=480,
    )
    parser.add_argument(
        "--timestamp_file",
        help="File name to record the timestamps",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--angular_velocity_file",
        help="File name to record the angular velocity",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--cam_intrinsic_file",
        help="File name to record the camera intrinsics",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--gt_pose_file",
        help="File name to record the ground truth camera poses",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--imu_file",
        help="File name to record the IMU data",
        type=str,
        default=None,
        required=False,
    )
    args = parser.parse_args()

    motion_rotate_dynamic_omega(args)