import numpy as np
import habitat_sim
from matplotlib import pyplot as plt
import pathlib
import argparse
import json
from tqdm import tqdm
from typing import Optional, List


def make_figure8_table(amplitude_x, amplitude_z, samples=20001):
    """Arc-length lookup for p(theta)=(A sin(theta), 0, B sin(2 theta))."""
    theta = np.linspace(0.0, 2.0 * np.pi, samples)
    dx = amplitude_x * np.cos(theta)
    dz = 2.0 * amplitude_z * np.cos(2.0 * theta)
    speed_theta = np.hypot(dx, dz)
    dtheta = theta[1] - theta[0]
    length = np.concatenate(([0.0], np.cumsum(
        0.5 * (speed_theta[:-1] + speed_theta[1:]) * dtheta
    )))
    return theta, length, float(length[-1])


def get_camera_pose(center, amplitude_x, amplitude_z, duration_figure8,
                    accel_duration, t, theta_table, arc_table, path_length):
    """Return pose, optical-frame gyro and accelerometer at time t.

    duration_figure8 is the duration of one FULL figure-eight at cruise speed.
    During acceleration the path is traversed at a smoothstep speed profile.
    """
    cruise_speed = path_length / duration_figure8
    if accel_duration > 0 and t < accel_duration:
        u = t / accel_duration
        speed = cruise_speed * (3*u*u - 2*u*u*u)
        tangential_accel = cruise_speed * (6*u - 6*u*u) / accel_duration
        distance = cruise_speed * accel_duration * (u**3 - 0.5*u**4)
    else:
        speed = cruise_speed
        tangential_accel = 0.0
        distance = cruise_speed * (t - 0.5*accel_duration)

    loops = np.floor(distance / path_length)
    phase_distance = distance - loops * path_length
    theta = np.interp(phase_distance, arc_table, theta_table) + loops * 2*np.pi

    dx = amplitude_x * np.cos(theta)
    dz = 2.0 * amplitude_z * np.cos(2.0 * theta)
    ddx = -amplitude_x * np.sin(theta)
    ddz = -4.0 * amplitude_z * np.sin(2.0 * theta)
    q = np.hypot(dx, dz)
    tangent = np.array([dx / q, 0.0, dz / q])
    # Signed curvature in XZ plane, with +theta increasing.
    curvature = (dx * ddz - dz * ddx) / (q**3)
    # Unit normal pointing left of tangent in the XZ coordinate plot.
    normal = np.array([-tangent[2], 0.0, tangent[0]])
    acceleration_world = tangential_accel*tangent + speed**2*curvature*normal

    position = np.asarray(center, dtype=float) + np.array([
        amplitude_x*np.sin(theta), 0.0, amplitude_z*np.sin(2.0*theta)
    ])
    # Habitat camera forward is local -Z.
    yaw = np.arctan2(-tangent[0], -tangent[2])
    rotation = np.quaternion(np.cos(yaw/2), 0.0, np.sin(yaw/2), 0.0)

    # Habitat local X is right, local Y up, local -Z forward.
    right = np.array([np.cos(yaw), 0.0, -np.sin(yaw)])
    # Optical frame X right, Y down, Z forward.
    angular_velocity = np.array([0.0, speed*curvature, 0.0])
    linear_acceleration = np.array([
        np.dot(acceleration_world, right), -9.80665,
        np.dot(acceleration_world, tangent)
    ])
    return position, rotation, angular_velocity, linear_acceleration


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


def motion_figure8(args: argparse.Namespace) -> None:
    # Parameter setup
    scene_path: pathlib.Path = pathlib.Path(args.scene_path)
    output_folder_path = pathlib.Path(args.output_folder_path)
    (output_folder_path / "rgb").mkdir(parents=True, exist_ok=True)
    (output_folder_path / "semantic").mkdir(parents=True, exist_ok=True)
    duration_one_circle = args.duration_figure8
    num_rotations = args.num_rotations
    accel_duration = args.accel_duration
    if duration_one_circle <= 0 or args.amplitude_x <= 0 or args.amplitude_z <= 0 or accel_duration < 0 or num_rotations < 1 or args.fps <= 0:
        raise ValueError("duration_figure8, amplitude_x, amplitude_z, fps must be positive; accel_duration >= 0; num_rotations >= 1")
    total_duration = accel_duration + duration_one_circle * num_rotations
    fps = args.fps
    cam_width = args.cam_width
    cam_height = args.cam_height
    timestamp_file: Optional[pathlib.Path] = None
    angular_velocity_file: Optional[pathlib.Path] = None
    cam_intrinsics_file: Optional[pathlib.Path] = None
    gt_pose_file: Optional[pathlib.Path] = None
    imu_file: Optional[pathlib.Path] = None
    center = np.array(args.center, dtype=float)
    amplitude_x = args.amplitude_x
    amplitude_z = args.amplitude_z
    theta_table, arc_table, path_length = make_figure8_table(amplitude_x, amplitude_z)
    if args.config_yaml is not None:
        config_yaml_path = pathlib.Path(args.config_yaml)
        config_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        center_yaml = ", ".join(f"{coordinate:.12g}" for coordinate in center)
        with open(config_yaml_path, "w", encoding="utf-8") as f:
            f.write(f"scene_path: {json.dumps(str(scene_path))}\n")
            f.write(f"duration_figure8: {duration_one_circle}\n")
            f.write(f"accel_duration: {accel_duration}\n")
            f.write(f"fps: {fps}\n")
            f.write(f"num_rotations: {num_rotations}\n")
            f.write(f"center: [{center_yaml}]\n")
            f.write(f"amplitude_x: {amplitude_x}\n")
            f.write(f"amplitude_z: {amplitude_z}\n")
            f.write(f"cam_width: {cam_width}\n")
            f.write(f"cam_height: {cam_height}\n")
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

    # Include the final sample at the end of the constant-speed phase.
    sample_times = np.arange(int(np.ceil(total_duration * fps)) + 1) / fps
    sample_times = sample_times[sample_times < total_duration]
    sample_times = np.append(sample_times, total_duration)
    path_positions: List = []
    path_rotations: List = []
    path_angvels: List = []
    path_accels: List = []
    for time in sample_times:
        pos, rot, ang_vel, accel = get_camera_pose(
            center, amplitude_x, amplitude_z, duration_one_circle,
            accel_duration, time, theta_table, arc_table, path_length
        )
        path_positions.append(pos)
        path_rotations.append(rot)
        path_angvels.append(ang_vel)
        path_accels.append(accel)

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
    for num_frames, (current_time, pos, rot, ang_vel, accel) in enumerate(tqdm(
        zip(sample_times, path_positions, path_rotations, path_angvels, path_accels),
        desc="Recording frames", total=len(path_positions)
    )):
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
                f.write(f"{current_time:.6f} {ang_vel[0]:.6f} {ang_vel[1]:.6f} {ang_vel[2]:.6f}\n")

        # Analytic IMU in optical frame, co-located with the camera.
        if imu_file is not None:
            with open(imu_file, "a") as f:
                f.write(
                    f"{current_time:.6f} "
                    f"{ang_vel[0]:.6f} {ang_vel[1]:.6f} {ang_vel[2]:.6f} "
                    f"{accel[0]:.6f} {accel[1]:.6f} {accel[2]:.6f}\n"
                )

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
                                     This script records a constant-arc-speed figure-eight around a specified center.
                                     The IMU and GT pose data use camera optical axes (X right, Y down, Z forward).
                                     || Example usage: python examples/motion_rotate_dynamic_omega.py
                                     --scene_path /data/Replica/room_0/habitat/mesh_semantic.ply 
                                     --output_folder_path data/tmp 
                                     --duration_figure8 10.0 --fps 30 
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
        "--duration_figure8",
        help="Seconds per complete figure-eight at constant cruise speed",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--accel_duration",
        help="Seconds to accelerate smoothly from rest to circular cruising speed",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--fps",
        help="Frames per second for recording",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--num_rotations",
        help="Number of full figure-eight loops after acceleration",
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
    parser.add_argument("--amplitude_x", type=float, default=1.0,
                        help="Half-width of figure-eight in X [m]")
    parser.add_argument("--amplitude_z", type=float, default=0.5,
                        help="Z amplitude of figure-eight [m]")
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
    parser.add_argument(
        "--config_yaml",
        help="Path to save the run configuration as a YAML file",
        type=str,
        default=None,
        required=False,
    )
    args = parser.parse_args()

    motion_figure8(args)