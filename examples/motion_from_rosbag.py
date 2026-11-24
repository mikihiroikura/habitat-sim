import numpy as np
import habitat_sim
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation, Slerp
import pathlib
import argparse
import json
from tqdm import tqdm
from typing import Any, Optional
from rosbags.highlevel import AnyReader


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


def read_rosbag_trajectory(
    rosbag_path: pathlib.Path,
    topic: str,
    fps: int,
    init_pos: np.ndarray,
    init_quat: np.ndarray,
):
    """Read and resample poses, aligning the first pose to the requested start pose."""
    if fps <= 0:
        raise ValueError("fps must be a positive integer")

    samples = []
    bag_start_time_ns = 0
    with AnyReader([rosbag_path]) as input_bag:
        bag_start_time_ns = input_bag.start_time
        connections = [connection for connection in input_bag.connections if connection.topic == topic]
        if not connections:
            available_topics = sorted({connection.topic for connection in input_bag.connections})
            raise ValueError(
                f"Topic {topic!r} was not found in {rosbag_path}. "
                f"Available topics: {', '.join(available_topics)}"
            )

        message_count = sum(connection.msgcount for connection in connections)
        for connection, bag_timestamp_ns, raw_message in tqdm(
            input_bag.messages(connections=connections),
            total=message_count,
            desc=f"Reading trajectory: {topic}",
        ):
            message: Any = input_bag.deserialize(raw_message, connection.msgtype)
            pose = message.pose
            header_stamp = message.header.stamp
            stamp = header_stamp.sec + header_stamp.nanosec * 1e-9
            if not np.isfinite(stamp) or stamp == 0.0:
                stamp = bag_timestamp_ns * 1e-9

            position = np.array(
                [pose.position.x, pose.position.y, pose.position.z], dtype=float
            )
            quaternion_xyzw = np.array(
                [pose.orientation.x, pose.orientation.y,
                 pose.orientation.z, pose.orientation.w],
                dtype=float,
            )
            quaternion_norm = np.linalg.norm(quaternion_xyzw)
            if quaternion_norm < 1e-8:
                raise ValueError(f"Invalid zero quaternion in topic {topic!r}")
            samples.append((stamp, position, quaternion_xyzw / quaternion_norm))

    if not samples:
        raise ValueError(f"No PoseStamped messages found on topic {topic!r}")

    # Sort by the message's own timestamp and keep the latest pose at duplicate times.
    samples.sort(key=lambda sample: sample[0])
    unique_samples = {}
    for stamp, position, quaternion_xyzw in samples:
        unique_samples[stamp] = (position, quaternion_xyzw)
    sample_times = np.asarray(list(unique_samples.keys()), dtype=float)
    source_positions = np.asarray(
        [sample[0] for sample in unique_samples.values()], dtype=float
    )
    source_quaternions = np.asarray(
        [sample[1] for sample in unique_samples.values()], dtype=float
    )

    if len(sample_times) == 1:
        frame_times = sample_times.copy()
        path_positions = source_positions.copy()
        sampled_rotations = Rotation.from_quat(source_quaternions)
    else:
        frame_times = np.arange(sample_times[0], sample_times[-1], 1.0 / fps)
        if len(frame_times) == 0 or frame_times[-1] < sample_times[-1]:
            frame_times = np.append(frame_times, sample_times[-1])
        else:
            frame_times[-1] = sample_times[-1]

        path_positions = np.column_stack([
            np.interp(frame_times, sample_times, source_positions[:, axis])
            for axis in range(3)
        ])
        source_rotations = Rotation.from_quat(source_quaternions)
        sampled_rotations = Slerp(sample_times, source_rotations)(frame_times)

    target_quaternion = np.asarray(init_quat, dtype=float)
    target_quaternion_norm = np.linalg.norm(target_quaternion)
    if target_quaternion_norm < 1e-8:
        raise ValueError("init_quat must be a non-zero quaternion in [x, y, z, w] order")
    target_rotation = Rotation.from_quat(target_quaternion / target_quaternion_norm)
    initial_alignment = target_rotation * sampled_rotations[0].inv()
    target_position = np.asarray(init_pos, dtype=float)
    path_positions = target_position + initial_alignment.apply(
        path_positions - path_positions[0]
    )
    sampled_rotations = initial_alignment * sampled_rotations
    frame_quaternions = sampled_rotations.as_quat(canonical=False)
    path_rotations = [
        np.quaternion(q[3], q[0], q[1], q[2]) for q in frame_quaternions
    ]
    frame_times -= bag_start_time_ns * 1e-9  # Make the first timestamp zero

    return frame_times, path_positions, path_rotations


def write_rosbag_imu(rosbag_path: pathlib.Path, output_path: pathlib.Path, imu_topic: str, stop_recording_time: float) -> int:
    """Write every sensor_msgs/Imu message with time relative to bag start."""
    message_count = 0
    with AnyReader([rosbag_path]) as input_bag:
        imu_connections = [connection for connection in input_bag.connections if connection.topic == imu_topic]
        if not imu_connections:
            raise ValueError(f"No {imu_topic} topics found in {rosbag_path}")

        bag_start_time_ns = input_bag.start_time
        total_messages = sum(connection.msgcount for connection in imu_connections)
        with output_path.open("w", encoding="utf-8") as imu_file:
            for connection, timestamp_ns, raw_message in tqdm(
                input_bag.messages(connections=imu_connections),
                total=total_messages,
                desc="Reading IMU topics",
            ):
                message: Any = input_bag.deserialize(raw_message, connection.msgtype)
                relative_time_s = (timestamp_ns - bag_start_time_ns) * 1e-9
                angular_velocity = message.angular_velocity
                linear_acceleration = message.linear_acceleration
                imu_file.write(
                    f"{relative_time_s:.9f} "
                    f"{angular_velocity.x:.9f} {angular_velocity.y:.9f} "
                    f"{angular_velocity.z:.9f} "
                    f"{linear_acceleration.x:.9f} {linear_acceleration.y:.9f} "
                    f"{linear_acceleration.z:.9f}\n"
                )
                message_count += 1

                if stop_recording_time > 0 and relative_time_s >= stop_recording_time:
                    break

    return message_count



def write_rosbag_gtpose(rosbag_path: pathlib.Path, output_path: pathlib.Path, gtpose_topic: str, stop_recording_time: float) -> int:
    """Write PoseStamped GT poses with time relative to bag start.

    Each output row contains relative time, position (x, y, z), and quaternion
    (x, y, z, w).
    """
    message_count = 0
    with AnyReader([rosbag_path]) as input_bag:
        gtpose_connections = [connection for connection in input_bag.connections if connection.topic == gtpose_topic]
        if not gtpose_connections:
            raise ValueError(f"No {gtpose_topic} topics found in {rosbag_path}")

        bag_start_time_ns = input_bag.start_time
        total_messages = sum(connection.msgcount for connection in gtpose_connections)
        with output_path.open("w", encoding="utf-8") as gtpose_file:
            for connection, timestamp_ns, raw_message in tqdm(
                input_bag.messages(connections=gtpose_connections),
                total=total_messages,
                desc="Reading GT Pose topics",
            ):
                message: Any = input_bag.deserialize(raw_message, connection.msgtype)
                relative_time_s = (timestamp_ns - bag_start_time_ns) * 1e-9
                position = message.pose.position
                orientation = message.pose.orientation
                gtpose_file.write(
                    f"{relative_time_s:.9f} "
                    f"{position.x:.9f} {position.y:.9f} {position.z:.9f} "
                    f"{orientation.x:.9f} {orientation.y:.9f} "
                    f"{orientation.z:.9f} {orientation.w:.9f}\n"
                )
                message_count += 1

                if stop_recording_time > 0 and relative_time_s >= stop_recording_time:
                    break

    return message_count


def motion_rotate_dynamic_omega(args: argparse.Namespace) -> None:
    # Parameter setup
    scene_path: pathlib.Path = pathlib.Path(args.scene_path)
    output_folder_path = pathlib.Path(args.output_folder_path)
    rosbag_path: pathlib.Path = pathlib.Path(args.input_rosbag_path)
    gt_pose_rostopic: str = args.gt_pose_rostopic
    (output_folder_path / "rgb").mkdir(parents=True, exist_ok=True)
    (output_folder_path / "semantic").mkdir(parents=True, exist_ok=True)
    stop_recording_time: float = args.stop_recording
    fps = args.fps
    cam_width = args.cam_width
    cam_height = args.cam_height
    timestamp_file: Optional[pathlib.Path] = None
    cam_intrinsics_file: Optional[pathlib.Path] = None
    gt_pose_file: Optional[pathlib.Path] = None
    imu_file: Optional[pathlib.Path] = None
    imu_rostopic: str = args.imu_rostopic
    if args.config_yaml is not None:
        config_yaml_path = pathlib.Path(args.config_yaml)
        config_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_yaml_path, "w", encoding="utf-8") as f:
            f.write(f"scene_path: {json.dumps(str(scene_path))}\n")
            f.write(f"rosbag_path: {json.dumps(str(rosbag_path))}\n")
            f.write(f"gt_pose_rostopic: {json.dumps(gt_pose_rostopic)}\n")
            f.write(f"fps: {fps}\n")
            f.write(f"cam_width: {cam_width}\n")
            f.write(f"cam_height: {cam_height}\n")
            f.write(f"init_pos: {json.dumps(args.init_pos)}\n")
            f.write(f"init_quat: {json.dumps(args.init_quat)}\n")
            f.write(f"imu_rostopic: {json.dumps(imu_rostopic)}\n")
            f.write(f"stop_recording: {json.dumps(args.stop_recording)}\n")
    if args.timestamp_file is not None:
        timestamp_file = pathlib.Path(args.output_folder_path) / pathlib.Path(args.timestamp_file)
        timestamp_file.parent.mkdir(parents=True, exist_ok=True)
        open(timestamp_file, "w").close()  # Create or clear the timestamp file
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

    # Extract and resample camera poses from the requested PoseStamped topic.
    (
        sample_times,
        path_positions,
        path_rotations,
    ) = read_rosbag_trajectory(
        rosbag_path,
        gt_pose_rostopic,
        fps,
        np.asarray(args.init_pos, dtype=float),
        np.asarray(args.init_quat, dtype=float),
    )

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

    # Record IMU data from rosbag if required
    if imu_file is not None:
        imu_message_count = write_rosbag_imu(rosbag_path, imu_file, imu_rostopic, stop_recording_time)
        print(f"Wrote {imu_message_count} IMU messages to {imu_file}")

    # Record GT poses from rosbag if required
    if gt_pose_file is not None:
        gtpose_message_count = write_rosbag_gtpose(
            rosbag_path, gt_pose_file, gt_pose_rostopic, stop_recording_time
        )
        print(f"Wrote {gtpose_message_count} GT poses to {gt_pose_file}")

    # Record images along the trajectory extracted from the rosbag.
    for num_frames, (current_time, pos, rot) in enumerate(tqdm(
        zip(sample_times, path_positions, path_rotations),
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
                
        # Stop recording if the specified stop time is reached
        if stop_recording_time > 0 and current_time >= stop_recording_time:
            break

    if stop_recording_time > 0:
        print(f"Reached stop recording time: {stop_recording_time:.2f}s. Stopping.")

    sim.close()
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Record RGB and semantic images by replaying a PoseStamped trajectory from a ROS bag.
                                     Pose positions and orientations are interpreted in Habitat-Sim world coordinates.
                                     The IMU and GT pose data use camera optical axes (X right, Y down, Z forward).
                                     || Example usage: python examples/motion_from_rosbag.py
                                     --scene_path /data/Replica/room_0/habitat/mesh_semantic.ply 
                                     --input_rosbag_path /data/trajectory.bag
                                     --gt_pose_rostopic /optitrack/davis
                                     --output_folder_path data/tmp 
                                     --fps 30 
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
        "--input_rosbag_path",
        help="Path to the input ROS bag file",
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
        "--init_pos",
        type=float,
        nargs=3,
        default=[3.8866307735443115, -0.12736272811889648, -0.9417892694473267],
        metavar=("X", "Y", "Z"),
        help="Initial position of the camera [x y z] in Habitat world coordinates"
    )
    parser.add_argument(
        "--init_quat",
        type=float,
        nargs=4,
        default=[0, 0, 0, 1],
        metavar=("X", "Y", "Z", "W"),
        help="Initial orientation of the camera [x y z w] in Habitat world coordinates"
    )
    parser.add_argument(
        "--fps",
        help="Frames per second for recording",
        type=int,
        default=30,
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
        "--gt_pose_rostopic",
        help="ROS topic containing geometry_msgs/PoseStamped trajectory messages",
        type=str,
        required=False,
        default="/optitrack/davis",
    )
    parser.add_argument(
        "--imu_file",
        help="File name to record the IMU data",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--imu_rostopic",
        help="ROS topic containing sensor_msgs/Imu messages",
        type=str,
        default="/dvs/imu",
        required=False,
    )
    parser.add_argument(
        "--config_yaml",
        help="Path to save the run configuration as a YAML file",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--stop_recording",
        type=float,
        default=-1.0,
        metavar=("TIME"),
        help="Time at which to stop recording"
    )
    args = parser.parse_args()

    motion_rotate_dynamic_omega(args)