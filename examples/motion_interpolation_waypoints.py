import numpy as np
import habitat_sim
from matplotlib import pyplot as plt
import pathlib
import quaternion
import argparse
from tqdm import tqdm
from typing import Optional, List
import json


def load_waypoints(json_path: str) -> List[dict]:
    with open(json_path, "r") as f:
        data = json.load(f)
    waypoints = []
    for wp in data:
        pos = np.array(wp["position"], dtype=float)
        rot = quaternion.from_float_array(wp["rotation"])
        waypoints.append({"position": pos, "rotation": rot})

    return waypoints


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


def motion_interpolation_waypoints(args: argparse.Namespace) -> None:
    # Parameter setup
    scene_path: pathlib.Path = pathlib.Path(args.scene_path)
    output_folder_path = pathlib.Path(args.output_folder_path)
    (output_folder_path / "rgb").mkdir(parents=True, exist_ok=True)
    (output_folder_path / "semantic").mkdir(parents=True, exist_ok=True)
    waypoints: List[dict] = load_waypoints(args.json_waypoints)
    duration = args.duration
    fps = args.fps
    cam_width = args.cam_width
    cam_height = args.cam_height
    timestamp_file: Optional[pathlib.Path] = None
    angular_velocity_file: Optional[pathlib.Path] = None
    cam_intrinsics_file: Optional[pathlib.Path] = None
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

    # Path interpolation
    path_positions: List = []
    path_rotations: List = []
    total_dist = sum(
        np.linalg.norm(waypoints[i+1]["position"] - waypoints[i]["position"])
        for i in range(len(waypoints) - 1)
    )
    total_rot_dist = sum(
        quaternion.rotation_intrinsic_distance(waypoints[i]["rotation"], waypoints[i+1]["rotation"])
        for i in range(len(waypoints) - 1)
    )
    for i in range(len(waypoints)-1):
        p0 = waypoints[i]["position"]
        p1 = waypoints[i+1]["position"]
        r0 = waypoints[i]["rotation"]
        r1 = waypoints[i+1]["rotation"]
        dist = np.linalg.norm(p1 - p0)
        rot_dist = quaternion.rotation_intrinsic_distance(r0, r1)
        seg_duration = 0
        if total_dist > 0:
            seg_duration = int(round(dist / total_dist * duration * fps))
        else:
            seg_duration = int(round(rot_dist / total_rot_dist * duration * fps))

        t = np.linspace(0, 1, seg_duration + 1)[:, None]
        positions = (1 - t) * p0 + t * p1
        rots = quaternion.slerp(r0, r1, 0, 1, t)

        if i == 0:
            # Put all for the first segment
            path_positions.append(positions)
            path_rotations.append(rots)
        else:
            # Remove the first point to avoid duplication
            path_positions.append(positions[1:])
            path_rotations.append(rots[1:])
    path_positions = list(np.vstack(path_positions))
    path_rotations = list(np.vstack(path_rotations))

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

    for num_frames, (pos, rot) in enumerate(tqdm(zip(path_positions, path_rotations), desc="Recording frames", total=len(path_positions))):
        # Set current state
        agent = sim.get_agent(0)
        state = agent.get_state()
        state.position = pos.astype(np.float32)
        state.rotation = rot[0]
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
                if num_frames == 0:
                    ang_vel = np.array([0.0, 0.0, 0.0])
                else:
                    prev_rot = path_rotations[num_frames - 1][0]
                    delta_rot = rot[0] * prev_rot.conjugate()
                    angle = 2 * np.arccos(np.clip(delta_rot.w, -1.0, 1.0))
                    if angle > np.pi:
                        angle -= 2 * np.pi
                    axis = np.array([delta_rot.x, delta_rot.y, delta_rot.z])
                    if np.linalg.norm(axis) < 1e-6:
                        axis = np.array([0.0, 0.0, 0.0])
                    else:
                        axis = axis / np.linalg.norm(axis)
                    ang_vel = (angle / dt) * axis
                f.write(f"{current_time:.6f} {ang_vel[0]:.6f} {ang_vel[1]:.6f} {ang_vel[2]:.6f}\n")

        # Update timestamp and frame count
        current_time += dt

    sim.close()
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Record RGB images from moving camera sensor in a Habitat-Sim simulator.
                                     || Example usage: python examples/motion_interpolation_waypoints.py
                                     --scene_path /data/Replica/room_0/habitat/mesh_semantic.ply 
                                     --output_folder_path data/tmp 
                                     --json_waypoints data/tmp/waypoints.json --duration 5.0 --fps 10 
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
        "--json_waypoints",
        help="Path to the waypoints JSON file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--duration",
        help="Duration of the motion in seconds",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--fps",
        help="Frames per second for recording",
        type=int,
        default=10,
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
    args = parser.parse_args()

    motion_interpolation_waypoints(args)