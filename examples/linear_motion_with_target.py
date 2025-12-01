import numpy as np
import habitat_sim
from matplotlib import pyplot as plt
import pathlib
import quaternion
import argparse
from tqdm import tqdm
from typing import Optional


def linear_motion_with_target(args: argparse.Namespace) -> None:
    # Parameter setup
    scene_path: pathlib.Path = pathlib.Path(args.scene_path)
    output_folder_path = pathlib.Path(args.output_folder_path)
    (output_folder_path / "rgb").mkdir(parents=True, exist_ok=True)
    (output_folder_path / "semantic").mkdir(parents=True, exist_ok=True)
    start_pos = np.array(args.start_pos)
    end_pos = np.array(args.end_pos)
    velocity = (end_pos - start_pos) / args.duration
    init_orientation = quaternion.from_float_array(args.init_orientation)
    duration = args.duration
    fps = args.fps
    cam_width = args.cam_width
    cam_height = args.cam_height
    timestamp_file: Optional[pathlib.Path] = None
    if args.timestamp_file is not None:
        timestamp_file = pathlib.Path(args.output_folder_path) / pathlib.Path(args.timestamp_file)
        timestamp_file.parent.mkdir(parents=True, exist_ok=True)
        open(timestamp_file, "w").close()  # Create or clear the timestamp file

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

    # Set initial state
    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = start_pos
    state.rotation = init_orientation
    agent.set_state(state)

    # Record images during linear motion
    num_frames = int(duration * fps)
    dt = 1.0 / fps
    current_time: float = 0.0

    for frame in tqdm(range(num_frames), desc="Recording frames"):
        # Get observations
        observation = sim.get_sensor_observations()
        rgb_image = observation["rgb_camera"]
        semantic_image = observation["semantic_camera"]

        # Save image to output folder
        plt.imsave(output_folder_path / "rgb" / f"{frame:06d}.png", rgb_image)
        plt.imsave(output_folder_path / "semantic" / f"{frame:06d}.png", semantic_image)
        
        # Save timestamp if required
        if timestamp_file is not None:
            with open(timestamp_file, "a") as f:
                f.write(f"{current_time:.6f}\n")

        # Update agent position and orientation
        agent = sim.get_agent(0)
        state = agent.get_state()
        state.position += velocity * dt
        agent.set_state(state)
        current_time += dt

    sim.close()
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Record RGB images from moving camera sensor in a Habitat-Sim simulator.
                                     || Example usage: python examples/linear_motion_with_target.py
                                     --scene_path /data/Replica/room_0/habitat/mesh_semantic.ply 
                                     --output_folder_path data/tmp 
                                     --start_pos 4.6 -0.32 -0.86 --end_pos 2.3 -0.32 -0.86
                                     --init_orientation 0.0 0.0 1.0 0.0 --duration 5.0 --fps 10 
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
        "--start_pos",
        help="Starting position of the agent",
        type=float,
        nargs=3,
        default=[4.6, -0.32, -0.86],
    )
    parser.add_argument(
        "--end_pos",
        help="End position of the agent",
        type=float,
        nargs=3,
        default=[2.3, -0.32, -0.86],
    )
    parser.add_argument(
        "--init_orientation",
        help="Initial orientation of the agent",
        type=float,
        nargs=4,
        default=[0.0, 0.0, 1.0, 0.0],
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
        help="Path to the timestamp file",
        type=str,
        default=None,
        required=False,
    )
    args = parser.parse_args()

    linear_motion_with_target(args)