import habitat_sim.bindings
import numpy as np
import cv2
import habitat_sim
import argparse
import pathlib
from typing import Optional
from habitat_sim.utils.common import quat_rotate_vector
import quaternion  # noqa: F401


init_pos = None


def controlCameraPosition(sim: habitat_sim.Simulator, key: int) -> None:
    # Params for motion definition
    move_dist = 0.1
    rotate_deg = 5.0
    
    # Control the camera position using keyboard inputs
    agent = sim.get_agent(0)
    new_state = agent.get_state()
    
    if key == ord('w'):
        # Move forward
        forward_vec = quat_rotate_vector(new_state.rotation, np.array([0.0, 0.0, -1]))
        new_state.position += np.array(forward_vec) * move_dist
    elif key == ord('s'):
        # Move backward
        backward_vec = quat_rotate_vector(new_state.rotation, np.array([0.0, 0.0, 1]))
        new_state.position += np.array(backward_vec) * move_dist
    elif key == ord('a'):
        # Move left
        left_vec = quat_rotate_vector(new_state.rotation, np.array([-1, 0.0, 0.0]))
        new_state.position += np.array(left_vec) * move_dist
    elif key == ord('d'):
        # Move right
        right_vec = quat_rotate_vector(new_state.rotation, np.array([1, 0.0, 0.0]))
        new_state.position += np.array(right_vec) * move_dist
    elif key == ord('z'):
        # Move up
        up_vec = quat_rotate_vector(new_state.rotation, np.array([0.0, 1.0, 0.0]))
        new_state.position += np.array(up_vec) * move_dist
    elif key == ord('x'):
        # Move down
        down_vec = quat_rotate_vector(new_state.rotation, np.array([0.0, -1.0, 0.0]))
        new_state.position += np.array(down_vec) * move_dist
    elif key == ord('q'):
        # Rotate left (yaw)
        rot = quaternion.from_rotation_vector(np.array([0, np.deg2rad(rotate_deg), 0]))
        new_state.rotation = new_state.rotation * rot
    elif key == ord('e'):
        # Rotate right (yaw)
        rot = quaternion.from_rotation_vector(np.array([0, np.deg2rad(-rotate_deg), 0]))
        new_state.rotation = new_state.rotation * rot
    elif key == ord('r'):
        # Rotate up (pitch)
        rot = quaternion.from_rotation_vector(np.array([np.deg2rad(rotate_deg), 0, 0]))
        new_state.rotation = new_state.rotation * rot
    elif key == ord('f'):
        # Rotate down (pitch)
        rot = quaternion.from_rotation_vector(np.array([np.deg2rad(-rotate_deg), 0, 0]))
        new_state.rotation = new_state.rotation * rot

    elif key == ord(' '):
        # Reset position
        global init_pos
        new_state.position = init_pos
        new_state.rotation = quaternion.from_float_array([1, 0, 0, 0])  # No rotation

    agent.set_state(new_state)
    print(f"Position: {sim.get_agent(0).get_state().sensor_states['rgb_camera'].position}, Orientation: {sim.get_agent(0).get_state().sensor_states['rgb_camera'].rotation}")
    return None


def camera_image_viewer(args: argparse.Namespace) -> None:
    # Parameter setup
    scene_path: pathlib.Path = args.scene_path
    navmesh_path: pathlib.Path = args.navmesh_path
    output_folder_path: Optional[pathlib.Path] = None
    if args.output_folder_path is not None:
        output_folder_path = pathlib.Path(args.output_folder_path)
        output_folder_path.mkdir(parents=True, exist_ok=True)

    # Simulator configuration
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path.__str__()
    sim_cfg.enable_physics = True

    # Setup RGB camera sensor
    rgb_sensor = habitat_sim.bindings.CameraSensorSpec()
    rgb_sensor.uuid = "rgb_camera"
    rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor.resolution = [args.cam_height, args.cam_width]
    rgb_sensor.position = [0.0, 0.0, 0.0] # Relative to agent's position

    # Initialize simulator
    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

    # Initialize agent position (=camera position) from valid navigable point
    sim.pathfinder.load_nav_mesh(navmesh_path.__str__())
    global init_pos
    init_pos = sim.pathfinder.get_random_navigable_point()
    agent = sim.get_agent(0)
    new_state = agent.get_state()
    new_state.position = init_pos
    agent.set_state(new_state)
    print(f"Camera initialized at position: {init_pos}")

    # Start rendering loop
    while True:
        observations = sim.get_sensor_observations()
        rgb_image = observations["rgb_camera"]

        # Display the RGB image
        cv2.imshow("RGB Camera View", rgb_image)
        k = cv2.waitKey(30)

        # # Save the RGB image if output folder is specified
        # if output_folder_path is not None:
        #     img_filename = output_folder_path / f"rgb_image_{len(list(output_folder_path.glob('rgb_image_*.png')))}.png"
        #     cv2.imwrite(img_filename.__str__(), cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2BGR))
        #     print(f"Saved image to: {img_filename}")

        # Break the loop on 'ESC' key press
        if k == 27: #ESC key
            break
        elif k != -1:
            controlCameraPosition(sim, k)
            
            
    sim.close()
    return None



if __name__== "__main__":
    parser = argparse.ArgumentParser(
        description="""Visualize RGB image from camera sensor in a Habitat-Sim simulator.
                                     || Example usage: python examples/camera_image_viewer.py"""
    )
    parser.add_argument(
        "-s",
        "--scene_path",
        help="Absolute or relative Path to scene file .ply or .glb",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--navmesh_path",
        help="Absolute or relative Path to navmesh file .navmesh",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_folder_path",
        help="Output folder path for .npz files",
        type=str,
        default=None,
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
    args = parser.parse_args()

    camera_image_viewer(args)