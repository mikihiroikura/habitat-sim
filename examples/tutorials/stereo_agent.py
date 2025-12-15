# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np

# Need to import quaternion library here despite it not being used or else importing
# habitat_sim below will cause an invalid free() when audio is enabled in sim compilation
import quaternion  # noqa: F401

import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb

cv2 = None

def semantic_to_rgb(semantic_obs):
    # semantic_obs は (H, W) の形状で、各ピクセルにオブジェクトIDが入っています。
    # d3_40_colors_rgb は40色のリストです。IDを40で割った余りを使って色を割り当てます。
    semantic_img = d3_40_colors_rgb[semantic_obs % 40]
    return semantic_img

# Helper function to render observations from the stereo agent
def _render(sim, display, depth=False, semantic=False):
    for _ in range(100):
        # Just spin in a circle
        obs = sim.step("turn_right")
        # Put the two stereo observations next to each other
        stereo_pair = np.concatenate([obs["left_sensor"], obs["right_sensor"]], axis=1)

        # If it is a depth pair, manually normalize into [0, 1]
        # so that images are always consistent
        if depth:
            stereo_pair = np.clip(stereo_pair, 0, 10)
            stereo_pair /= 10.0
            
        if semantic:
            stereo_pair = semantic_to_rgb(stereo_pair)

        # If in RGB/RGBA format, change first to RGB and change to BGR
        if len(stereo_pair.shape) > 2:
            stereo_pair = stereo_pair[..., 0:3][..., ::-1]

        # display=False is used for the smoke test
        if display:
            cv2.imshow("stereo_pair", stereo_pair)
            k = cv2.waitKey()
            if k == ord("q"):
                break


def main(display=True):
    global cv2
    # Only import cv2 if we are doing to display
    if display:
        import cv2 as _cv2

        cv2 = _cv2

        cv2.namedWindow("stereo_pair")

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = (
        "/data/Replica/room_0/habitat/mesh_semantic.ply"
    )
    backend_cfg.enable_physics = True

    # First, let's create a stereo RGB agent
    left_rgb_sensor = habitat_sim.bindings.CameraSensorSpec()
    # Give it the uuid of left_sensor, this will also be how we
    # index the observations to retrieve the rendering from this sensor
    left_rgb_sensor.uuid = "left_sensor"
    left_rgb_sensor.resolution = [512, 512]
    # The left RGB sensor will be 1.5 meters off the ground
    # and 0.25 meters to the left of the center of the agent
    left_rgb_sensor.position = 1.5 * habitat_sim.geo.UP + 0.25 * habitat_sim.geo.LEFT

    # Same deal with the right sensor
    right_rgb_sensor = habitat_sim.CameraSensorSpec()
    right_rgb_sensor.uuid = "right_sensor"
    right_rgb_sensor.resolution = [512, 512]
    # The right RGB sensor will be 1.5 meters off the ground
    # and 0.25 meters to the right of the center of the agent
    right_rgb_sensor.position = 1.5 * habitat_sim.geo.UP + 0.25 * habitat_sim.geo.RIGHT

    agent_config = habitat_sim.AgentConfiguration()
    # Now we simply set the agent's list of sensor specs to be the two specs for our two sensors
    agent_config.sensor_specifications = [left_rgb_sensor, right_rgb_sensor]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_config]))

    sim.pathfinder.load_nav_mesh("/data/Replica/room_0/habitat/mesh_semantic.navmesh")
    print(f"NavMesh bounds: {sim.pathfinder.get_bounds()}") # ロード確認用（範囲を表示）
        
    # ランダムな有効座標を取得
    start_pos = sim.pathfinder.get_random_navigable_point()
    
    agent = sim.get_agent(0)
    new_state = agent.get_state()
    new_state.position = start_pos
    agent.set_state(new_state)

    _render(sim, display)
    sim.close()

    # Now let's do the exact same thing but for a depth camera stereo pair!
    left_depth_sensor = habitat_sim.CameraSensorSpec()
    left_depth_sensor.uuid = "left_sensor"
    left_depth_sensor.resolution = [512, 512]
    left_depth_sensor.position = 1.5 * habitat_sim.geo.UP + 0.25 * habitat_sim.geo.LEFT
    # The only difference is that we set the sensor type to DEPTH
    left_depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH

    right_depth_sensor = habitat_sim.CameraSensorSpec()
    right_depth_sensor.uuid = "right_sensor"
    right_depth_sensor.resolution = [512, 512]
    right_depth_sensor.position = (
        1.5 * habitat_sim.geo.UP + 0.25 * habitat_sim.geo.RIGHT
    )
    # The only difference is that we set the sensor type to DEPTH
    right_depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH

    agent_config = habitat_sim.AgentConfiguration()
    agent_config.sensor_specifications = [left_depth_sensor, right_depth_sensor]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_config]))

    _render(sim, display, depth=True)
    
    # Now let's do the exact same thing but for a depth camera stereo pair!
    left_semantic_camera = habitat_sim.CameraSensorSpec()
    left_semantic_camera.uuid = "left_sensor"
    left_semantic_camera.resolution = [512, 512]
    left_semantic_camera.position = 1.5 * habitat_sim.geo.UP + 0.25 * habitat_sim.geo.LEFT
    # The only difference is that we set the sensor type to SEMANTIC
    left_semantic_camera.sensor_type = habitat_sim.SensorType.SEMANTIC

    right_semantic_camera = habitat_sim.CameraSensorSpec()
    right_semantic_camera.uuid = "right_sensor"
    right_semantic_camera.resolution = [512, 512]
    right_semantic_camera.position = (
        1.5 * habitat_sim.geo.UP + 0.25 * habitat_sim.geo.RIGHT
    )
    # The only difference is that we set the sensor type to SEMANTIC
    right_semantic_camera.sensor_type = habitat_sim.SensorType.SEMANTIC 

    agent_config = habitat_sim.AgentConfiguration()
    agent_config.sensor_specifications = [left_semantic_camera, right_semantic_camera]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_config]))

    _render(sim, display, depth=False, semantic=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=True)
    args = parser.parse_args()
    main(display=args.display)
