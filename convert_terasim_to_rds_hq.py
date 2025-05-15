import numpy as np
import tensorflow as tf
import click
import imageio as imageio_v1

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Union
from termcolor import cprint
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from waymo_open_dataset import dataset_pb2, label_pb2
from waymo_open_dataset.utils import frame_utils
from google.protobuf import json_format
from utils.wds_utils import write_to_tar, encode_dict_to_npz_bytes
from utils.bbox_utils import interpolate_pose

import sumolib
import xml.etree.ElementTree as ET
class TeraSim_Dataset:
    def __init__(self, terasim_record_root: Union[str, Path]):
        self.clip_id = terasim_record_root.stem
        self.sumo_net_path = terasim_record_root / 'map.net.xml'
        self.sumo_net = sumolib.net.readNet(self.sumo_net_path, withInternal=True, withPedestrianConnections=True)
        self.fcd_path = terasim_record_root / 'fcd_all.xml'
        self.av_id = "CAV"
        self.fcd_data = ET.parse(self.fcd_path).getroot()
        
    def __iter__(self):
        return self

WaymoProto2SemanticLabel = {
    label_pb2.Label.Type.TYPE_UNKNOWN: "Unknown",
    label_pb2.Label.Type.TYPE_VEHICLE: "Car",
    label_pb2.Label.Type.TYPE_PEDESTRIAN: "Pedestrian",
    label_pb2.Label.Type.TYPE_SIGN: "Sign",
    label_pb2.Label.Type.TYPE_CYCLIST: "Cyclist",
}

CameraNames = ['front', 'front_left', 'front_right', 'side_left', 'side_right']

SourceFps = 10 # waymo's recording fps
TargetFps = 30 # cosmos's expected fps
IndexScaleRatio = int(TargetFps / SourceFps)

if int(tf.__version__.split(".")[0]) < 2:
    tf.enable_eager_execution()

# make sure the GPU memory is not exhausted
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except Exception as e:
        print(e)

def convert_waymo_intrinsics(output_root: Path, clip_id: str, dataset: tf.data.TFRecordDataset):
    """
    read the first frame and convert the intrinsics to wds format

    Minimal required format:
        sample['pinhole_intrinsic.{camera_name}.npy'] = np.ndarray with shape (4, 4)
    """
    sample = {'__key__': clip_id}

    for frame_idx, data in enumerate(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        for camera_calib in frame.context.camera_calibrations:
            camera_name = get_camera_name(camera_calib.name).lower()

            intrinsic = camera_calib.intrinsic
            fx, fy, cx, cy = intrinsic[:4]
            w, h = camera_calib.width, camera_calib.height

            sample[f'pinhole_intrinsic.{camera_name}.npy'] = \
                np.array([fx, fy, cx, cy, w, h])

        write_to_tar(sample, output_root / 'pinhole_intrinsic' / f'{clip_id}.tar')

        # only process the first frame
        break

def get_map_elements_from_edge(edge):
    """
    Extract map elements (lane, road_line, road_edge) from a SUMO edge.
    
    Args:
        edge: SUMO edge object
        
    Returns:
        dict: Dictionary containing different types of map elements
    """
    # Initialize dictionary to store map elements
    map_elements = {
        'lane': [],          # Center lines of lanes
        'road_line': [],     # Lane markings
        'road_edge': []      # Road boundaries
    }
    
    # Get edge basic information
    shape = edge.getShape()  # Get edge shape points
    lanes = edge.getLanes()  # Get all lanes in this edge
    
    # Calculate total width from all lanes
    total_width = sum(lane.getWidth() for lane in lanes)
    
    # Process road boundaries (road_edge)
    left_boundary = []
    right_boundary = []
    for i in range(len(shape)-1):
        # Calculate direction vector between current and next point
        dx = shape[i+1][0] - shape[i][0]
        dy = shape[i+1][1] - shape[i][1]
        
        # Calculate perpendicular vector
        perp_x = -dy
        perp_y = dx
        
        # Normalize perpendicular vector
        length = np.sqrt(perp_x**2 + perp_y**2)
        perp_x /= length
        perp_y /= length
        
        # Calculate left and right boundary points using total width
        left_point = [shape[i][0] + perp_x * total_width/2, shape[i][1] + perp_y * total_width/2, 0]
        right_point = [shape[i][0] - perp_x * total_width/2, shape[i][1] - perp_y * total_width/2, 0]
        
        left_boundary.append(left_point)
        right_boundary.append(right_point)
    
    map_elements['road_edge'].extend([left_boundary, right_boundary])
    
    # Process lanes and lane markings
    for i, lane in enumerate(lanes):
        # Get lane center line
        lane_shape = lane.getShape()
        map_elements['lane'].append([[x, y, 0] for x, y in lane_shape])
        
        # Get lane width
        lane_width = lane.getWidth()
        
        # Calculate lane markings
        if i < len(lanes) - 1:  # Not the last lane
            lane_line = []
            for j in range(len(lane_shape)):
                if j < len(lane_shape) - 1:
                    # Calculate direction vector
                    dx = lane_shape[j+1][0] - lane_shape[j][0]
                    dy = lane_shape[j+1][1] - lane_shape[j][1]
                    
                    # Calculate perpendicular vector
                    perp_x = -dy
                    perp_y = dx
                    
                    # Normalize
                    length = np.sqrt(perp_x**2 + perp_y**2)
                    perp_x /= length
                    perp_y /= length
                    
                    # Calculate lane marking point
                    line_point = [lane_shape[j][0] + perp_x * lane_width/2, 
                                lane_shape[j][1] + perp_y * lane_width/2, 0]
                    lane_line.append(line_point)
            
            map_elements['road_line'].append(lane_line)
    
    return map_elements

def get_crosswalk_and_driveway(net):
    """
    Extract crosswalk and driveway information from SUMO network.
    
    Args:
        net: SUMO network object
        
    Returns:
        dict: Dictionary containing crosswalk and driveway information
    """
    map_elements = {
        'crosswalk': [],
        'driveway': []
    }
    
    # Get all junctions
    junctions = net.getNodes()
    
    for junction in junctions:
        junction_type = junction.getType()
        
        # Process crosswalks
        if junction_type in ["traffic_light", "priority"]:
            connections = junction.getConnections()
            
            for connection in connections:
                if connection.getTLLinkIndex() >= 0:
                    junction_shape = junction.getShape()
                    crosswalk = [[point[0], point[1], 0] for point in junction_shape]
                    map_elements['crosswalk'].append(crosswalk)
        
        # Process driveways
        for edge in junction.getIncoming() + junction.getOutgoing():
            edge_type = edge.getType()
            if edge_type in ["residential", "living_street"]:
                edge_shape = edge.getShape()
                driveway = [[point[0], point[1], 0] for point in edge_shape]
                map_elements['driveway'].append(driveway)
    
    return map_elements

def convert_terasim_hdmap(output_root: Path, clip_id: str, dataset: TeraSim_Dataset):
    """
    Convert TeraSim map data to RDS-HQ format.
    
    Args:
        output_root: Output directory path
        clip_id: Clip ID
        dataset: TeraSim dataset object
    """
    def hump_to_underline(hump_str):
        import re
        return re.sub(r'([a-z])([A-Z])', r'\1_\2', hump_str).lower()
    from scenparse.processors.sumo2waymo import SUMO2Waymo
    sumo2waymo = SUMO2Waymo(dataset.sumo_net_path)
    sumo2waymo.parse(have_road_edges=True, have_road_lines=True)
    scenario = sumo2waymo.convert_to_scenario(scenario_id=clip_id)

    hdmap_names_polyline = ["lane", "road_line", "road_edge"]
    hdmap_names_polygon = ["crosswalk", "speed_bump", "driveway"]
    
    hdmap_name_to_data = {}
    for hdmap_name in hdmap_names_polyline + hdmap_names_polygon:
        hdmap_name_to_data[hump_to_underline(hdmap_name)] = []


    map_features_list = json_format.MessageToDict(scenario)['mapFeatures']

    for hdmap_content in map_features_list:
        hdmap_name = list(hdmap_content.keys())
        hdmap_name.remove("id")
        hdmap_name = hdmap_name[0]
        hdmap_name_lower = hump_to_underline(hdmap_name)

        hdmap_data = hdmap_content[hdmap_name]
        if hdmap_name_lower in hdmap_names_polyline:
            hdmap_data = hdmap_data['polyline']
            polyline = [[point['x'], point['y'], point['z']] for point in hdmap_data]
            hdmap_name_to_data[hdmap_name_lower].append(polyline)
        elif hdmap_name_lower in hdmap_names_polygon:
            hdmap_data = hdmap_data['polygon']
            polygon = [[point['x'], point['y'], point['z']] for point in hdmap_data]
            hdmap_name_to_data[hdmap_name_lower].append(polygon)
        else:
            print(f"Unkown hdmap item name: {hdmap_name}, skip this item")

    # Plot all HDMap elements for visualization
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    
    for i, (hdmap_name, hdmap_data) in enumerate(hdmap_name_to_data.items()):
        if len(hdmap_data) == 0:
            continue
            
        color = colors[i % len(colors)]
        for polyline in hdmap_data:
            polyline = np.array(polyline)
            plt.plot(polyline[:, 0], polyline[:, 1], color=color, alpha=0.5, label=hdmap_name)
        
    plt.title(f'HDMap Elements Visualization - {clip_id}')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)') 
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    
    # Save plot
    plt.savefig("sumo_waymo_hdmap_visualize.png")
    plt.close()

    

    # convert to cosmos's name convention for easier processing
    hdmap_name_to_cosmos = {
        'lane': 'lanes',
        'road_line': 'lanelines',
        'road_edge': 'road_boundaries',
        'crosswalk': 'crosswalks',
        'speed_bump': None,
        'driveway': None
    }

    for hdmap_name, hdmap_data in hdmap_name_to_data.items():
        hdmap_name_in_cosmos = hdmap_name_to_cosmos[hdmap_name]
        if hdmap_name_in_cosmos is None:
            continue

        if hdmap_name in hdmap_names_polyline:
            vertex_indicator = 'polyline3d'
        else:
            vertex_indicator = 'surface'

        # to match cosmos format, the easiest way is to add 'vertices' key for the polyline or polygon
        sample = {'__key__': clip_id, f'{hdmap_name_in_cosmos}.json': {'labels': []}}

        for each_polyline_or_polygon in hdmap_data:
            sample[f'{hdmap_name_in_cosmos}.json']['labels'].append({
                'labelData': {
                    'shape3d': {
                        vertex_indicator: {
                            'vertices': each_polyline_or_polygon
                        }
                    }
                }
            })

        write_to_tar(sample, output_root / f'3d_{hdmap_name_in_cosmos}' / f'{clip_id}.tar')

def convert_waymo_pose(output_root: Path, clip_id: str, dataset: tf.data.TFRecordDataset):
    """
    read all frames and convert the pose to wds format. interpolate the pose to the target fps

    Minimal required format:
        sample_camera_to_world['{frame_idx:06d}.pose.{camera_name}.npy'] = np.ndarray with shape (4, 4). opencv convention
        sample_vehicle_to_world['{frame_idx:06d}.vehicle_pose.npy'] = np.ndarray with shape (4, 4). flu convention
    """
    sample_camera_to_world = {'__key__': clip_id}
    sample_vehicle_to_world = {'__key__': clip_id}

    camera_name_to_camera_to_vehicle = {}

    # get camera_to_vehicle from the first frame
    for frame_idx, data in enumerate(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        for camera_calib in frame.context.camera_calibrations:
            camera_name = get_camera_name(camera_calib.name).lower()
            camera_to_vehicle = np.array(camera_calib.extrinsic.transform).reshape((4, 4)) # FLU convention
            camera_name_to_camera_to_vehicle[camera_name] = camera_to_vehicle

        # only process the first frame
        break

    for frame_idx, data in enumerate(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        for image_data in frame.images:
            camera_name = get_camera_name(image_data.name).lower()
            vehicle_to_world = np.array(image_data.pose.transform).reshape((4, 4))
            camera_to_vehicle = camera_name_to_camera_to_vehicle[camera_name]
            camera_to_world = vehicle_to_world @ camera_to_vehicle # FLU convention
            camera_to_world_opencv = np.concatenate(
                [-camera_to_world[:, 1:2], -camera_to_world[:, 2:3], camera_to_world[:, 0:1], camera_to_world[:, 3:4]],
                axis=1
            )
            sample_camera_to_world[f"{frame_idx * IndexScaleRatio:06d}.pose.{camera_name}.npy"] = camera_to_world_opencv

        sample_vehicle_to_world[f"{frame_idx * IndexScaleRatio:06d}.vehicle_pose.npy"] = vehicle_to_world

    # interpolate the pose to the target fps
    # source index: 0,    1,    2,    3, ..., 10
    # target index: 0,1,2,3,4,5,6,7,8,9, ..., 30,31,32
    max_target_frame_idx = frame_idx * IndexScaleRatio

    # interpolate the vehicle pose to the target fps
    for target_frame_idx in range(max_target_frame_idx):
        if f"{target_frame_idx:06d}.vehicle_pose.npy" not in sample_vehicle_to_world:
            nearest_prev_frame_idx = target_frame_idx // IndexScaleRatio * IndexScaleRatio
            nearest_prev_frame_pose = sample_vehicle_to_world[f"{nearest_prev_frame_idx:06d}.vehicle_pose.npy"]
            nearest_next_frame_idx = (target_frame_idx // IndexScaleRatio + 1) * IndexScaleRatio
            nearest_next_frame_pose = sample_vehicle_to_world[f"{nearest_next_frame_idx:06d}.vehicle_pose.npy"]
            sample_vehicle_to_world[f"{target_frame_idx:06d}.vehicle_pose.npy"] = \
                interpolate_pose(nearest_prev_frame_pose, nearest_next_frame_pose, (target_frame_idx - nearest_prev_frame_idx) / IndexScaleRatio)

    # add the last two frames
    approx_motion = sample_vehicle_to_world[f"{max_target_frame_idx:06d}.vehicle_pose.npy"] - sample_vehicle_to_world[f"{max_target_frame_idx - 1:06d}.vehicle_pose.npy"]
    approx_motion[:3, :3] = 0
    sample_vehicle_to_world[f"{max_target_frame_idx + 1:06d}.vehicle_pose.npy"] = sample_vehicle_to_world[f"{max_target_frame_idx:06d}.vehicle_pose.npy"] + approx_motion
    sample_vehicle_to_world[f"{max_target_frame_idx + 2:06d}.vehicle_pose.npy"] = sample_vehicle_to_world[f"{max_target_frame_idx:06d}.vehicle_pose.npy"] + 2 * approx_motion

    # interpolate the camera pose to the target fps
    for camera_name in CameraNames:
        for target_frame_idx in range(max_target_frame_idx):
            if f"{target_frame_idx:06d}.pose.{camera_name}.npy" not in sample_camera_to_world:
                nearest_prev_frame_idx = target_frame_idx // IndexScaleRatio * IndexScaleRatio
                nearest_prev_frame_pose = sample_camera_to_world[f"{nearest_prev_frame_idx:06d}.pose.{camera_name}.npy"]
                nearest_next_frame_idx = (target_frame_idx // IndexScaleRatio + 1) * IndexScaleRatio
                nearest_next_frame_pose = sample_camera_to_world[f"{nearest_next_frame_idx:06d}.pose.{camera_name}.npy"]
                sample_camera_to_world[f"{target_frame_idx:06d}.pose.{camera_name}.npy"] = \
                    interpolate_pose(nearest_prev_frame_pose, nearest_next_frame_pose, (target_frame_idx - nearest_prev_frame_idx) / IndexScaleRatio)

        # add the last two frames
        approx_motion = sample_camera_to_world[f"{max_target_frame_idx:06d}.pose.{camera_name}.npy"] - sample_camera_to_world[f"{max_target_frame_idx - 1:06d}.pose.{camera_name}.npy"]
        approx_motion[:3, :3]  = 0
        sample_camera_to_world[f"{max_target_frame_idx + 1:06d}.pose.{camera_name}.npy"] = sample_camera_to_world[f"{max_target_frame_idx:06d}.pose.{camera_name}.npy"] + approx_motion
        sample_camera_to_world[f"{max_target_frame_idx + 2:06d}.pose.{camera_name}.npy"] = sample_camera_to_world[f"{max_target_frame_idx:06d}.pose.{camera_name}.npy"] + 2 * approx_motion

    write_to_tar(sample_camera_to_world, output_root / 'pose' / f'{clip_id}.tar')
    write_to_tar(sample_vehicle_to_world, output_root / 'vehicle_pose' / f'{clip_id}.tar')


def convert_waymo_timestamp(output_root: Path, clip_id: str, dataset: tf.data.TFRecordDataset):
    """
    read all frames and convert the timestamp to wds format.
    """
    sample = {'__key__': clip_id}
    
    for frame_idx, data in enumerate(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        timestamp_micros = frame.timestamp_micros
        sample[f"{frame_idx * IndexScaleRatio:06d}.timestamp_micros.txt"] = str(timestamp_micros)
        
    write_to_tar(sample, output_root / 'timestamp' / f'{clip_id}.tar')

def convert_waymo_bbox(output_root: Path, clip_id: str, dataset: tf.data.TFRecordDataset):
    """
    read all frames and convert the bbox to wds format

    Minimal required format:
        sample['{frame_idx:06d}.all_object_info.json'] = {
            'object_id 1' : {
                'object_to_world' : np.ndarray with shape (4, 4),
                'object_lwh' : np.ndarray with shape (3,),
                'object_is_moving' : bool,
                'object_type' : str
            },
            'object_id 2' : {
                ...
            },
            ...
        }
    """
    sample = {'__key__': clip_id}
    min_moving_speed = 0.2

    valid_bbox_types = [
        label_pb2.Label.Type.TYPE_VEHICLE,
        label_pb2.Label.Type.TYPE_PEDESTRIAN,
        label_pb2.Label.Type.TYPE_CYCLIST
    ]

    for frame_idx, data in enumerate(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        vehicle_to_world = np.array(frame.pose.transform).reshape((4, 4))
        sample[f"{frame_idx * IndexScaleRatio:06d}.all_object_info.json"] = {}

        for label in frame.laser_labels:
            if label.type not in valid_bbox_types:
                continue

            if not label.camera_synced_box.ByteSize():
                continue

            object_id = label.id
            object_type = WaymoProto2SemanticLabel[label.type]

            center_in_vehicle = np.array([label.camera_synced_box.center_x, label.camera_synced_box.center_y, label.camera_synced_box.center_z, 1]).reshape((4, 1))
            center_in_world = vehicle_to_world @ center_in_vehicle
            heading = label.camera_synced_box.heading
            rotation_in_vehicle = R.from_euler("xyz", [0, 0, heading], degrees=False).as_matrix()
            rotation_in_world = vehicle_to_world[:3, :3] @ rotation_in_vehicle

            object_to_world = np.eye(4)
            object_to_world[:3, :3] = rotation_in_world
            object_to_world[:3, 3] = center_in_world.flatten()[:3]

            object_lwh = np.array([label.camera_synced_box.length, label.camera_synced_box.width, label.camera_synced_box.height])
            
            speed = np.sqrt(label.metadata.speed_x**2 + label.metadata.speed_y**2 + label.metadata.speed_z**2)
            object_is_moving = bool(speed > min_moving_speed)

            sample[f"{frame_idx * IndexScaleRatio:06d}.all_object_info.json"][object_id] = {
                'object_to_world': object_to_world.tolist(),
                'object_lwh': object_lwh.tolist(),
                'object_is_moving': object_is_moving,
                'object_type': object_type
            }

    write_to_tar(sample, output_root / 'all_object_info' / f'{clip_id}.tar')



def convert_terasim_to_wds(
    terasim_record_root: Union[str, Path],
    output_wds_path: Union[str, Path],
    single_camera: bool = False
):
    terasim_record_path = Path(terasim_record_root)
    clip_id = terasim_record_path.stem
    output_wds_path = Path(output_wds_path)

    if not terasim_record_path.exists():
        raise FileNotFoundError(f"Terasim record file not found: {terasim_record_path}")
    
    dataset = TeraSim_Dataset(terasim_record_root)

    convert_terasim_hdmap(output_wds_path, clip_id, dataset)
    convert_terasim_intrinsics(output_wds_path, clip_id, dataset)
    convert_terasim_pose(output_wds_path, clip_id, dataset)
    convert_terasim_timestamp(output_wds_path, clip_id, dataset)
    convert_terasim_bbox(output_wds_path, clip_id, dataset)

@click.command()
@click.option("--terasim_record_root", "-i", type=str, help="Terasim record root", default="/home/mtl/cosmos-av-sample-toolkits/terasim_demo")
@click.option("--output_wds_path", "-o", type=str, help="Output wds path", default="/home/mtl/cosmos-av-sample-toolkits/terasim_demo_render_ftheta")
@click.option("--num_workers", "-n", type=int, default=1, help="Number of workers")
@click.option("--single_camera", "-s", type=bool, default=False, help="Convert only front camera")
def main(terasim_record_root: str, output_wds_path: str, num_workers: int, single_camera: bool):
    all_filenames = list(Path(terasim_record_root).iterdir())
    print(f"Found {len(all_filenames)} TeraSim records")
    for filename in all_filenames:
        convert_terasim_to_wds(filename, output_wds_path, single_camera)
    # with ProcessPoolExecutor(max_workers=num_workers) as executor:
    #     futures = [
    #         executor.submit(
    #             convert_terasim_to_wds,
    #             terasim_record_root=filename,
    #             output_wds_path=output_wds_path,
    #             single_camera=single_camera
        #     ) 
        #     for filename in all_filenames
        # ]
        
        # for future in tqdm(
        #     as_completed(futures), 
        #     total=len(all_filenames),
        #     desc="Converting tfrecords"
        # ):
        #     try:
        #         future.result() 
        #     except Exception as e:
        #         print(f"Failed to convert due to error: {e}")

if __name__ == "__main__":
    main()
