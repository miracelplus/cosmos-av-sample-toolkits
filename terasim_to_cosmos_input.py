from pathlib import Path
import sumolib
import json

import dotenv
import os
import requests
from openai import OpenAI
from PIL import Image
import io
import base64

# Load environment variables
dotenv.load_dotenv()

from convert_terasim_to_rds_hq import convert_terasim_to_wds
from render_from_rds_hq import render_sample_hdmap


# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_street_view_image(latitude, longitude, heading=0, pitch=0, fov=90):
    """
    Get a street view image from Google Street View API
    """
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    url = f"https://maps.googleapis.com/maps/api/streetview"
    params = {
        'size': '600x400',  # Image size
        'location': f'{latitude},{longitude}',
        'heading': heading,
        'pitch': pitch,
        'fov': fov,
        'key': api_key
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to get street view image: {response.status_code}")

def analyze_image_with_llm(image_data_list):
    """
    Analyze the image using GPT-4 Vision and generate environment description
    """
    # Convert image to base64
    base64_image_list = [base64.b64encode(image_data).decode('utf-8') for image_data in image_data_list]
    
    # Construct message content with multiple images
    content = [
        {
            "type": "text", 
            "text": "Please provide a detailed, description of the permanent environment and visual style as seen in this sequence of street view images. These images are captured from a moving vehicles perspective at different points along a real-world route, similar to a virtual drive-through using Google Street View. Focus exclusively on the fixed background elements such as road layout (including straight roads, roundabouts, intersections, crossings, merging lanes), sidewalks, bicycle lanes, types of buildings or architecture, vegetation, and the overall atmosphere of the area. Describe how the environment changes as the vehicle moves along the route, noting transitions between different types of roads or urban features (for example, moving from a straight road to a roundabout, or from a residential area to a commercial zone). Ignore all temporary or moving elements such as vehicles, pedestrians, weather, or construction work. Make sure to mention any changes in perspective or viewing angle between the images, and describe the spatial relationships between key features (e.g., “a bicycle lane appears on the right,” “the road curves gently to the left,” “a row of houses lines the street on one side,” “a roundabout comes into view ahead,” etc.). Your description should read as a smooth, chronological narrative that captures the permanent visual style and setting across the entire sequence, making it easy for a video generation model to recreate the scene. Do not list the images separately; instead, weave the observations into a single, cohesive story."
        }
    ]
    
    # Add each image to the content
    for base64_image in base64_image_list:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": content
        }],
        max_tokens=500
    )
    
    return response.choices[0].message.content






folder_path = Path("terasim_dataset")
experiment_folder_list = list(folder_path.glob("*"))

for experiment_folder in experiment_folder_list:
    print(experiment_folder)

collision_record_folder_path = Path("terasim_dataset/Ann_Arbor_Michigan_USA_roundabout_9c3da1bd/raw_data/roundabout_fail_to_yield_3")

def get_vehicle_position_view(fcd_path: Path, veh_id: str, timestep_to_crash: int, timestep_interval: int = 10) -> tuple[float, float, float]:
    import xml.etree.ElementTree as ET
    tree = ET.parse(fcd_path)
    root = tree.getroot()
    all_timesteps = list(root.findall("timestep"))
    all_timesteps.sort(key=lambda x: float(x.get("time")))
    target_timestep_list = all_timesteps[-timestep_to_crash::timestep_interval]
    vehicle_position_angle_list = []
    for target_timestep in target_timestep_list:
        for vehicle in target_timestep.findall("vehicle"):
            if vehicle.get("id") == veh_id:
                vehicle_position_angle_list.append((float(vehicle.get("x")), float(vehicle.get("y")), float(vehicle.get("angle"))))
    return vehicle_position_angle_list

def get_streetview_image_and_description(collision_record_folder_path: Path):
    collision_record_file_path = collision_record_folder_path / "monitor.json"
    fcd_path = collision_record_folder_path / "fcd_all.xml"
    with open(collision_record_file_path, "r") as f:
        collision_record = json.load(f)
    map_path = collision_record_folder_path.parent.parent / "map.net.xml"
    sumo_net = sumolib.net.readNet(map_path)
    veh_1_id, veh_2_id = collision_record["veh_1_id"], collision_record["veh_2_id"]
    veh_1_obs = collision_record["veh_1_obs"]
    veh_2_obs = collision_record["veh_2_obs"]

    vehicle_position_angle_list = get_vehicle_position_view(fcd_path, veh_1_id, 40) # get 5s (50 timesteps) before crash
    # veh_2_x, veh_2_y, veh_2_angle = get_vehicle_position_view(fcd_path, veh_2_id, veh_2_obs["ego"]["timestep"])

    vehicle_lon_lat_angle_list = []
    for x, y, angle in vehicle_position_angle_list:
        lon, lat = sumo_net.convertXY2LonLat(x, y)
        vehicle_lon_lat_angle_list.append((lon, lat, angle))

    # veh_1_lon, veh_1_lat = sumo_net.convertXY2LonLat(veh_1_x, veh_1_y)
    # veh_2_lon, veh_2_lat = sumo_net.convertXY2LonLat(veh_2_x, veh_2_y)

    # veh_1_sumo_angle = veh_1_obs["ego"]["heading"]
    # veh_1_cosmos_angle = (90 - veh_1_sumo_angle) % 360

    image_data_list = []
    for i, (lon, lat, angle) in enumerate(vehicle_lon_lat_angle_list):
        image_data = get_street_view_image(lat, lon, heading=angle, fov=120)
        image_data_list.append(image_data)

        with open(collision_record_folder_path / f"streetview_image_{i}.jpg", 'wb') as f:
            f.write(image_data)
        print(f"Street view image saved as {collision_record_folder_path} / streetview_image_{i}.jpg")

    description = analyze_image_with_llm(image_data_list)
    with open(collision_record_folder_path / f"streetview_description.txt", 'w') as f:
        f.write(description)
    print(f"Environment description saved as {collision_record_folder_path} / streetview_description.txt")
    return veh_1_id, veh_2_id
        

if __name__ == "__main__":
    veh_1_id, veh_2_id = get_streetview_image_and_description(collision_record_folder_path)
    convert_terasim_to_wds(
        terasim_record_root=collision_record_folder_path,
        output_wds_path=collision_record_folder_path / "wds",
        single_camera=True,
        av_id=veh_1_id
    )
    settings = json.load(open("config/dataset_waymo.json", "r"))
    render_sample_hdmap(
        input_root=collision_record_folder_path / "wds",
        output_root=collision_record_folder_path / "render",
        clip_id=collision_record_folder_path.stem,
        settings=settings,
        camera_type="ftheta",
    )
