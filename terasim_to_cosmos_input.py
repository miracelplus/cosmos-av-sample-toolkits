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

def analyze_image_with_llm(image_data):
    """
    Analyze the image using GPT-4 Vision and generate environment description
    """
    # Convert image to base64
    base64_image = base64.b64encode(image_data).decode('utf-8')
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please describe the environment and setting of this street view image. Focus on static elements like buildings, roads, vegetation, weather conditions, and overall atmosphere. Ignore any moving objects or people. This description will be used as a prompt for video generation."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=300
    )
    
    return response.choices[0].message.content






folder_path = Path("terasim_dataset")
experiment_folder_list = list(folder_path.glob("*"))

for experiment_folder in experiment_folder_list:
    print(experiment_folder)

collision_record_folder_path = Path("terasim_dataset/Ann_Arbor_Michigan_USA_roundabout_9c3da1bd/raw_data/roundabout_fail_to_yield_3")


def get_streetview_image_and_description(collision_record_folder_path: Path):
    collision_record_file_path = collision_record_folder_path / "monitor.json"
    with open(collision_record_file_path, "r") as f:
        collision_record = json.load(f)
    map_path = collision_record_folder_path.parent.parent / "map.net.xml"
    sumo_net = sumolib.net.readNet(map_path)
    veh_1_id, veh_2_id = collision_record["veh_1_id"], collision_record["veh_2_id"]
    veh_1_obs = collision_record["veh_1_obs"]
    veh_2_obs = collision_record["veh_2_obs"]

    veh_1_obs_position = veh_1_obs["ego"]["position"]
    veh_2_obs_position = veh_2_obs["ego"]["position"]

    veh_1_lon, veh_1_lat = sumo_net.convertXY2LonLat(veh_1_obs_position[0], veh_1_obs_position[1])
    veh_1_sumo_angle = veh_1_obs["ego"]["heading"]
    veh_1_cosmos_angle = (90 - veh_1_sumo_angle) % 360

    image_data = get_street_view_image(veh_1_lat, veh_1_lon, heading=veh_1_cosmos_angle, fov=120)
    description = analyze_image_with_llm(image_data)

    with open(collision_record_folder_path / "streetview_image.jpg", 'wb') as f:
        f.write(image_data)
    print(f"Street view image saved as {collision_record_folder_path} / streetview_image.jpg")

    with open(collision_record_folder_path / "streetview_description.txt", 'w') as f:
        f.write(description)
    print(f"Environment description saved as {collision_record_folder_path} / streetview_description.txt")
    return veh_1_id, veh_2_id
        

if __name__ == "__main__":
    veh_1_id, veh_2_id = get_streetview_image_and_description(collision_record_folder_path)
    convert_terasim_to_wds(
        terasim_record_root=collision_record_folder_path,
        output_wds_path=collision_record_folder_path / "wds",
        single_camera=True,
        av_id=veh_2_id
    )
    settings = json.load(open("config/dataset_waymo.json", "r"))
    render_sample_hdmap(
        input_root=collision_record_folder_path / "wds",
        output_root=collision_record_folder_path / "render",
        clip_id=collision_record_folder_path.stem,
        settings=settings,
        camera_type="ftheta",
    )
