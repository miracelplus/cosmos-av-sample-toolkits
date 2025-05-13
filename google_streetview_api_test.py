import dotenv
import os
import requests
from openai import OpenAI
from PIL import Image
import io
import base64

# Load environment variables
dotenv.load_dotenv()

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

def analyze_image_with_gpt4(image_data):
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

def main():
    # Example coordinates (you can change these)
    latitude = 42.28035054606169
    longitude = -83.74084602616077
    
    try:
        # Get street view image
        image_data = get_street_view_image(latitude, longitude, heading=180, fov=120)

        # Save street view image
        with open('streetview_image.jpg', 'wb') as f:
            f.write(image_data)
        print("Street view image saved as 'streetview_image.jpg'")
        
        # Analyze image with GPT-4
        description = analyze_image_with_gpt4(image_data)
        
        print("Generated Environment Description:")
        print(description)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()



