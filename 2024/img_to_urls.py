import os
import requests
import time
import sys

CLIENT_ID ="client_id"
def check_rate_limit():
    headers = {
        'Authorization': f'Client-ID {CLIENT_ID}'
    }
    response = requests.get("https://api.imgur.com/3/credits", headers=headers)
    if response.status_code == 200:
        print(response.json())
    else:
        print(f"Error {response.status_code}: {response.text}")

check_rate_limit()


def upload_image_to_imgur(image_path, max_retries=3):
    headers = {
        'Authorization': f'Client-ID {CLIENT_ID}'
    }
    
    retries = 0
    while retries < max_retries:
        try:
            with open(image_path, 'rb') as image_file:
                files = {
                    'image': image_file
                }
                response = requests.post(
                    url="https://api.imgur.com/3/upload",
                    headers=headers,
                    files=files
                )
                
            if response.status_code == 200:
                return response.json()['data']['link']
            elif response.status_code == 429:
                print(f"Error {response.status_code}: {response.text}. Waiting before retry...")
                time.sleep(30)
                retries += 1
            else:
                print(f"Error {response.status_code}: {response.text}")
                return None
        except Exception as e:
            print(f"Exception occurred: {e}")
            return None

    sys.exit("Too many retries. Program exiting.") 


def upload_images_from_directory(image_dir, completed_file='completed_box_images.txt', urls_file='box_image_urls.txt'):
    image_urls = []

    if os.path.exists(completed_file):
        with open(completed_file, 'r') as f:
            uploaded_images = set(f.read().splitlines())
    else:
        uploaded_images = set()

    image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))], key=lambda x: int(x.split('.')[0]))

    for i, image_filename in enumerate(image_filenames):
        if image_filename not in uploaded_images:
            image_path = os.path.join(image_dir, image_filename)
            print(f"Uploading {image_filename}... ({i+1}/{len(image_filenames)})")
            image_url = upload_image_to_imgur(image_path)

            if image_url:
                image_urls.append(image_url)
                print(f"Uploaded: {image_url}")

                with open(completed_file, 'a') as f:
                    f.write(f"{image_filename}\n")

                with open(urls_file, 'a') as f:
                    f.write(f"{image_url},\n")
            else:
                sys.exit(f"Failed to upload: {image_filename}")
            time.sleep(10)
    
    return image_urls

image_dir = "path_to_image_dir"

image_urls = upload_images_from_directory(image_dir)

print("모든 이미지 업로드 완료.")
