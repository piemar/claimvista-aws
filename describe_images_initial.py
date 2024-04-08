import base64
import json
import os
import random
import json
import logging
import pymongo
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
from db_config import db_connection_string

mongo_client = pymongo.MongoClient(db_connection_string)
db = mongo_client["vehicle_damage"]
collection = db["vehicle_damage"]  
    
def process_image(client, base64_image_data):
    # Invoke the model with the prompt and the encoded image
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Can you describe the damage to the vehicle, including a title and the severity (categorized as low, medium or high)? Please return json instead of text. The json structure should use the headings 'title', 'description', and 'severity'."
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_image_data,
                        },
                    },
                ],
            }
        ],
    }

    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
        )
        
        # Process and print the response
        result = json.loads(response.get("body").read())
        resp_json=json.loads(result.get("content")[0]["text"].replace("\n", "").replace("`", ""))
        resp_json["severity"] = resp_json["severity"].lower()        
        return resp_json;
    
    except ClientError as err:
        logger.error(
            "Couldn't invoke Claude 3 Sonnet. Here's why: %s: %s",
            err.response["Error"]["Code"],
            err.response["Error"]["Message"],
        )
        raise


def encode_image(image_path):
    """Encode the image as a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def estimate_cost(severity):
    """Estimate the cost of the damage based on the severity."""
    if severity == "low":
        return random.randint(300, 1500)
    elif severity == "medium":
        return random.randint(1000, 5000)
    else:
        return random.randint(3000, 20000)


def image_exists(image_path):
    """Check if the image already exists in the database."""
    doc = collection.find_one({"image_path": image_path})
    return bool(doc)


def main():
    """Main function to process the images and store the data in the database."""
    client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    images = os.listdir("./dataset")
    for image_path in images:
        if image_exists(image_path):
            print(f"Image {image_path} already exists in the database")
        else:
            relative_path = os.path.join("./dataset", image_path)
            base64_image = encode_image(relative_path)
            image_data = process_image(client, base64_image)
            image_data["image_path"] = image_path
            image_data["image_base64"] = base64_image
            image_data["cost_estimate"] = estimate_cost(image_data["severity"])
            collection.insert_one(image_data)
            print(image_data)


if __name__ == "__main__":
    main()
