import pymongo
import json
import boto3
from botocore.exceptions import ClientError
from db_config import db_connection_string

mongo_client = pymongo.MongoClient(db_connection_string)
db = mongo_client["vehicle_damage"]
collection = db["vehicle_damage"]


def get_embedding(client, damage_description):
    """Get the vector embedding for the damage description."""
    

    body = json.dumps({
        "inputText": damage_description,
    })
    
    model_id = 'amazon.titan-embed-text-v1' #look for embeddings in the modelID
    accept = 'application/json' 
    content_type = 'application/json'
    
    # Invoke model 
    response = client.invoke_model(
        body=body, 
        modelId=model_id, 
        accept=accept, 
        contentType=content_type
    )
    response_body = json.loads(response['body'].read())
    return response_body.get('embedding')
    
 

def main():
    """Generate embeddings for the damage descriptions."""
    client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

    for document in collection.find():
        embedding = document.get("embedding")
        if not embedding:
            damage_description = document["description"]
            embedding = get_embedding(client, damage_description)
            collection.update_one({"_id": document["_id"]}, {"$set": {"embedding": embedding}})
            print(embedding)
        print(f"Embedding for {document['image_path']} already exists. Skipping.")


if __name__ == "__main__":
    main()
