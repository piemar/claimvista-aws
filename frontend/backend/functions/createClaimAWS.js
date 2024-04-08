
import {
  BedrockRuntimeClient,
  InvokeModelCommand,
} from "@aws-sdk/client-bedrock-runtime";
import { TextDecoder } from 'node:util';
const AWS_REGION = "us-east-1";
const IMAGE_TO_TEXT_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0";
const TEXT_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v1";

exports = async function ({ query, headers, body }, response) {
  const awsAccessKeyId = context.values.get("aws-access-key-id");
  const awsSecretAccessKey = context.values.get("aws-secret-access-key");
  var serviceName = "mongodb-atlas";

  // Update these to reflect your db/collection
  var dbName = "vehicle_damage";
  var collName = "vehicle_damage";

  // Get a collection from the context
  var collection = context.services.get(serviceName).db(dbName).collection(collName);
  var searchResult;

  const base64Image = JSON.parse(body.text());


  var AWS = require('aws-sdk');
  AWS.config.credentials = new AWS.Credentials({ accessKeyId: awsAccessKeyId, "secretAccessKey": awsSecretAccessKey, region: "us-east-1" });
  // Create a new Bedrock Runtime client instance.
  const client = new BedrockRuntimeClient({ credentials: AWS.config.credentials, region: "us-east-1" });

  // Prepare the payload for the model.

  const imageToTextPayload =
  {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 1000,
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image",
            "source": {
              "type": "base64",
              "media_type": "image/jpeg",
              "data": base64Image
            }
          },
          {
            "type": "text",
            "text": "Can you describe the damage to the vehicle, including a title and the severity (categorized as low, medium or high)? Please return json instead of text. The json structure should use the headings 'title', 'description', and 'severity'."
          }
        ]
      }
    ]

  }

  var dataImageDescription = "";
  var dataImageSeverity = "";
  var dataImageTitle = "";
  var dataEmbeddings = "";
  const command = new InvokeModelCommand({
    modelId: IMAGE_TO_TEXT_MODEL_ID,
    contentType: "application/json",
    body: JSON.stringify(imageToTextPayload)
  });

  const apiResponse = await client.send(command);
  // Decode and return the response(s)
  const decodedResponseBody = new TextDecoder().decode(apiResponse.body);
  const responseBody = JSON.parse(decodedResponseBody);
  const responses = responseBody.content;
  if (responses.length === 1) {
    console.log(`Response: ${responses[0].text}`);
  } else {
    console.log("Returned multiple responses:");
    console.log(responses);
  }

  console.log(`\nNumber of input tokens:   ${responseBody.usage.input_tokens}`);
  console.log(`Number of output tokens: ${responseBody.usage.output_tokens}`);

  const imageDescription = await responses[0].text;
  parsedResponse = JSON.parse(imageDescription);
  dataImageDescription = parsedResponse.description;
  dataImageSeverity = parsedResponse.severity.toLowerCase();
  dataImageTitle = parsedResponse.title
  // 
  // Generate embeddings from image imageDescription
  //
  const textEmbeddingPayload = {
    "inputText": `${parsedResponse.description}`
  }

  const commandGenerateTextEmbeddings = new InvokeModelCommand({
    modelId: TEXT_EMBEDDING_MODEL_ID,
    contentType: "application/json",
    accept: "*/*",
    body: JSON.stringify(textEmbeddingPayload)
  });

  const apiResponseEmbeddings = await client.send(commandGenerateTextEmbeddings);
  console.log(JSON.stringify(apiResponseEmbeddings.body));
  // Decode and return the response(s)
  const decodedResponseEmbeddingBody = new TextDecoder().decode(apiResponseEmbeddings.body);
  const responseEmbeddingBody = JSON.parse(decodedResponseEmbeddingBody);
  const parsed_embedding = responseEmbeddingBody.embedding;


  /* Vector search */
  var pipeline = [
    {
      "$vectorSearch": {
        "index": "default",
        "path": "embedding",
        "queryVector": parsed_embedding,
        "numCandidates": 200,
        "limit": 3,
      },
    },
    {
      "$project": {
        "_id": 0,
        "description": 1,
        "severity": 1,
        "score": { "$meta": "vectorSearchScore" },
        "cost_estimate": 1
      },
    },
  ];

  try {
    searchResult = await collection.aggregate(pipeline).toArray();
  } catch (err) {
    console.log("Error occurred while executing aggregation:", err.message);

    return { error: err.message };
  }

  //get average cost_estimate of similar claims
  const avg_cost_estimate = searchResult.map((item) => item.cost_estimate).reduce((accumulator, currentValue) => accumulator + currentValue, 0) / searchResult.length
  response.setBody(
    JSON.stringify({
      message: "Image processed and description generated successfully",
      description: dataImageDescription,
      title: dataImageTitle,
      severity: dataImageSeverity,
      embedding: parsed_embedding,
      cost_estimate: avg_cost_estimate,
    })
  );

};


