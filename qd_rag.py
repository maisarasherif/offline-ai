import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(url="http://localhost:6333")

if not client.collection_exists(collection_name="pms"):
    client.create_collection(
        collection_name="pms",
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )


dummy_data = [
    "Porto Marine Services (PMS) is a company in Abu Dhabi",
    "PMS is a Leading Marine Solutions Provider",
    "Maysara Sherif is an Engineer who works at PMS",
    "Maysara Sherif have been at PMS for 8 months",
]

def generate_response(prompt: str):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "deepseek-r1:8b",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

def main():

    

    for i, text in enumerate(dummy_data):
        response = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": "mxbai-embed-large", "input": text},
        )
        data = response.json()
        embeddings = data["embeddings"][0]
        client.upsert(
            collection_name="pms",
            wait=True,
            points=[PointStruct(id=i, vector=embeddings, payload={"text": text})],
        )
    
    
    prompt = input("Enter a prompt: ")
    
    adjusted_prompt = f"Represent this sentence for searching relevant passages: {prompt}"

    response = requests.post(
             "http://localhost:11434/api/embed",
             json={"model": "mxbai-embed-large", "input": adjusted_prompt},
         )
    data = response.json()
    embeddings = data["embeddings"][0]

    search_result = client.query_points(
        collection_name="pms",
        query=embeddings,
        with_payload=True,
        limit=3
    )

    relevant_passages = "\n".join([f"- {point.payload['text']}" for point in search_result.points])
    augmented_prompt = f"""
        the following are relevant passages: 
        <retrieved-data>
        {relevant_passages}.
        </retrieved-data>
        
        Here is the original user prompt, Answer with the help of retrieved passages: 
        <user-prompt>
        {prompt}
        </user-prompt>
    """

    response = generate_response(augmented_prompt)
    print(response)




if __name__ == "__main__":
    main()