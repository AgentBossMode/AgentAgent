import requests
import json
import pickle

url = "https://zt2bymp3d5izmx2ehyay3rl4i40aqlqm.lambda-url.us-east-1.on.aws/api/story"
headers = {"Content-Type": "application/json"}
data = {"topic": "a friendly dragon"}

# For regular response
# response = requests.get(url, headers=headers, json=data)
# print(response.text)

# For streaming response
response = requests.get(url, headers=headers, json=data, stream=True)
outputs = ""

for chunk in response.iter_content(chunk_size=1024):
    output = pickle.loads(chunk)
    print(output)
