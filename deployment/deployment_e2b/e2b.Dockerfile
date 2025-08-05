# You can use most Debian-based base images
FROM e2bdev/code-interpreter:latest

# Install dependencies and customize sandbox
RUN pip install "langgraph-cli[inmem]" langgraph langchain copilotkit langchain-openai
COPY "langgraph_sample.json" "/home/user/langgraph.json"
WORKDIR /home/user/server
COPY "index.js" "/home/user/server/index.js"
COPY "package.json" "/home/user/server/package.json"
RUN npm install
