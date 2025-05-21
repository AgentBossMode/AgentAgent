# You can use most Debian-based base images
FROM e2bdev/code-interpreter:latest

# Install dependencies and customize sandbox
RUN pip install "langgraph-cli[inmem]" langgraph langchain

COPY "langgraph_sample.json" "/home/user/langgraph.json"
