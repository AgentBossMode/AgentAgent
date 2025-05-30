# Python image to use.
FROM python:3.12-alpine

# Set the working directory to /app
WORKDIR /app

RUN apk add --no-cache libgcc gcc musl-dev python3-dev libffi-dev


# Install any needed packages specified in requirements.txt
RUN pip install "langgraph-cli[inmem]" langgraph langchain

# Copy the rest of the working directory contents into the container at /app
COPY . .

ENTRYPOINT ["entrypoint.sh"]
