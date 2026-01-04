# Promptius AI

Repository of experiments:

1. This repo is set up to use uv. Run `uv sync` to install required dependencies:
    ```bash
    uv sync
    ```
    Or, if you prefer not to use uv:
    ```bash
    pip install -r requirements.txt
    ```

2. To get the environment variables file, copy the .env.example file to a new .env file, and replace placeholders with your keys

3. To run langgraph server, go to the root folder and run the following:
    ```bash
    uv run langgraph dev
    ```

4. To run unit tests:
    ```bash
    pytest -s tests
    ```