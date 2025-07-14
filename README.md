# Promptius AI

Repository of experiments:

1. This repo is set up to use uv. Run uv sync to install require deps:
```
uv sync
```
Or, if you prefer not to use uv:
```
pip install
```

2. To run langgraph server, go to root folder and run the following

```
uv run langgraph dev
```

3. To run uts 
```
pytest -s tests
```