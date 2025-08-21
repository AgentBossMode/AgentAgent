
def get_filtered_file(file: str):
    if file.startswith("```python"):
            file = file[len("```python"):]
    if file.endswith("```"):
        file = file[:-len("```")]
    return file
