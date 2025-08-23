from langchain_core.runnables import RunnableConfig


def check_is_test(config: RunnableConfig):
    if "type" in config and config["type"] == "test":
        is_test =True
    elif "metadata" in config and "type" in config["metadata"] and config["metadata"]["type"] == "test":
        is_test = True
    else:
        is_test = False
    return is_test

