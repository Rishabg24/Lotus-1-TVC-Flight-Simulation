import json

CONFIG_FILE = "config.json"
with open(CONFIG_FILE, "r") as f:
    data = json.load(f) # load motor data
