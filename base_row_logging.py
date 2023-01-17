import requests
from datetime import datetime

API_KEY = 'UgGXclbc69HKdMQhcVAFZ9ZQTL0UW0uS'

data_dict = {"/home/nawaf/NAS/vision/Nawaf/OCR/Experiments_outputs/trdg_training_results":1,
             "/home/nawaf/NAS/vision/Nawaf/OCR/Experiments_outputs/EvArEST_Eval_lmdb":2,
             "/home/nawaf/NAS/vision/Nawaf/OCR/Experiments_outputs/trdg_random_dataset":3}

def authenticate():
    request = requests.post(
    "http://10.62.54.24",
    headers={
        "Authorization": f"{API_KEY}"
    })
    return request

def create_row(json):
    
    for k,v in json.items():
        json[k] = str(v)    
    json["train_data"] = [data_dict[json["train_data"]]]
    json["valid_data"] = [data_dict[json["valid_data"]]]

    added_row = requests.post(
    "http://localhost/api/database/rows/table/201/?user_field_names=true",
    headers={
        "Authorization": f"Token {API_KEY}",
        "Content-Type": "application/json"
    },
    json=json)
    row_id = added_row.json()["id"]
    
    return row_id

def get_row():
    
    pass 

def update_row():
    
    pass