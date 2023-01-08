import json

with open('./train.jsonl', 'r', encoding="utf8") as json_file:
    json_list_t = list(json_file)

with open('./validation.jsonl', 'r', encoding="utf8") as json_file:
    json_list_v = list(json_file)

data = []

for json_str in json_list_t:
    result = json.loads(json_str)
    data.append(result)

for json_str in json_list_v:
    result = json.loads(json_str)
    data.append(result)

with open('./all_data.json', 'w+', encoding="utf8") as dest:
    dest.write(json.dumps(data))