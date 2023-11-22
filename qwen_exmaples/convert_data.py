import json


data=[]
with open('train2.jsonl','r',encoding='utf-8' ) as f:
    for line in f.readlines():
        print(json.loads(line.strip()))
        # data.append(json.loads(line.strip()))
        sample=json.loads(line.strip())
        data.append({
            "id": "105ba62f",
            "conversations": [
                {
                    "from": "user",
                    "value": sample["instruction"]+sample["input"]
                },
                {
                    "from": "assistant",
                    "value": sample["output"]
                }
            ]
        }
        )

with open('train.json','w',encoding='utf-8') as f:
    json.dump(data,f,ensure_ascii=False,indent=4)