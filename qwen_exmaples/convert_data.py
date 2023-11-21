import json


data=[]
with open('train2.jsonl','r',encoding='utf-8' ) as f:
    for line in f.readlines():
        print(json.loads(line.strip()))
        data.append(json.loads(line.strip()))



with open('train.json','w',encoding='utf-8') as f:
    json.dump(data,f,ensure_ascii=False,indent=4)