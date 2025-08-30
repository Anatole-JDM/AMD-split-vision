import json
from tqdm import tqdm
# d["image"] = d["id"]

# d["conversations"][0]["value"]:
# "<img><Your_Image_Dir_Path>/densecap_data/GCC_train_000189030.jpg</img> " -> "<image>\n"

# d["conversations"][i]["from"]:
# "user" -> "human"
# "assistant" -> "gpt"

file = "train_monkey_no_densecap"
# file = "train_monkey"
# file = "train_monkey_densecap"

with open(f"/work1/aroger/alexisroger/data/{file}.json", 'r') as f:
    data = json.load(f)

for i in tqdm(range(len(data))):
    data[i]["image"] = data[i]["id"]
    for j in range(len(data[i]["conversations"])):
        if data[i]["conversations"][j]["from"] == "user":
            data[i]["conversations"][j]["from"] = "human"
        elif data[i]["conversations"][j]["from"] == "assistant":
            data[i]["conversations"][j]["from"] = "gpt"
        else:
            print("ERROR    from:", data[i]["conversations"][j]["from"])

    # remove <img></img> from value
    tmp = data[i]["conversations"][0]["value"].find("</img>")
    if tmp != -1:
        data[i]["conversations"][0]["value"] = "<image>\n" + data[i]["conversations"][0]["value"][tmp+6:]
    else:
        print("ERROR    value:", data[i]["conversations"][0]["value"])

data.append(
    {
        "id": "0",
        "conversations": [
            {
                "from": "human",
                "value": "Is the zebra fat? Answer: "
            },
            {
                "from": "gpt",
                "value": "no"
            }
        ],
        "question_id": 0
    }
)

json.dump(data, open(f"/work1/aroger/alexisroger/data/{file}_2.json", 'w'), indent=4)