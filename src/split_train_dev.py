import json

with open("data/interim/conala-train.json") as file:
    data = json.load(file)

n = len(data)
train = data[: int(n * 0.8)]
dev = data[int(n * 0.8) :]

with open("data/conala-train.json", "w") as file:
    json.dump(train, file)

with open("data/conala-dev.json", "w") as file:
    json.dump(dev, file)
