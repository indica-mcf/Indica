import json

with open("/home/ciaran.jones/PycharmProjects/Indica/trends/info_dict.json") as jsonFile:
    data = json.load(jsonFile)

    print(data)

    for key in data:
        print(key)
