import json
import jsonConverter

if __name__ == '__main__':
    with open('source/train.json') as f:
        train = json.load(f)
    with open('source/test.json') as f2:
        test = json.load(f2)

    print(len(train))
    print(len(test))