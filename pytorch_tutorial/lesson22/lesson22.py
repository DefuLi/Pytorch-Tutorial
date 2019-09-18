import os

name2label = {}
for name in sorted(os.listdir(os.path.join("alldir"))):
    print(name)
    print(name2label.keys())
    name2label[name] = len(name2label.keys())

