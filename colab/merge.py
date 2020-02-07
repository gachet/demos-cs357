import json
import os

for dir in os.listdir('.'):
    if '-' not in dir or '.ipynb' in dir:
        continue
    cells = []
    new_j = None
    for top, _, files in os.walk(dir):
        files = sorted(files)
        for file in files:
            with open(os.path.join(top, file)) as f:
                j = json.load(f)
                if new_j is None:
                    new_j = j
                else:
                    new_j['cells'].extend(j['cells'])

    with open(dir + '.ipynb', 'w') as f:
        json.dump(new_j, f)
