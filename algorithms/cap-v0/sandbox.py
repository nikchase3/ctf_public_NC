import datetime
import os
import numpy as np
import json

a = {}
a['first'] = 1
a['second'] = 2

fn = 'test.json'
fp = './' + fn
with open(fp, 'w') as f:
    json.dump(a, f)


with open(fp, 'r') as f:
    b = json.load(f)

print(b['first'])
print(b['second'])