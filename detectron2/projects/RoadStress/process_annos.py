import json
import re

def combine_annos(anno1, anno2, model_id):
    with open(anno1) as f1:
        annos1 = json.load(f1)

    with open(anno2) as f2:
        annos2 = json.load(f2)
    
    result_annos = {**annos2, **annos1}
    with open('combined_annos_' + model_id + '.json', "w") as f:
        f.write(json.dumps(result_annos, separators=(',', ':')))
    f.close()

    str = ""
    with open('combined_annos_' + model_id + '.json', "r") as f:
        str = f.read()
        re.sub('\n', '', str)
        re.sub(' ', '', str)

    with open('combined_annos_' + model_id + '.json', "w") as f:
        f.write(str)

    return 'combined_annos_' + model_id + '.json'