import json
import pandas as pd

openfile = open('ground_truths/ground_truth.json')
jsondata = json.load(openfile)

data = pd.DataFrame([[list(i.keys())[0], list(i.values())[0]] for i in jsondata], columns=['tag', 'text'])


def handcrafted_stage(x):
    if x['tag'] == '<h2>':
        text = x['text']
        try:
            stage = text[text.find('stage') + 6:].split()[0].lower()
        except IndexError:
            return None
        print(stage)
        if stage == '0':
            return 0
        elif stage == '1' or stage == 'i':
            return 1
        elif stage == '2' or stage == 'ii':
            return 2
        elif stage == '3' or stage == 'iiia' or stage == 'iiib':
            return 3
        elif stage == '4' or stage == 'iv':
            return 4
        else:
            return None


data['stage_level'] = data.apply(lambda row: handcrafted_stage(row), axis=1)
first_idx = (data['stage_level'] == 0).to_list().index(True)
last_idx = (data['stage_level'] == 4).to_list().index(True)
# last_idx = (data['stage_level'] == 4).idxmax() --- this is okay either
the_rest = data.iloc[last_idx+1:]
last_idx = (the_rest['tag'] == '<h2>').idxmax()
data = data.iloc[first_idx:last_idx,]
stage_level = data['stage_level'].to_numpy().copy()

cache = 0
for i, value in enumerate(stage_level):
    if value >= 0:
        cache = value
    else:
        stage_level[i] = cache
data['stage_level'] = stage_level
data = data[data['tag'] != '<h2>']
data.to_csv('ground_truths/ground_truth.csv', index=False)