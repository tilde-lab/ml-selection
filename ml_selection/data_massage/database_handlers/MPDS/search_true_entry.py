# Finding the largest ranges of entry, print range in format {some_key: [from, to]...}

file_with_entry = "ml_selection/structures_props/raw_mpds/s_entries_all.txt"

myfile = open(file_with_entry, mode="r", encoding="utf_8")
entrys = myfile.read().splitlines()

idxs = {"0": []}

last = 0
last_key = "0"

for i in range(len(entrys)):
    if " S" not in entrys[i]:
        continue
    entry = entrys[i].replace(" S", "")
    if int(last) + 1 == int(entry):
        idxs[str(last_key)].append(entry)
    else:
        idxs[str(entry)] = [entry]
        last_key = str(entry)

    last = int(entry)

for key in idxs.copy().keys():
    if len(idxs[key]) < 200:
        idxs.pop(key, None)

for key in idxs.keys():
    idxs[key] = [idxs[key][0], idxs[key][-1]]

print(idxs)
