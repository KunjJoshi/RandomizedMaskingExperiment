import os

rmft_folder = 'results/gpt_rmft'
files = [os.path.join(rmft_folder, file) for file in os.listdir(rmft_folder)]

for file in files:
    if 'webcrawl' in file:
        os.remove(file)


files = [os.path.join(rmft_folder, file) for file in os.listdir(rmft_folder)]
for file in files:
    if 'wbecrawl' in file:
        new_file = file.replace('wbecrawl', 'webcrawl')
        os.rename(file, new_file)

