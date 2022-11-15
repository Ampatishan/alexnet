import os

file_path = 'LOC_synset_mapping.txt'
names =[]
labels = []

files = open(file_path,'r')
lines = files.readlines()

for line in lines:
    words = line.split(' ')
    names.append(words[0])
    labels.append(" ".join(words[1:])[:-1])
mapping = dict(map(lambda i,j : (i,j) , names,labels))
print(mapping)

for folder in os.listdir('ILSVRC2012_img_train'):
    if folder in mapping.keys():
        os.rename(os.path.join('ILSVRC2012_img_train',folder),os.path.join('ILSVRC2012_img_train',mapping[folder]))

