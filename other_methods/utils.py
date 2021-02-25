"""Basic Functions"""

import csv


# Load Sentences from Raw Corpus
def load_raw_corpus(file_name):
    with open(file_name, 'r') as rf:
        raw_corpus = list(filter(None, [stc.strip() for stc in rf.readlines()]))
    return raw_corpus


# Save Language Sequence
def write_gazetteer(file_name, gazetteer):
    with open(file_name, 'w') as wf:
        for word in gazetteer:
            wf.write(word+'\n')


# Load Language Sequence
def load_gazetteer(file_name):
    with open(file_name, 'r') as rf:
        gazetteer = [word.strip() for word in rf.readlines()]
    return gazetteer


# Save Phrase Sequence
def write_phrases(file_name, phrases):
    with open(file_name, 'w') as wf:
        for phrase in phrases:
            for word in phrase:
                wf.write(word+'\n')
            wf.write('\n')


# Load Phrase Sequence
def load_phrases(file_name):
    with open(file_name, 'r') as rf:
        lines = rf.readlines()
    phrase = []
    phrases = []
    for line in lines:
        if line is '\n':
            phrases.append(phrase)
            phrase = []
        else:
            phrase.append(line.strip())
    return phrases


# Save CSV Corpus
def write_csv(file_name, anns):
    with open(file_name, 'w') as wf:
        csv_writer = csv.writer(wf)
        for ann in anns:
            for item in ann:
                csv_writer.writerow(item)
            csv_writer.writerow(['' for _ in item])


# Load CSV Corpus
def load_csv(file_name):
    phrases = []
    phrase = []
    with open(file_name, 'r') as rf:
        csv_reader = csv.reader(rf)
        for row in csv_reader:
            if row[0] is '':
                phrases.append(phrase)
                phrase = []
            else:
                phrase.append(row)
    return phrases


# Extract annotations from corpus
def item_extraction(file_name, mode):
    stack = []
    with open(file_name, 'r') as rf:
        csv_reader = csv.reader(rf)
        buffer = ''
        for row in csv_reader:
            if row[mode] is 'B' and len(buffer) == 0:
                buffer += row[0]
            elif row[mode] is 'B' and len(buffer) > 0:
                stack.append(buffer)
                buffer = row[0]
            elif row[mode] is 'I' and len(buffer) == 0:
                print('illegal annotation')
            elif row[mode] is 'I' and len(buffer) > 0:
                buffer += row[0]
            elif len(buffer) == 0:
                continue
    stack = list(set(stack))
    return stack

