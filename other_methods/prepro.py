"""
Extract labeled data
"""
import csv
from other_methods.nlp_tool import *
from other_methods.utils import *


def seg_data(file_name):
    # Load data
    with open(file_name, "r") as rf:
        csv_reader = csv.reader(rf)
        content = [row for row in csv_reader]

    # Load stop words
    stop_words = load_gazetteer("data/stopwords/self_stopwords.txt")

    # Seg data
    nlp = NLPTool()
    seg_content = []
    for stc, label in content:
        seg_stc = nlp.seg(stc)
        tmp_stc = []
        for word in seg_stc:
            if word not in stop_words:
                tmp_stc.append(word)
        seg_content.append([label, "\t".join(tmp_stc)])

    # Dump data
    with open(file_name[:-4]+"_seg.csv", "w") as wf:
        csv_writer = csv.writer(wf)
        for item in seg_content:
            csv_writer.writerow(item)


if __name__ == '__main__':
    # seg_data()
    pass
