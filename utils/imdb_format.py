""" Read all data in IMDB and merge them to a csv file. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
from absl import app
from absl import flags

# Define parameters from out-scope
# Para-Name, Undefined Value, Explain
FLAGS = flags.FLAGS
flags.DEFINE_string("raw_data_dir", "../data/IMDB_raw/aclImdb", "raw data dir")
flags.DEFINE_string("output_dir", "../data/IMDB_raw/csv", "output_dir")
flags.DEFINE_string("train_id_path", "../data/IMDB_raw/train_id_list.txt", "path of id list")


def dump_raw_data(contents: list, file_path: str):
    """ Dump contents into file_path """
    with open(file_path, "w") as ouf:
        writer = csv.writer(ouf, delimiter="\t", quotechar="\"")
        for line in contents:
            writer.writerow(line)


def clean_web_text(st: str):
    """ Clean text in web format to normal text. """
    st = st.replace("<br />", " ")
    st = st.replace("&quot;", "\"")
    st = st.replace("<p>", " ")
    if "<a href=" in st:
        while "<a href=" in st:
            start_pos = st.find("<a href=")
            end_pos = st.find(">", start_pos)
            if end_pos != -1:
                st = st[:start_pos] + st[end_pos + 1:]
            else:
                print("incomplete href")
                print("before", st)
                st = st[:start_pos] + st[start_pos + len("<a href=")]
                print("after", st)

        st = st.replace("</a>", "")
    st = st.replace("\n", " ")
    st = st.replace("\\", " ")
    while "  " in st:
        st = st.replace("  ", " ")
    return st


def load_data_by_id(sub_set: str, id_path: str):
    """
    Purpose:
        id_path: the file contains all the ids of examples;
    Return:
        Contents: List[(content, label, uid)]
    """
    with open(id_path) as inf:
        id_list = inf.readlines()
    # id in format: Label_FileName;
    contents = []
    for example_id in id_list:
        example_id = example_id.strip()
        label = example_id.split("_")[0]
        # Data structure: "train/test" / "neg/pos/unsup" / "txt_id";
        # sub_set is "train" or "test";
        file_path = os.path.join(FLAGS.raw_data_dir, sub_set, label, example_id[len(label) + 1:])
        with open(file_path) as inf:
            st_list = inf.readlines()
            assert len(st_list) == 1
            st = clean_web_text(st_list[0].strip())
            contents += [(st, label, example_id)]
    return contents


def load_all_data(sub_set: str):
    """
    Purpose:
        Load all the IMDB data from sub set;
    Return:
        Contents: List[(content, label, uid)]
    """
    contents = []
    for label in ["pos", "neg", "unsup"]:
        data_path = os.path.join(FLAGS.raw_data_dir, sub_set, label)
        if not os.path.exists(data_path):
            continue
        # Each item in data_path is an example;
        # The file content is the content;
        # The data_path is the label;
        # Id is formed by label and filename;
        for filename in os.listdir(data_path):
            file_path = os.path.join(data_path, filename)
            with open(file_path) as inf:
                st_list = inf.readlines()
                assert len(st_list) == 1
                st = clean_web_text(st_list[0].strip())
                example_id = "{}_{}".format(label, filename)
                contents += [(st, label, example_id)]
    return contents


def main(_):
    # Load train by method load_data_by_id;
    # Dump all the data into output_dir/train.csv
    header = ["content", "label", "id"]
    contents = load_data_by_id("train", FLAGS.train_id_path)
    try:
        os.mkdir(FLAGS.output_dir)
    except FileExistsError:
        pass
    dump_raw_data(
        [header] + contents,
        os.path.join(FLAGS.output_dir, "train.csv"),
    )
    # Load test data by method load_all_data;
    # Dump all the data into output_dir/test.csv
    contents = load_all_data("test")
    dump_raw_data(
        [header] + contents,
        os.path.join(FLAGS.output_dir, "test.csv"),
    )
    # Structure of train/test file
    # "content, label, id"
    # "


if __name__ == "__main__":
    app.run(main)
