"""
Class to:
    Load Raw Data from CSV file;
    Transform Raw Data into List(InputExample)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv

from absl import flags

FLAGS = flags.FLAGS


class InputExample(object):
    """ A single training/test example for simple sequence classification. """

    def __init__(self, guid, text_a, text_b=None, label=None):
        """
        Purpose:
            Constructs a InputExample.
        Args:
            guid:   Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
                    sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
                    Only must be specified for sequence pair tasks.
            label:  (Optional) string. The label of the example. This should be
                    specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.label = label
        self.text_a = text_a
        self.text_b = text_b
        self.word_list_a = None
        self.word_list_b = None


class DataProcessor(object):
    """ Base class for data converters for sequence classification data sets. """
    def __init__(self, raw_data_dir):
        self.raw_data_dir = raw_data_dir

    def get_train_examples(self):
        """ Gets a collection of supervised `InputExample`s for the train set. """
        raise NotImplementedError()

    def get_test_examples(self):
        """ Gets a collection of `InputExample`s for the test set. """
        raise NotImplementedError()

    def get_unsup_examples(self):
        """ Gets a collection of unsupervised `InputExample`s for the train set """
        raise NotImplementedError()

    def get_pred_examples(self):
        """ Gets a collection of `InputExample`s for the predicting set """
        raise NotImplementedError()

    def get_labels(self):
        """ Gets the list of labels for this data set. """
        raise NotImplementedError()

    def get_train_size(self):
        """ Gets the size of train data """
        raise NotImplementedError()

    @classmethod
    # This function remain to be revised for torch use.
    def _read_csv(cls, input_file, quotechar=None, delimiter="\t"):
        """ Reads a tab separated value file. """
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


def clean_web_text(st):
    """
    Purpose:
        Clean text with web notation.
    Return:
        str: sentence
    """
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
    st = st.replace("\\n", " ")
    st = st.replace("\\", " ")
    while "  " in st:
        st = st.replace("  ", " ")
    return st


class IMDbProcessor(DataProcessor):
    """
    Purpose:
        Processor for the CoLA data set (GLUE version).
    Args:
        raw_data_dir: the dir contains train, test, and un-sup data;
        unsup_set: "unsup_ext" or "unsup_in";
                    if "unsup_ext", get data from unsup_ext.csv;
                    if "unsup_in", get data form train.csv;
    """
    def __init__(self, raw_data_dir):
        super(IMDbProcessor, self).__init__(raw_data_dir=raw_data_dir)

    def get_train_examples(self):
        """ See base class. """
        return self._create_examples(lines=self._read_csv(os.path.join(self.raw_data_dir, "train.csv"), quotechar='"'),
                                     set_type="train")

    def get_test_examples(self):
        """ See base class."""
        return self._create_examples(lines=self._read_csv(os.path.join(self.raw_data_dir, "test.csv"), quotechar='"'),
                                     set_type="test")

    def get_pred_examples(self):
        """ See base class."""
        return self._create_examples(lines=self._read_csv(os.path.join(self.raw_data_dir, "pred.csv"), quotechar='"'),
                                     set_type="pred")

    def get_unsup_examples(self, unsup_set="unsup_in"):
        """ See base class. """
        if unsup_set == "unsup_ext":
            return self._create_examples(lines=self._read_csv(os.path.join(self.raw_data_dir, "unsup_ext.csv"), quotechar='"'),
                                         set_type="unsup_ext", skip_unsup=False)
        elif unsup_set == "unsup_in":
            return self._create_examples(lines=self._read_csv(os.path.join(self.raw_data_dir, "train.csv"), quotechar='"'),
                                         set_type="unsup_in", skip_unsup=False)

    def get_labels(self):
        """ See base class. """
        return ["pos", "neg"]

    def _create_examples(self, lines, set_type, skip_unsup=True):
        """
        Purpose:
            Creates examples for the training and dev sets;
            Get all the examples in the data set except these longer than 500 words;
            If skip_unsup, skip all the data item with label "unsup";
        Return:
            list[InputExample]
        """
        examples = []
        for (i, line) in enumerate(lines):
            # The first line is the table title.
            if i == 0:
                continue
            if skip_unsup and line[1] == "unsup":
                continue
            if line[1] == "unsup" and len(line[0]) < 500:
                continue
            guid = "%s-%s" % (set_type, line[2])
            text_a = line[0]
            label = line[1]
            text_a = clean_web_text(text_a)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def get_train_size(self):
        return 25000

    def get_dev_size(self):
        return 25000


class DPPProcessor(DataProcessor):
    """
    Purpose:
        Processor for the CoLA data set (GLUE version).
    Args:
        raw_data_dir: the dir contains train, test, and un-sup data;
        unsup_set: "unsup_ext" or "unsup_in";
                    if "unsup_ext", get data from unsup_ext.csv;
                    if "unsup_in", get data form train.csv;
    """
    def __init__(self, raw_data_dir):
        super(DPPProcessor, self).__init__(raw_data_dir=raw_data_dir)

    def get_train_examples(self):
        """ See base class. """
        return self._create_examples(lines=self._read_csv(os.path.join(self.raw_data_dir, "train.csv"), quotechar='"'),
                                     set_type="train")

    def get_test_examples(self):
        """ See base class."""
        return self._create_examples(lines=self._read_csv(os.path.join(self.raw_data_dir, "test.csv"), quotechar='"'),
                                     set_type="test")

    def get_pred_examples(self):
        """ See base class."""
        return self._create_examples(lines=self._read_csv(os.path.join(self.raw_data_dir, "pred.csv"), quotechar='"'),
                                     set_type="pred")

    def get_unsup_examples(self, unsup_set="unsup_in"):
        """ See base class. """
        if unsup_set == "unsup_ext":
            return self._create_examples(lines=self._read_csv(os.path.join(self.raw_data_dir, "unsup_ext.csv"), quotechar='"'),
                                         set_type="unsup_ext", skip_unsup=False)
        elif unsup_set == "unsup_in":
            return self._create_examples(lines=self._read_csv(os.path.join(self.raw_data_dir, "train.csv"), quotechar='"'),
                                         set_type="unsup_in", skip_unsup=False)

    def get_labels(self):
        """ See base class. """
        return ["pos", "neg"]

    def _create_examples(self, lines, set_type, skip_unsup=True):
        """
        Purpose:
            Creates examples for the training and dev sets;
            Get all the examples in the data set except these longer than 500 words;
            If skip_unsup, skip all the data item with label "unsup";
        Return:
            list[InputExample]
        """
        examples = []
        for (i, line) in enumerate(lines):
            # The first line is the table title.
            if i == 0:
                continue
            if skip_unsup and line[1] == "unsup":
                continue
            if line[1] == "unsup" and len(line[0]) < 500:
                continue
            guid = "%s-%s" % (set_type, line[2])
            text_a = line[0]
            label = line[1]
            text_a = clean_web_text(text_a)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def get_train_size(self):
        return 25000

    def get_dev_size(self):
        return 25000


def get_processor(task_name):
    """
    Purpose:
        Ensemble the processors in this python file;
        Return relevant processor according to task name;
    Return:
        Implemented instance of relevant processor class;
    """
    task_name = task_name.lower()
    if task_name == "imdb":
        raw_data_dir = "{}/{}/{}".format("../data/raw_data", "IMDB_raw", "csv")
    elif task_name == "dpp":
        raw_data_dir = "{}/{}/{}".format("../data/raw_data", "DPP_raw", "csv")
    else:
        raw_data_dir = ""

    processors = {
        "imdb": IMDbProcessor,
        "dpp": DPPProcessor,
    }
    processor = processors[task_name](raw_data_dir)
    return processor
