"""
Create class data_loader for UDA;
With eval, sup, and unsup mode;
Match to the main file;
Remain to be updated for paired sentences classification;
"""

from bert_uda.utils.proc_data_utils import *

# Get batch size of eval data, supervised and unsupervised data
# Input:
#       unsup pair
#       sup item
# Hyper-parameters:
#       unsup_ratio: the ratio of unsup to sup
#       mode: "test", "train_eval", "pred"
# For each epoch:
#
#       train_eval mode:
#       output: sup_batch, unsup_batch, eval_batch
#           sup_input_ids, sup_segment_ids, sup_input_mask, label_ids = sup_batch
#           ori_input_ids, ori_segment_ids, ori_input_mask, aug_input_ids, aug_segment_ids, aug_input_mask = unsup_batch
#
#       test mode:
#       output: sup_batch
#
#       pred mode:
#       output: pre_batch
#           input_ids, segment_ids, input_mask = pre_batch


class DataSet:
    def __init__(self, sup_data: list = None, unsup_data: list = None, pred_data: list = None, test_data: list = None):
        """
        Args:
            sup_data: list[TorchInput]
            unsup_data: list[(TorchInput, TorchInput)]
                        TorchInput
                                |____guid
                                |____label
                                |____input_ids
                                |____attention_mask
                                |____token_type_ids
        """
        self.eval_data = None
        self.test_data = test_data
        self.pred_data = pred_data
        self.sup_data = sup_data
        self.unsup_data = unsup_data
        random.shuffle(self.sup_data) if self.sup_data else None
        random.shuffle(self.unsup_data) if self.unsup_data else None

    def split_train_eval(self, eval_ratio: float):
        """
        Purpose:
            Split eval data and train data from sup_data.
        Return:
            list [train_sup]
            list [eval_sup]
        """
        sup_len = len(self.sup_data)
        eval_sup_len = int(eval_ratio * sup_len)
        eval_sup_data = self.sup_data[:eval_sup_len]
        train_sup_data = self.sup_data[eval_sup_len:]
        self.sup_data = train_sup_data
        self.eval_data = eval_sup_data

    def get_train(self, sup_batch_size: int, unsup_ratio: float):
        """
        Purpose:
            Get sup and unsup data for train mode
        Args:
            sup_batch_size: the number of sup data in each epoch
            unsup_ratio: unsup_batch_size / sup_batch_size
        Return:
            train_sup, train_unsup for each epoch
        """
        assert len(self.unsup_data) > unsup_ratio * len(self.sup_data)
        unsup_batch_size = int(unsup_ratio * sup_batch_size)

        for i in range(int(len(self.sup_data) / sup_batch_size)):
            ####################
            # Get sup examples #
            ####################
            if (i+1)*sup_batch_size > len(self.sup_data):
                sup_examples = self.sup_data[i*sup_batch_size:]
            else:
                sup_examples = self.sup_data[i*sup_batch_size: (i+1)*sup_batch_size]
                random.shuffle(self.sup_data)

            # Transform into standard supervised bert input data
            sup_input_ids = torch.cat([sup_example.input_ids for sup_example in sup_examples], dim=0)
            sup_token_type_ids = torch.cat([sup_example.token_type_ids for sup_example in sup_examples], dim=0)
            sup_attention_mask = torch.cat([sup_example.attention_mask for sup_example in sup_examples], dim=0)
            label_ids = torch.cat([sup_example.label for sup_example in sup_examples], dim=0)
            sup_batch = (sup_input_ids, sup_token_type_ids, sup_attention_mask, label_ids)

            ######################
            # Get unsup examples #
            ######################
            if unsup_batch_size > 0:
                if (i+1)*unsup_batch_size > len(self.unsup_data):
                    unsup_examples = self.unsup_data[i*unsup_batch_size:]
                else:
                    unsup_examples = self.unsup_data[i*unsup_batch_size: (i+1)*unsup_batch_size]
                # Unsupervised data contains original examples and augmented examples
                ori_examples = [unsup_pair[0] for unsup_pair in unsup_examples]
                aug_examples = [unsup_pair[0] for unsup_pair in unsup_examples]

                ori_input_ids = torch.cat([ori_example.input_ids for ori_example in ori_examples], dim=0)
                ori_token_type_ids = torch.cat([ori_example.token_type_ids for ori_example in ori_examples], dim=0)
                ori_attention_mask = torch.cat([ori_example.attention_mask for ori_example in ori_examples], dim=0)

                aug_input_ids = torch.cat([aug_example.input_ids for aug_example in aug_examples], dim=0)
                aug_token_type_ids = torch.cat([aug_example.token_type_ids for aug_example in aug_examples], dim=0)
                aug_attention_mask = torch.cat([aug_example.attention_mask for aug_example in aug_examples], dim=0)

                unsup_batch = (ori_input_ids, ori_token_type_ids, ori_attention_mask,
                               aug_input_ids, aug_token_type_ids, aug_attention_mask)
            else:
                unsup_batch = None

            yield sup_batch, unsup_batch

    def get_test(self, test_batch_size: int = 4):
        """
        Purpose:
            Get test data for train mode
        Args:
            test_batch_size: the number of test data in each epoch
        Return:
            test_batch for each epoch
        """
        for i in range(int(len(self.test_data) / test_batch_size)):
            # Get sup examples and unsup examples
            if (i + 1) * test_batch_size > len(self.test_data):
                test_examples = self.test_data[i * test_batch_size:]
            else:
                test_examples = self.test_data[i * test_batch_size: (i + 1) * test_batch_size]

            # Transform into standard input
            test_input_ids = torch.cat([test_example.input_ids for test_example in test_examples], dim=0)
            test_token_type_ids = torch.cat([test_example.token_type_ids for test_example in test_examples], dim=0)
            test_attention_mask = torch.cat([test_example.attention_mask for test_example in test_examples], dim=0)
            label_ids = torch.cat([test_example.label for test_example in test_examples], dim=0)
            test_batch = (test_input_ids, test_token_type_ids, test_attention_mask, label_ids)

            yield test_batch

    def get_eval(self, eval_batch_size: int = 4):
        """
        Purpose:
            Get eval data for train_eval mode
        Args:
            eval_batch_size: the number of test data in each epoch
        Return:
            eval_batch for each epoch
        """
        for i in range(int(len(self.eval_data) / eval_batch_size)):
            # Get sup examples and unsup examples
            if (i + 1) * eval_batch_size > len(self.eval_data):
                eval_examples = self.eval_data[i * eval_batch_size:]
            else:
                eval_examples = self.eval_data[i * eval_batch_size: (i + 1) * eval_batch_size]

            # Transform into standard input
            eval_input_ids = torch.cat([eval_example.input_ids for eval_example in eval_examples], dim=0)
            eval_token_type_ids = torch.cat([eval_example.token_type_ids for eval_example in eval_examples], dim=0)
            eval_attention_mask = torch.cat([eval_example.attention_mask for eval_example in eval_examples], dim=0)
            label_ids = torch.cat([eval_example.label for eval_example in eval_examples], dim=0)
            eval_batch = (eval_input_ids, eval_token_type_ids, eval_attention_mask, label_ids)

            yield eval_batch

    def get_pred(self, pred_batch_size: int = 4):
        """
        Purpose:
            Get pred data for pred mode
        Args:
            pred_batch_size: the number of pred data in each epoch
        Return:
            pred_batch for each epoch
        """
        for i in range(int(len(self.pred_data) / pred_batch_size)):
            # Get sup examples and unsup examples
            if (i + 1) * pred_batch_size > len(self.pred_data):
                pred_examples = self.pred_data[i * pred_batch_size:]
            else:
                pred_examples = self.pred_data[i * pred_batch_size: (i + 1) * pred_batch_size]

            # Transform into standard input
            pred_input_ids = torch.cat([pred_example.input_ids for pred_example in pred_examples], dim=0)
            pred_token_type_ids = torch.cat([pred_example.token_type_ids for pred_example in pred_examples], dim=0)
            pred_attention_mask = torch.cat([pred_example.attention_mask for pred_example in pred_examples], dim=0)
            pred_batch = (pred_input_ids, pred_token_type_ids, pred_attention_mask)

            yield pred_batch

    def load_from_file(self, sup_file: str = None, unsup_file: str = None, test_file: str = None, pred_file: str = None):
        """
        Purpose:
            Load relevant data from pt file
        """
        if sup_file:
            self.sup_data = torch.load(sup_file)
        if unsup_file:
            self.unsup_data = torch.load(unsup_file)
        if test_file:
            self.test_data = torch.load(test_file)
        if pred_file:
            self.pred_data = torch.load(pred_file)
        if self.sup_data:
            random.shuffle(self.sup_data)
        if self.unsup_data:
            random.shuffle(self.unsup_data)

    def restrict_data(self, sup_data_size: int):
        """
        Purpose:
            Abandon the size of sup data to test the performance of UAD
        """
        pos_number, neg_number = 0, 0
        pos_examples = []
        neg_examples = []

        for example in self.sup_data:
            if example.label[0, 0] == 1:
                pos_examples.append(example)
                pos_number += 1
            elif example.label[0, 1] == 1:
                neg_examples.append(example)
                neg_number += 1
            if pos_number == sup_data_size and neg_number == sup_data_size:
                break

        if pos_number < sup_data_size or neg_number < sup_data_size:
            raise AssertionError

        self.sup_data = pos_examples + neg_examples
        random.shuffle(self.sup_data)
