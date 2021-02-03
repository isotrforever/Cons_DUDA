"""
Class to define and manage configuration
"""
import os
import torch


class CFG:
    def __init__(self):
        ###################
        # Task Definition #
        ###################
        self.task_name = "dpp"
        # Mode contains "train_eval", "test" and "pred"
        self.mode = "train_eval"

        ####################
        # Data Preparation #
        ####################
        # Data Augmentation
        # Agu_ops includes "unif-0.3", tf_idf-0.3", "bt"
        # back_translation_file contains translated data in text file
        self.aug_ops = "tf_idf-0.1"
        # Input_length is the length in bert mode
        self.input_length = 128

        #############
        # UDA Model #
        #############
        # Bert model definition
        # Load from pre-trained model by hugging face: "bert-base-uncased", "bert-base-chinese"
        self.bert_type = "bert-base-chinese"
        # Load fine tune bert model in torch version
        self.ft_bert_file = None

        # Bert based classifier
        self.n_labels = 2
        self.drop_pob = 0.1

        #########################################
        # Train_eval, test, and prediction mode #
        #########################################
        # Data batch size
        self.sup_data_size = 100
        self.sup_batch_size = 8
        self.eval_batch_size = 4
        self.test_batch_size = 4
        self.pred_batch_size = 4
        # Action steps
        self.eval_steps = 500
        self.save_steps = 1000
        self.total_train_steps = 6000

        # Train and eval model
        # Split sup data into eval and train data in eval_ratio
        self.eval_ratio = 0.1
        # Augmentation trick parameters
        self.uda_softmax_temp = 0.85
        self.unsup_mask_confidence = 0.65
        self.tsa_schedule = "linear_schedule"
        # Train configuration
        self.uda_coeff = 1.
        self.unsup_ratio = 3.
        # Optimizer
        self.lr = 1e-4
        self.momentum = 0.9
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ##########################
        # Generate relevant file #
        ##########################
        path_dic = {"imdb": "./data/raw_data/IMDB_raw", "dpp": "./data/raw_data/DPP_raw"}
        # Data file is the file for the input of train, test, and pred file
        self.sup_file = os.path.join(path_dic[self.task_name], "sup", "train_bert_input.pt")
        self.unsup_file = os.path.join(path_dic[self.task_name], "unsup", self.aug_ops, "unsup_bert_input.pt")
        self.test_file = os.path.join(path_dic[self.task_name], "sup", "test_bert_input.pt")
        self.pred_file = os.path.join(path_dic[self.task_name], "pred", "pred_bert_input.pt")
        # Prediction result
        self.result_file = os.path.join(path_dic[self.task_name], "pred", "pred_bert_output.pt")

        # Back translation file
        self.back_translation_file = os.path.join(path_dic[self.task_name], "back_translation", "para.txt")

        # File to save trained model
        self.model_path = os.path.join(path_dic[self.task_name], "ckpt")
