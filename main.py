"""
Main file to train_eval/test/predict
"""

from bert_uda.uda import *
from bert_uda.bert import *
from bert_uda.load_data import *
from bert_uda.configuration import CFG


class ModeManager:
    """
    A class to manage train, test, pred, and train_eval mode;
    The first step: run it on cpu;
    """
    def __init__(self,
                 model: nn.Module = None,
                 data_loader: DataSet = None,
                 optimizer: torch.optim = None,
                 device=None
                 ):
        """
        :param model: basic model for classification, model's parameters should be loaded before
        :param data_loader: class data_set containing sup_data and unsup_data
        :param optimizer: optimizer from torch, the parameters in model must be registered in before
        :param device: calculation device
        """
        self.device = device
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.model = model
        if self.model:
            self.model = model.to(self.device)

    def train_eval(self,
                   total_train_steps: int,
                   sup_batch_size: int,
                   unsup_ratio: float,
                   save_steps: int,
                   eval_steps: int,
                   eval_batch_size: int,
                   model_path: str,
                   uda_coeff: float,
                   tsa_schedule: str,
                   unsup_mask_confidence: float,
                   uda_softmax_temp: float,
                   ):
        """
        Purpose:
            Train and eval model with UDA method;
        Args:
            :param uda_softmax_temp:
            :param unsup_mask_confidence:
            :param tsa_schedule:
            :param uda_coeff:
            :param total_train_steps: training will stop when reached the total train steps;
            :param sup_batch_size: the size of each sup batch;
            :param unsup_ratio: unsup_batch_size / sup_batch_size;
            :param save_steps: save trained model each save steps;
            :param eval_steps: eval the model each eval steps
            :param eval_batch_size: the size of each btach input when evaluation
            :param model_path: path to save the checkpoint
        """
        # Get sup and unsup criterion
        # if reduction is "none"
        unsup_criterion = nn.KLDivLoss(reduction="none").to(device=self.device)
        sup_criterion = nn.CrossEntropyLoss(reduction="none").to(device=self.device)

        # Assert model into train mode, for there exists dropout layer
        self.model.train()

        # Initial some hyper-parameters
        global_step = 0
        loss_sum = 0.

        # Calculate epoch number
        batch_number = int(len(self.data_loader.sup_data) / sup_batch_size)
        epoch_number = int(total_train_steps / batch_number) + 1

        for epoch in range(epoch_number):
            print("****** processing epoch {} ******".format(epoch))
            ########################
            # Train for each batch #
            ########################
            # Get train batch from data loader
            iter_bar = tqdm(self.data_loader.get_train(sup_batch_size=sup_batch_size, unsup_ratio=unsup_ratio))
            for sup_batch, unsup_batch in iter_bar:
                final_loss, sup_loss, unsup_loss = get_uda_model_loss(base_model=self.model,
                                                                      sup_criterion=sup_criterion,
                                                                      unsup_criterion=unsup_criterion,
                                                                      sup_batch=sup_batch,
                                                                      unsup_batch=unsup_batch,
                                                                      num_train_steps=total_train_steps,
                                                                      global_step=global_step,
                                                                      uda_coeff=uda_coeff,
                                                                      tsa_schedule=tsa_schedule,
                                                                      unsup_mask_confidence=unsup_mask_confidence,
                                                                      uda_softmax_temp=uda_softmax_temp,
                                                                      device=self.device,
                                                                      )
                self.optimizer.zero_grad()
                final_loss.backward()
                self.optimizer.step()
                global_step += 1

                #############################
                # Show relevant information #
                #############################
                # Change torch.tensor to float
                loss_sum += final_loss.item()
                if global_step % 20 == 0:
                    print("\n"
                          "Final loss is: {}.\n"
                          "Unsup loss is: {}.\n"
                          "Sup loss is: {}\n".format(final_loss.item(),
                                                     unsup_loss.item(),
                                                     sup_loss.item(),))

                ##################################
                # Save model if reach save steps #
                ##################################
                if global_step % save_steps == 0:
                    model_file = os.path.join(model_path, str(global_step)+".pt")
                    self.save(check_point=model_file)

                #################################
                # Renew loss_sum after 20 steps #
                #################################
                if global_step % 20 == 0:
                    print("Average Loss {}".format(loss_sum / 20))
                    loss_sum = 0

                #########################
                # Eval when check steps #
                #########################
                if global_step % eval_steps == 0:
                    self._eval(batch_size=eval_batch_size, batch_number=25)

    def _eval(self, batch_size: int, batch_number: int):
        """
        Purpose:
            Used in train_eval mode to evaluate the performance of training model
        """
        print("****** Step into eval mode ******")
        self.model.eval()
        assert batch_size * batch_number < len(self.data_loader.eval_data)
        batch_iter = [batch for batch in self.data_loader.get_eval(eval_batch_size=batch_size)]
        random.shuffle(batch_iter)
        batch_iter = batch_iter[:batch_number]

        pred_labels = self.predict(batch_iter=batch_iter)
        real_labels = [label_ids.max(dim=-1)[1]
                       for _, _, _, label_ids in
                       batch_iter]
        real_labels = torch.cat(real_labels, dim=-1)

        pred_labels = pred_labels.to(self.device)
        real_labels = real_labels.to(self.device)

        compare = (pred_labels == real_labels).to(dtype=torch.float)
        accuracy = torch.sum(compare) / compare.shape[0]

        print("\n****** Evaluation finished ******")
        print("The accuracy is {}".format(accuracy))

        # Return to train mode
        self.model.train()

        return accuracy

    def test(self, test_batch_size: int):
        """
        Purpose:
            predict the label of test data
        """
        print("****** Step into test mode ******")
        self.model.eval()

        pred_labels = self.predict(batch_iter=self.data_loader.get_test(test_batch_size=test_batch_size))

        real_labels = [label_ids.max(dim=-1)[1]
                       for _, _, _, label_ids in
                       self.data_loader.get_test(test_batch_size=test_batch_size)]
        real_labels = torch.cat(real_labels, dim=-1)

        pred_labels = pred_labels.to(self.device)
        real_labels = real_labels.to(self.device)

        compare = (pred_labels == real_labels).to(dtype=torch.float)
        accuracy = torch.sum(compare) / compare.shape[0]
        print("The accuracy is {}".format(accuracy))
        print("****** Test finished ******")

        # Reset to train mode
        self.model.train()

        return accuracy, pred_labels

    def predict(self, batch_iter):
        """
        Purpose:
            predict the label of item in batch_iter
        Args:
            batch_iter: iterable data to be predicted, usually from pred_data, sup_data, eval_data, and test_data
        """
        self.model.eval()
        pred_bar = tqdm(batch_iter)

        buffer = []
        for batch in pred_bar:
            # Get inputs
            if len(batch) == 3:
                input_ids, token_type_ids, attention_mask = batch
            else:
                input_ids, token_type_ids, attention_mask, _ = batch
            input_ids = input_ids.to(device=self.device)
            token_type_ids = token_type_ids.to(device=self.device)
            attention_mask = attention_mask.to(device=self.device)
            # Prediction
            logits = self.model(input_ids, token_type_ids, attention_mask)
            pred_label = logits.max(dim=-1)[1]
            buffer.append(pred_label)

        pred_labels = torch.cat(buffer, dim=-1)
        self.model.train()
        return pred_labels

    def save(self, check_point: str):
        """
        Purpose:
            Save trained model to check_point file.
        Args:
            check_point: The path to save model.
        """
        torch.save(self.model, check_point)

    def load_model(self, model_file: str):
        """
        Purpose:
            Save trained model to check_point file.
        Args:
            model_file: The file to load model.
        """
        self.model = torch.load(model_file)
        self.model.to(device=self.device)


def main(cfg: CFG):
    """
    Purpose:
        The entrance of class;
    Args:
        mode: "train_eval", "test", "pred"
    """
    if cfg.mode == "train_eval":
        # Get model, optimizer, and data loader
        model = get_bert_classifier(n_labels=cfg.n_labels,
                                    drop_prob=cfg.drop_pob,
                                    bert_type=cfg.bert_type,
                                    ft_bert_file=cfg.ft_bert_file,
                                    )

        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=cfg.lr,
                                    momentum=cfg.momentum,
                                    )

        data_loader = DataSet()
        data_loader.load_from_file(sup_file=cfg.sup_file, unsup_file=cfg.unsup_file, test_file=None, pred_file=None)
        data_loader.split_train_eval(eval_ratio=cfg.eval_ratio)
        data_loader.restrict_data(sup_data_size=cfg.sup_data_size)

        # Create mode manager
        manager = ModeManager(model=model,
                              optimizer=optimizer,
                              data_loader=data_loader,
                              device=cfg.device,
                              )

        manager.train_eval(total_train_steps=cfg.total_train_steps,
                           sup_batch_size=cfg.sup_batch_size,
                           unsup_ratio=cfg.unsup_ratio,
                           save_steps=cfg.save_steps,
                           eval_steps=cfg.eval_steps,
                           eval_batch_size=cfg.eval_batch_size,
                           model_path=cfg.model_path,
                           uda_coeff=cfg.uda_coeff,
                           tsa_schedule=cfg.tsa_schedule,
                           unsup_mask_confidence=cfg.unsup_mask_confidence,
                           uda_softmax_temp=cfg.uda_softmax_temp,
                           )

    elif cfg.mode == "test":
        data_loader = DataSet()
        data_loader.load_from_file(test_file=cfg.test_file)
        manager = ModeManager(data_loader=data_loader, device=cfg.device)
        for model_file in os.listdir(cfg.model_path):
            model_file = os.path.join(cfg.model_path, model_file)
            manager.load_model(model_file=model_file)
            manager.test(test_batch_size=cfg.test_batch_size)

    elif cfg.mode == "pred":
        data_loader = DataSet()
        data_loader.load_from_file(pred_file=cfg.pred_file)
        manager = ModeManager(data_loader=data_loader, device=cfg.device)
        label_dic = {}
        for model_file in os.listdir(cfg.model_path):
            model_file = os.path.join(cfg.model_path, model_file)
            manager.load_model(model_file=model_file)
            labels = manager.predict(batch_iter=manager.data_loader.get_pred(pred_batch_size=cfg.pred_batch_size))
            label_dic[model_file] = labels
        torch.save(label_dic, cfg.result_file)
        return label_dic


if __name__ == "__main__":
    cfg = CFG()
    main(cfg=cfg)
