"""
UDA Model in torch version
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_tsa_threshold(schedule: str, global_step: int, num_train_steps: int, start: int, end: int):
    # The stage of the training progress
    training_progress = torch.tensor(float(global_step) / float(num_train_steps))

    if schedule == "linear_schedule":
        threshold = training_progress
    elif schedule == "exp_schedule":
        scale = 5
        threshold = torch.exp((training_progress - 1) * scale)
    elif schedule == 'log_schedule':
        scale = 5
        threshold = torch.tensor(1) - torch.exp((-training_progress) * scale)
    else:
        threshold = 0.7

    output = threshold * (end - start) + start

    return output


def get_uda_model_loss(sup_criterion,
                       sup_batch: list,
                       global_step: int,
                       num_train_steps: int,
                       base_model: nn.Module,

                       unsup_criterion=None,
                       uda_coeff: float = None,
                       unsup_batch: list = None,

                       tsa_schedule: str = None,
                       unsup_mask_confidence: float = None,
                       uda_softmax_temp: float = -1.,

                       device=None
                       ):
    """
    Purpose:
        Get UDA model in different version;
    Return:
        Loss with calculation graph;
        Base model id;
    """
    #################################
    # Get sup_batch and unsup_batch #
    #################################
    # sup _____________none         #
    # aug ______|                   #
    # ori _____________ori          #
    #################################
    # Get sup features
    sup_input_ids, sup_token_type_ids, sup_attention_mask, label_ids = sup_batch
    # Change one-hot label_ids into K format
    label_ids = torch.max(label_ids, dim=-1)[1]
    # Merge unsup and sup features
    if unsup_batch:
        ori_input_ids, ori_token_type_ids, ori_attention_mask, aug_input_ids, aug_token_type_ids, aug_attention_mask = unsup_batch
        # Cat labeled data and augmented data
        input_ids = torch.cat((sup_input_ids, aug_input_ids), dim=0)
        token_type_ids = torch.cat((sup_token_type_ids, aug_token_type_ids), dim=0)
        attention_mask = torch.cat((sup_attention_mask, aug_attention_mask), dim=0)
    else:
        input_ids = sup_input_ids
        token_type_ids = sup_token_type_ids
        attention_mask = sup_attention_mask

    #########################################################
    # Ensure that all the data and model on the same device #
    #########################################################
    if not device:
        device = input_ids.device

    input_ids = input_ids.to(device=device)
    token_type_ids = token_type_ids.to(device=device)
    attention_mask = attention_mask.to(device=device)
    label_ids = label_ids.to(device=device)
    base_model.to(device=device)

    if unsup_batch:
        ori_input_ids = ori_input_ids.to(device=device)
        ori_token_type_ids = ori_token_type_ids.to(device=device)
        ori_attention_mask = ori_attention_mask.to(device=device)

    #####################################
    # Calculate sup loss and unsup loss #
    #####################################
    # Calculate the result of supervised and augmented
    logits = base_model(input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask)

    ################
    # Get sup loss #
    ################
    # Sup_loss shape: sup_batch_size
    sup_size = label_ids.shape[0]
    sup_loss = sup_criterion(logits[:sup_size], label_ids)

    if tsa_schedule:
        # Calculate the tsa threshold at current train step
        tsa_threshold = get_tsa_threshold(schedule=tsa_schedule,
                                          global_step=global_step,
                                          num_train_steps=num_train_steps,
                                          start=1./logits.shape[-1],
                                          end=1)
        ###############################################################
        # Augmentation trick:                                         #
        # If sup loss larger than the threshold;                      #
        # loss mask is 0, sup_loss is 0;                              #
        # else keep the sup_loss;                                     #
        ###############################################################
        # larger_than_threshold shape: train_batch_size;
        larger_than_threshold = torch.exp(-sup_loss) > tsa_threshold
        # loss_mask: shape = 1
        # loss mask is a leaf in the calculation tree
        loss_mask = torch.ones_like(label_ids,
                                    dtype=torch.float32) * (torch.tensor(1) -
                                                            larger_than_threshold.type(torch.float32))
        sup_loss = torch.sum(sup_loss * loss_mask, dim=-1)
    else:
        sup_loss = torch.mean(sup_loss)

    # ##################
    # # Get unsup loss #
    # ##################
    # if unsup_batch:
    #     # The parameters of part to generate original logits is anchored.
    #     with torch.no_grad():
    #         ori_logits = base_model(input_ids=ori_input_ids,
    #                                 token_type_ids=ori_token_type_ids,
    #                                 attention_mask=ori_attention_mask)
    #
    #         ############################################
    #         # Augmentation Trick                       #
    #         # Confidence masking:                      #
    #         # If the confident lower than a threshold; #
    #         # Mask the result;                         #
    #         ############################################
    #         # ori_prob size: unsup_batch_size * len(labels)
    #         ori_prob = F.softmax(ori_logits, dim=-1)
    #         if unsup_mask_confidence:
    #             # torch.max return a tuple:
    #             # tuple[0] is the largest of each line
    #             # tuple[1] is the index of the largest item
    #             # unsup_loss_mask size: unsup_batch_size, [1, 0, 0, 1, ...]
    #             unsup_loss_mask = torch.max(ori_prob, dim=-1)[0] > unsup_mask_confidence
    #             unsup_loss_mask = unsup_loss_mask.type(torch.float32)
    #         else:
    #             unsup_loss_mask = torch.ones(len(logits) - sup_size, dtype=torch.float32)
    #         # Ensure unsup_loss_mask on the same device as sup/unsup data
    #         unsup_loss_mask = unsup_loss_mask.to(device=ori_input_ids.device)
    #
    #         ######################################
    #         # Augmentation Trick                 #
    #         # Soft-max temperature controlling   #
    #         # Referring to temperature algorithm #
    #         ######################################
    #         # Get sharp parameters;
    #         # if no temperature sharpening, uda_softmax_temp equals 1;
    #         uda_softmax_temp = uda_softmax_temp if uda_softmax_temp > 0 else 1.
    #         sharp_ori_log_prob = F.log_softmax(ori_logits / uda_softmax_temp, dim=-1)
    #
    #     # Augmented examples' results
    #     aug_prob = F.softmax(logits[sup_size:], dim=-1)
    #
    #     # Calculate the KLdiv loss between ori and aug data #
    #     # Shape: unsup_batch_size
    #     unsup_loss = torch.sum(unsup_criterion(sharp_ori_log_prob, aug_prob), dim=-1)

    ##################
    # Get unsup loss #
    ##################
    if unsup_batch:
        # The parameters of part to generate original logits is anchored.
        with torch.no_grad():
            ori_logits = base_model(input_ids=ori_input_ids,
                                    token_type_ids=ori_token_type_ids,
                                    attention_mask=ori_attention_mask)

            ############################################
            # Augmentation Trick                       #
            # Confidence masking:                      #
            # If the confident lower than a threshold; #
            # Mask the result;                         #
            ############################################
            # ori_prob size: unsup_batch_size * len(labels)
            ori_prob = F.softmax(ori_logits, dim=-1)
            if unsup_mask_confidence:
                # torch.max return a tuple:
                # tuple[0] is the largest of each line
                # tuple[1] is the index of the largest item
                # unsup_loss_mask size: unsup_batch_size, [1, 0, 0, 1, ...]
                unsup_loss_mask = torch.max(ori_prob, dim=-1)[0] > unsup_mask_confidence
                unsup_loss_mask = unsup_loss_mask.type(torch.float32)
            else:
                unsup_loss_mask = torch.ones(len(logits) - sup_size, dtype=torch.float32)
            # Ensure unsup_loss_mask on the same device as sup/unsup data
            unsup_loss_mask = unsup_loss_mask.to(device=device)

            ######################################
            # Augmentation Trick                 #
            # Soft-max temperature controlling   #
            # Referring to temperature algorithm #
            ######################################
            # Get sharp parameters;
            # if no temperature sharpening, uda_softmax_temp equals 1;
            uda_softmax_temp = uda_softmax_temp if uda_softmax_temp > 0 else 1.
            sharp_aug_log_prob = F.log_softmax(logits[sup_size:] / uda_softmax_temp, dim=-1)

        # Calculate the KLdiv loss between ori and aug data
        # Shape: unsup_batch_size
        unsup_loss = torch.sum(unsup_criterion(sharp_aug_log_prob, ori_prob), dim=-1)

        ############
        # Shape: 1 #
        ############
        unsup_loss = torch.sum(unsup_loss * unsup_loss_mask, dim=-1) / torch.max(torch.sum(unsup_loss_mask, dim=-1))
        final_loss = sup_loss + uda_coeff * unsup_loss

        return final_loss, sup_loss, unsup_loss

    return sup_loss, sup_loss, None
