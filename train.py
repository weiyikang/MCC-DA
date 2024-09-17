import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import copy
import wandb
from lib.utils.federated_utils import *
from lib.utils.avgmeter import AverageMeter
from train.utils import *
from train.loss import *
from train.context import disable_tracking_bn_stats
from train.ramps import exp_rampup

# pseudo-label, max confidence, 训练mnist-m数据集
def train_uda(train_dloader_list, test_dloader_list, model_list, classifier_list, optimizer_list, classifier_optimizer_list, epoch, writer,
        num_classes, domain_weight, source_domains, batchnorm_mmd, batch_per_epoch, confidence_gate_begin,
        confidence_gate_end, communication_rounds, total_epochs, malicious_domain, attack_level, args=None, pre_models=None, pre_classifiers=None):
    task_criterion = nn.CrossEntropyLoss().cuda()
    cos = nn.CosineSimilarity(dim=1).cuda()
    tau_define = args.tau
    source_domain_num = len(train_dloader_list[1:])
    for model in model_list:
        model.train()
    for classifier in classifier_list:
        classifier.train()
    # If communication rounds <1,
    # then we perform parameter aggregation after (1/communication_rounds) epochs
    # If communication rounds >=1:
    # then we extend the training epochs and use fewer samples in each epoch.
    if communication_rounds in [0.2, 0.5]:
        model_aggregation_frequency = round(1 / communication_rounds)
    else:
        model_aggregation_frequency = 1
    
    # train local source domain models
    for f in range(model_aggregation_frequency):
        current_domain_index = 0
        # Train model locally on source domains
        for train_dloader, model, classifier, optimizer, classifier_optimizer in zip(train_dloader_list[1:],
                                                                                     model_list[1:],
                                                                                     classifier_list[1:],
                                                                                     optimizer_list[1:],
                                                                                     classifier_optimizer_list[1:]):
            
            # check if the source domain is the malicious domain with poisoning attack
            source_domain = source_domains[current_domain_index]
            current_domain_index += 1
            if source_domain == malicious_domain and attack_level > 0:
                poisoning_attack = True
            else:
                poisoning_attack = False
            # save source domain models, before local training
            global_model = copy.deepcopy(model)
            global_classifier = copy.deepcopy(classifier)
            for i, (image_s, label_s) in enumerate(train_dloader):
                if i >= batch_per_epoch:
                    break
                image_s_w = image_s[0].cuda()
                image_s_s = image_s[1].cuda()
                label_s = label_s.long().cuda()
                true_label = label_s
                if poisoning_attack:
                    # perform poison attack on source domain
                    corrupted_num = round(label_s.size(0) * attack_level)
                    # provide fake labels for those corrupted data
                    label_s[:corrupted_num, ...] = (label_s[:corrupted_num, ...] + 1) % num_classes
                # reset grad
                optimizer.zero_grad()
                classifier_optimizer.zero_grad()
                # each source domain do optimize

                feature_s, _ = model(image_s_w)
                if args.pj == 1:
                    _, feature_cur = model(image_s_s)
                else:
                    feature_cur, _ = model(image_s_s)
                output_s = classifier(feature_s)
                # task_loss_s = task_criterion(output_s, label_s)
                
                # label smoth
                log_probs = torch.log_softmax(output_s, dim=1)
                label_s = torch.zeros(log_probs.size()).scatter_(1, label_s.unsqueeze(1).cpu(), 1)
                label_s = label_s.cuda()
                alpha = 0.1
                label_s = (1-alpha) * label_s + alpha / num_classes
                task_loss_s = (- label_s * log_probs).mean(0).sum()

                # scic: source domain conditional instance contrastive loss
                # positive pair features: extracted by the local target model;
                lambda_sic = args.sic
                feature_pos = []
                condition_mask = []
                # extract positive pair from local target domain model (pj: additional project head)
                with torch.no_grad():
                    if pre_models is not None:
                        if args.pj == 1:
                            feature, feature_pos = pre_models[0](image_s_w)
                        else:
                            feature_pos, _ = pre_models[0](image_s_w)
                            feature = feature_pos
                        logits = pre_classifiers[0](feature)
                        probs = torch.softmax(logits, dim=1)
                    else:
                        if args.pj == 1:
                            feature, feature_pos = global_model(image_s_w)
                        else:
                            feature_pos, _ = global_model(image_s_w)
                            feature = feature_pos
                        logits = global_classifier(feature)
                        probs = torch.softmax(logits, dim=1)
                    _, idx = probs.max(1)
                    condition_mask = (idx == true_label)*1.0 # (B)

                tau = tau_define
                loss_scic = ntxentM(feature_pos, feature_cur, tau=tau, mask=condition_mask)
                
                loss = task_loss_s + lambda_sic * loss_scic
                loss.backward()
                optimizer.step()
                classifier_optimizer.step()

    # epochs of local target domain
    epochs_target = 1
    # weights of model aggregation
    target_weight = [0, 0]
    consensus_focus_dict = {}
    for f in range(epochs_target):
        # train local target domain model by pseudo-labeling, such as knowledge vote
        confidence_gate = (confidence_gate_end - confidence_gate_begin) * (epoch / total_epochs) + confidence_gate_begin
        
        for i in range(1, len(train_dloader_list)):
            consensus_focus_dict[i] = 0
        for i, (image_t, label_t) in enumerate(train_dloader_list[0]):
            if i >= batch_per_epoch:
                break
            optimizer_list[0].zero_grad()
            classifier_optimizer_list[0].zero_grad()

            image_w = image_t[0].cuda()
            image_s = image_t[1].cuda()

            project_scr = []
            src_instance_weights = []
            
            with torch.no_grad():
                # pseudo-labeling strategy 1: Max prediction
                pred_u = []
                entropy_u = []
                for i in range(1, len(classifier_list)):
                    feature_t, feature_t_pj = model_list[i](image_w)
                    if args.pj == 1:
                        project_s = feature_t_pj
                    else:
                        project_s = feature_t
                    pred_uk = classifier_list[i](feature_t)
                    entropy_uk = Entropy(torch.softmax(pred_uk, dim=1))
                    pred_uk = pred_uk.unsqueeze(1)
                    entropy_uk = entropy_uk.view(entropy_uk.size()[0], -1)
                    pred_u.append(pred_uk)
                    entropy_u.append(entropy_uk)
                    project_s = project_s.unsqueeze(1)
                    project_scr.append(project_s)
                    
                pred_u = torch.cat(pred_u, 1) # (B, K, C)
                entropy_u = torch.cat(entropy_u, 1) # (B, K)
                project_scr = torch.cat(project_scr, 1) # (B, K, C)
                experts_max_p, experts_max_idx = pred_u.max(2) # (B, K)
                max_expert_p, max_expert_idx = experts_max_p.max(1) # (B)
                pseudo_label_u = []
                for i, experts_label in zip(max_expert_idx, experts_max_idx):
                    pseudo_label_u.append(experts_label[i])
                pseudo_label_u = torch.stack(pseudo_label_u, 0)
                max_pred = create_onehot(pseudo_label_u, num_classes)
                max_pred = max_pred.cuda()
                max_pred_mask = (max_expert_p >= confidence_gate).float()
                
                # pseudo-labeling strategy 2: Average prediction
                mean_pred = pred_u.mean(1)
                temperature = args.temperature
                mean_pred = torch.softmax(mean_pred / temperature, dim=1)
                max_mean_p, _ = mean_pred.max(1)
                mean_pred_mask = (max_mean_p >= confidence_gate).float()
                
                # feature fusion weights: entropy weighted
                weights = torch.softmax(1.0 / (entropy_u + 1e-6), dim=1) # (B, C)
                src_instance_weights = weights
                src_instance_weights = src_instance_weights.unsqueeze(2)
                src_instance_weights = src_instance_weights.expand(-1, -1, project_scr.size(2))
                
                # pseudo-labeling strategy 4: Entropy weighted average prediction
                weights = weights.unsqueeze(2)
                weights = weights.expand(-1, -1, num_classes) # (128, 4, 10)
                weighted_mean_pred = (weights * pred_u).sum(1)
                temperature = args.temperature
                weighted_mean_pred = torch.softmax(weighted_mean_pred / temperature, dim=1)
                max_weighted_mean_pred, _ = weighted_mean_pred.max(1)
                weighted_mean_pred_mask = (max_weighted_mean_pred >= confidence_gate).float()

            # pseudo-labeling strategy 3: knowledge vote
            with torch.no_grad():
                knowledge_list = [torch.softmax(classifier_list[i](model_list[i](image_w)[0]), dim=1).unsqueeze(1) for
                                i in range(0, len(classifier_list))]
                knowledge_list = torch.cat(knowledge_list, 1)
            _, kv_pred, kv_mask = knowledge_vote(knowledge_list, confidence_gate,
                                                                num_classes=num_classes)
            target_weight[0] += torch.sum(kv_mask).item()
            target_weight[1] += kv_mask.size(0)

            # choose pseudo-labeling strategies
            pseudo_label = []
            label_mask = []

            if args.pl == 1:
                # Max predictioin
                pseudo_label = max_pred
                label_mask = max_pred_mask
            elif args.pl == 2:
                # mean prediction
                pseudo_label = mean_pred
                label_mask = mean_pred_mask
            elif args.pl == 3:
                # knowledge vote
                pseudo_label = kv_pred
                label_mask = kv_mask
            else:
                pseudo_label = weighted_mean_pred
                label_mask = weighted_mean_pred_mask

            # # Mixup数据增广
            # lam = np.random.beta(2, 2)
            # batch_size = image_w.size(0)
            # index = torch.randperm(batch_size).cuda()
            # mixed_image = lam * image_w + (1 - lam) * image_w[index, :]
            # mixed_label = lam * pseudo_label + (1 - lam) * pseudo_label[index, :]
            # feature_t, _ = model_list[0](mixed_image)
            # output_t_cls = classifier_list[0](feature_t)
            # output_t = torch.log_softmax(output_t_cls, dim=1)
            # l_u = (-mixed_label * output_t).sum(1)
            # task_loss_t = (l_u * label_mask).mean()

            # task loss
            feature_t, _ = model_list[0](image_w)
            output_t_cls = classifier_list[0](feature_t)
            output_t = torch.log_softmax(output_t_cls, dim=1)
            l_u = (-pseudo_label * output_t).sum(1)
            task_loss_t = (l_u * label_mask).mean()
            
            # fused feature extracted from multiple local source domain models
            weighted_mean_projector = (src_instance_weights * project_scr).sum(1)
            
            if args.pj == 1:
                _, project_f2 = model_list[0](image_s)
            else:
                project_f2, _ = model_list[0](image_s)

            tau = tau_define
            loss_tic = ntxent(weighted_mean_projector, project_f2, tau=tau, mask=None)
            lambda_tic = args.tic
            loss = task_loss_t + lambda_tic * loss_tic
            loss.backward()
            optimizer_list[0].step()
            classifier_optimizer_list[0].step()
            consensus_focus_dict = calculate_consensus_focus(consensus_focus_dict, knowledge_list, confidence_gate,
                                                        source_domain_num, num_classes)
    
    # save the local models, before model aggregation
    pre_models = []
    pre_classifiers = []
    for i in range(0, len(model_list)):
        pre_models.append(copy.deepcopy(model_list[i]))
        pre_classifiers.append(copy.deepcopy(classifier_list[i]))

    # test the accuracy of local target domain model
    target_domain = '******target******'
    if args is not None:
        target_domain = args.target_domain
    acc = test(target_domain, source_domains, test_dloader_list, model_list, classifier_list, epoch, args, num_classes, states='local')
    
    # weights of model aggregation, avergae strategy
    domain_weight = []
    num_domains = len(model_list)
    for i in range(num_domains):
        domain_weight.append(1.0/num_domains)
    
    # model aggregation
    federated_avg(model_list, domain_weight, mode='fedavg')
    federated_avg(classifier_list, domain_weight, mode='fedavg')
    
    return acc, pre_models, pre_classifiers
