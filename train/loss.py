#!coding:utf-8
import torch
from torch.nn import functional as F

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def kl_div_with_logit(input_logits, target_logits):
    assert input_logits.size()==target_logits.size()
    targets = F.softmax(targets_logits, dim=1)
    return F.kl_div(F.log_softmax(input_logits,1), targets)

def entropy_y_x(logit):
    soft_logit = F.softmax(logit, dim=1)
    return -torch.mean(torch.sum(soft_logit* F.log_softmax(logit,dim=1), dim=1))

def softmax_loss_no_reduce(input_logits, target_logits, eps=1e-10):
    assert input_logits.size()==target_logits.size()
    target_soft = F.softmax(target_logits, dim=1)
    return -torch.sum(target_soft* F.log_softmax(input_logits+eps,dim=1), dim=1)

def softmax_loss_mean(input_logits, target_logits, eps=1e-10):
    assert input_logits.size()==target_logits.size()
    target_soft = F.softmax(target_logits, dim=1)
    return -torch.mean(torch.sum(target_soft* F.log_softmax(input_logits+eps,dim=1), dim=1))

def sym_mse(logit1, logit2):
    assert logit1.size()==logit2.size()
    return torch.mean((logit1 - logit2)**2)

def sym_mse_with_softmax(logit1, logit2):
    assert logit1.size()==logit2.size()
    return torch.mean((F.softmax(logit1,1) - F.softmax(logit2,1))**2)

def mse_with_softmax(logit1, logit2):
    assert logit1.size()==logit2.size()
    return F.mse_loss(F.softmax(logit1,1), F.softmax(logit2,1))

def one_hot(targets, nClass):
    logits = torch.zeros(targets.size(0), nClass).to(targets.device)
    return logits.scatter_(1,targets.unsqueeze(1),1)

def label_smooth(one_hot_labels, epsilon=0.1):
    nClass = labels.size(1)
    return ((1.-epsilon)*one_hot_labels + (epsilon/nClass))

def uniform_prior_loss(logits):
    logit_avg = torch.mean(F.softmax(logits,dim=1), dim=0)
    num_classes, device = logits.size(1), logits.device
    p = torch.ones(num_classes).to(device) / num_classes
    return -torch.sum(torch.log(logit_avg) * p)

def ntxent(a, b, tau=0.05, mask=None):
    a_norm = torch.norm(a, dim=1).reshape(-1, 1)
    a_cap = torch.div(a, a_norm)
    b_norm = torch.norm(b, dim=1).reshape(-1, 1)
    b_cap = torch.div(b, b_norm)
    a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)
    a_cap_b_cap_transpose = torch.t(a_cap_b_cap)
    b_cap_a_cap = torch.cat([b_cap, a_cap], dim=0)
    sim = torch.mm(a_cap_b_cap, a_cap_b_cap_transpose)
    sim_by_tau = torch.div(sim, tau)
    exp_sim_by_tau = torch.exp(sim_by_tau)
    sum_of_rows = torch.sum(exp_sim_by_tau, dim=1)
    exp_sim_by_tau_diag = torch.diag(exp_sim_by_tau)
    numerators = torch.exp(torch.div(torch.nn.CosineSimilarity()(a_cap_b_cap, b_cap_a_cap), tau))
    denominators = sum_of_rows - exp_sim_by_tau_diag
    num_by_den = torch.div(numerators, denominators)
    neglog_num_by_den = -torch.log(num_by_den)
    return torch.mean(neglog_num_by_den)

def ntxentM(a, b, tau=0.05, mask=None):
    a_norm = torch.norm(a, dim=1).reshape(-1, 1)
    a_cap = torch.div(a, a_norm)
    b_norm = torch.norm(b, dim=1).reshape(-1, 1)
    b_cap = torch.div(b, b_norm)
    a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)
    a_cap_b_cap_transpose = torch.t(a_cap_b_cap)
    b_cap_a_cap = torch.cat([b_cap, a_cap], dim=0)
    sim = torch.mm(a_cap_b_cap, a_cap_b_cap_transpose)
    sim_by_tau = torch.div(sim, tau)
    exp_sim_by_tau = torch.exp(sim_by_tau)
    sum_of_rows = torch.sum(exp_sim_by_tau, dim=1)
    exp_sim_by_tau_diag = torch.diag(exp_sim_by_tau)
    numerators = torch.exp(torch.div(torch.nn.CosineSimilarity()(a_cap_b_cap, b_cap_a_cap), tau))
    denominators = sum_of_rows - exp_sim_by_tau_diag
    num_by_den = torch.div(numerators, denominators)
    neglog_num_by_den = -torch.log(num_by_den)
    # print('neglog_num_by_den.shape:{}'.format(neglog_num_by_den.shape))
    mask = torch.cat([mask, mask], dim=0)
    neglog_num_by_den = neglog_num_by_den * mask
    # print('mask.shape:{}'.format(mask.shape))
    return torch.mean(neglog_num_by_den)
