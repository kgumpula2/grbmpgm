import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import cosine_schedule
import matplotlib.pyplot as plt
import matplotlib
from rbm import RBM

class GRBM(nn.Module):
    """ Gaussian-Bernoulli Restricted Boltzmann Machines (GRBM) """

    def __init__(self,
                 visible_size,
                 hidden_size,
                 CD_step=1,
                 CD_burnin=0,
                 init_var=1e-0,                 
                 inference_method='Gibbs',
                 Langevin_step=10,
                 Langevin_eta=1.0,
                 is_anneal_Langevin=True,
                 Langevin_adjust_step=0,
                 deep_hidden_sizes=None,
                 is_cuda=False) -> None:
        super().__init__()
        # we use samples in [CD_burnin, CD_step) steps
        assert CD_burnin >= 0 and CD_burnin <= CD_step
        assert inference_method in ['Gibbs', 'Langevin', 'Gibbs-Langevin']

        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.CD_step = CD_step
        self.CD_burnin = CD_burnin
        self.init_var = init_var
        self.inference_method = inference_method
        self.Langevin_step = Langevin_step
        self.Langevin_eta = Langevin_eta
        self.is_anneal_Langevin = is_anneal_Langevin
        self.Langevin_adjust_step = Langevin_adjust_step
        self.go_deep = False
        self.is_cuda = is_cuda

        self.W = nn.Parameter(torch.Tensor(visible_size, hidden_size))
        self.b = nn.Parameter(torch.Tensor(hidden_size))
        self.mu = nn.Parameter(torch.Tensor(visible_size))
        self.log_var = nn.Parameter(torch.Tensor(visible_size))
        self.reset_parameters()

        # these are all stacked normal Bernoulli RBM layers
        self.deep_rbms = []
        if deep_hidden_sizes is not None:
            last_size = hidden_size
            for hidden_size in deep_hidden_sizes:
                self.deep_rbms.append(RBM(n_vis=last_size, n_hid=hidden_size))
                if self.is_cuda:
                    self.deep_rbms[-1].cuda()
                last_size = hidden_size
        

    def reset_parameters(self):
        nn.init.normal_(self.W,
                        std=1.0 * self.init_var /
                        np.sqrt(self.visible_size + self.hidden_size))
        nn.init.constant_(self.b, 0.0)
        nn.init.constant_(self.mu, 0.0)
        nn.init.constant_(self.log_var,
                          np.log(self.init_var))  # init variance = 1.0

    def get_var(self):
        return self.log_var.exp().clip(min=1e-8)

    def set_Langevin_eta(self, eta):
        self.Langevin_eta = eta

    def set_Langevin_adjust_step(self, step):
        self.Langevin_adjust_step = step

    @torch.no_grad()
    def energy(self, v, h):
        # compute per-sample energy averaged over batch size
        B = v.shape[0]
        var = self.get_var()
        eng = 0.5 * ((v - self.mu)**2 / var).sum(dim=1)
        eng -= ((v / var).mm(self.W) * h).sum(dim=1) + h.mv(self.b)
        return eng / B

    @torch.no_grad()
    def marginal_energy(self, v):
        # compute per-sample energy averaged over batch size
        B = v.shape[0]
        var = self.get_var()
        eng = 0.5 * ((v - self.mu)**2 / var).sum(dim=1)
        eng -= F.softplus((v / var).mm(self.W) + self.b).sum(dim=1)
        return eng / B

    @torch.no_grad()
    def energy_grad_v(self, v, h):
        # compute the gradient (sample) of energy averaged over batch size
        B = v.shape[0]        
        var = self.get_var()
        return ((v - self.mu) / var - h.mm(self.W.T) / var) / B

    @torch.no_grad()
    def marginal_energy_grad_v(self, v):
        # compute the gradient (sample) of energy averaged over batch size
        B = v.shape[0]
        var = self.get_var()
        return ((v - self.mu) / var - torch.sigmoid((v / var).mm(self.W) + self.b).mm(self.W.T) / var) / B

    @torch.no_grad()
    def energy_grad_param(self, v, h):
        # compute the gradient (parameter) of energy averaged over batch size
        var = self.get_var()
        grad = {}
        grad['W'] = -torch.einsum("bi,bj->ij", v / var, h) / v.shape[0]
        grad['b'] = -h.mean(dim=0)
        grad['mu'] = ((self.mu - v) / var).mean(dim=0)
        grad['log_var'] = (-0.5 * (v - self.mu)**2 / var +
                           ((v / var) * h.mm(self.W.T))).mean(dim=0)
        return grad

    @torch.no_grad()
    def marginal_energy_grad_param(self, v):
        # compute the gradient (parameter) of energy averaged over batch size
        var = self.get_var()
        vv = v / var
        tmp = torch.sigmoid(vv.mm(self.W) + self.b)
        grad = {}
        grad['W'] = -torch.einsum("bi,bj->ij", vv, tmp) / v.shape[0]
        grad['b'] = -tmp.mean(dim=0)
        grad['mu'] = ((self.mu - v) / var).mean(dim=0)
        grad['log_var'] = (-0.5 * (v - self.mu)**2 / var +
                           (vv * tmp.mm(self.W.T))).mean(dim=0)
        return grad

    @torch.no_grad()
    def prob_h_given_v(self, v, var):
        h = torch.sigmoid((v / var).mm(self.W) + self.b)
        if self.go_deep:
            deep_h = self.h_to_deep_h(h)
            h = self.deep_h_to_h(deep_h)
        return h

    @torch.no_grad()
    def prob_v_given_h(self, h):
        return h.mm(self.W.T) + self.mu
    
    def h_to_deep_h(self, h):
        for rbm in self.deep_rbms:
            h = rbm.visible_to_hidden(h)
        return h
    
    def deep_h_to_h(self, h):
        for rbm in reversed(self.deep_rbms):
            h = rbm.hidden_to_visible(h)
        return h

    @torch.no_grad()
    def log_metropolis_ratio_Gibbs_Langevin(self, v_old, h_old, v_new, h_new, eta_list):
        """ Metropolis-Hasting ratio of accepting the move from old to new state """
        B = v_old.shape[0]
        var = self.get_var()
        eng_diff = -self.energy(v_new, h_new) + self.energy(v_old, h_old)
        state_h_new = (v_new / var).mm(self.W) + self.b
        state_h_old = (v_old / var).mm(self.W) + self.b
        log_prob_h_given_v_new = - \
            F.binary_cross_entropy_with_logits(
                state_h_old, h_old, reduction='none').sum(dim=1)
        log_prob_h_given_v_old = - \
            F.binary_cross_entropy_with_logits(
                state_h_new, h_new, reduction='none').sum(dim=1)

        eta = torch.tensor(eta_list).to(var.device)  # shape K X 1
        beta_in = 1.0 - eta.unsqueeze(1) / (B * var.unsqueeze(0))  # shape K X D
        beta = torch.flip(torch.cumprod(
            torch.flip(beta_in, [0]), 0), [0])  # shape K X D
        beta = F.pad(beta, [0, 0, 0, 1], "constant", 1.0)  # shape (K+1) X D        
        va = (beta[1:] * eta.view(-1, 1)).sum(dim=0) / (B * var)  # shape 1 X D
        tilde_sigma_sqrt = (
            (beta[1:]**2 * eta.view(-1, 1)).sum(dim=0)).sqrt()  # shape 1 X D
        proposal_eng_new = - torch.pow((v_old - beta[0] * v_new - va * (
            self.mu + h_new.mm(self.W.T))) / (2 * tilde_sigma_sqrt), 2.0).sum(dim=1)
        proposal_eng_old = - torch.pow((v_new - beta[0] * v_old - va * (
            self.mu + h_old.mm(self.W.T))) / (2 * tilde_sigma_sqrt), 2.0).sum(dim=1)
        
        return eng_diff + proposal_eng_new - proposal_eng_old + log_prob_h_given_v_new - log_prob_h_given_v_old

    @torch.no_grad()
    def log_metropolis_ratio_Langevin_one_step(self, v_old, v_new, grad_old, eta):
        """ Metropolis-Hasting ratio of accepting the move from old to new state """
        eng_diff = -self.marginal_energy(v_new) + self.marginal_energy(v_old)
        proposal_eng_new = - \
            torch.pow(v_old - v_new + eta *
                      self.marginal_energy_grad_v(v_new), 2.0).sum(dim=1) / (4 * eta)
        proposal_eng_old = - \
            torch.pow(v_new - v_old + eta * grad_old,
                      2.0).sum(dim=1) / (4 * eta)

        return eng_diff + proposal_eng_new - proposal_eng_old

    @torch.no_grad()
    def Gibbs_sampling_vh(self, v, num_steps=10, burn_in=0, mask=None, v_true=None):
        samples, var = [], self.get_var()
        std = var.sqrt()
        h = torch.bernoulli(self.prob_h_given_v(v, var))
        for ii in range(num_steps):
            # backward sampling
            mu = self.prob_v_given_h(h)
            v = mu + torch.randn_like(mu) * std

            if mask is not None and v_true is not None:
                v = torch.where(mask == 0, v_true, v)
            
            # forward sampling
            h = torch.bernoulli(self.prob_h_given_v(v, var))

            if ii >= burn_in:
                samples += [(v, h)]

        return samples

    @torch.no_grad()
    def Langevin_sampling_v(self,
                            v,
                            num_steps=10,
                            eta=1.0e+0,
                            burn_in=0,
                            is_anneal=True,
                            adjust_step=0,
                            mask=None,
                            v_true=None):
        eta_list = cosine_schedule(eta_max=eta, T=num_steps)
        samples = []

        for ii in range(num_steps):
            eta_ii = eta_list[ii] if is_anneal else eta
            grad_v = self.marginal_energy_grad_v(v)

            v_new = v - eta_ii * grad_v + \
                torch.randn_like(v) * np.sqrt(eta_ii * 2)
            
            if mask is not None and v_true is not None:
                v_new = torch.where(mask == 0, v_true, v_new)

            if ii >= adjust_step:
                tmp_u = torch.rand(v.shape[0]).to(v.device)
                log_ratio = self.log_metropolis_ratio_Langevin_one_step(
                    v, v_new, grad_v, eta_ii)
                ratio = torch.minimum(
                    torch.ones_like(log_ratio), log_ratio.exp())
                v = v_new * (tmp_u < ratio).float().view(
                    -1, 1) + v * (tmp_u >= ratio).float().view(-1, 1)
            else:
                v = v_new

            if ii >= burn_in:
                samples += [v]

        return samples

    @torch.no_grad()
    def Gibbs_Langevin_sampling_vh(self,
                                   v,
                                   num_steps=10,
                                   num_steps_Langevin=10,
                                   eta=1.0e+0,
                                   burn_in=0,
                                   is_anneal=True,
                                   adjust_step=0,
                                   mask=None,
                                   v_true=None):
        samples, var = [], self.get_var()
        eta_list = cosine_schedule(eta_max=eta, T=num_steps_Langevin)

        h = torch.bernoulli(self.prob_h_given_v(v, var))

        for ii in range(num_steps):
            v_old, h_old = v, h
            # backward sampling
            for jj in range(num_steps_Langevin):
                eta_jj = eta_list[jj] if is_anneal else eta
                grad_v = self.energy_grad_v(v, h)
                v = v - eta_jj * grad_v + \
                    torch.randn_like(v) * np.sqrt(eta_jj * 2)
                
            if mask is not None and v_true is not None:
                v = torch.where(mask == 0, v_true, v)

            # forward sampling
            h = torch.bernoulli(self.prob_h_given_v(v, var))

            if ii >= adjust_step:
                tmp_u = torch.rand(v.shape[0]).to(v.device)
                log_ratio = self.log_metropolis_ratio_Gibbs_Langevin(
                    v_old, h_old, v, h, eta_list)
                ratio = torch.minimum(
                    torch.ones_like(log_ratio), log_ratio.exp())                
                v = v * (tmp_u < ratio).float().view(
                    -1, 1) + v_old * (tmp_u >= ratio).float().view(-1, 1)
                h = h * (tmp_u < ratio).float().view(
                    -1, 1) + h_old * (tmp_u >= ratio).float().view(-1, 1)

            if ii >= burn_in:
                samples += [(v, h)]

        return samples

    @torch.no_grad()
    def reconstruction(self, v):
        v, var = v.view(v.shape[0], -1), self.get_var()
        prob_h = self.prob_h_given_v(v, var)
        v_bar = self.prob_v_given_h(prob_h)
        return F.mse_loss(v, v_bar)

    @torch.no_grad()
    def sampling(self, v_init, num_steps=1, save_gap=1, mask=None, v_true=None):
        v_shape = v_init.shape
        v = v_init.view(v_shape[0], -1)
        var = self.get_var()
        var_mean = var.mean().item()
        mask = mask.view(v_shape[0], -1) if mask is not None else None
        v_true = v_true.view(v_shape[0], -1) if v_true is not None else None

        if self.inference_method == 'Gibbs':
            samples = self.Gibbs_sampling_vh(v, num_steps=num_steps - 1, mask=mask, v_true=v_true)
            samples = [xx[0] for xx in samples]  # extract v
        elif self.inference_method == 'Langevin':
            samples = self.Langevin_sampling_v(v,
                                               num_steps=num_steps - 1,
                                               eta=self.Langevin_eta * var_mean,
                                               is_anneal=self.is_anneal_Langevin,
                                               adjust_step=self.Langevin_adjust_step,
                                               mask=mask,
                                               v_true=v_true)
        elif self.inference_method == 'Gibbs-Langevin':
            samples = self.Gibbs_Langevin_sampling_vh(
                v,
                num_steps=num_steps - 1,
                num_steps_Langevin=self.Langevin_step,
                eta=self.Langevin_eta * var_mean,
                is_anneal=self.is_anneal_Langevin,
                adjust_step=self.Langevin_adjust_step,
                mask=mask,
                v_true=v_true)
            samples = [xx[0] for xx in samples]  # extract v

        # use conditional mean as the last sample
        h = torch.bernoulli(self.prob_h_given_v(samples[-1], var))
        mu = self.prob_v_given_h(h)
        v_list = [(0, v_init)] + [(ii + 1, samples[ii].view(v_shape).detach())
                                  for ii in range(num_steps - 1)
                                  if (ii + 1) % save_gap == 0
                                  ] + [(num_steps, mu.view(v_shape).detach())]
        
        for i in range(len(v_list)):
            ind, v = v_list[i]
            if mask is not None and v_true is not None:
                v_list[i] = (ind, torch.where(mask.view(v_shape) == 0, v_true.view(v_shape), v))

        return v_list

    @torch.no_grad()
    def positive_grad(self, v):
        h = torch.bernoulli(self.prob_h_given_v(v, self.get_var()))
        grad = self.energy_grad_param(v, h)
        return grad

    @torch.no_grad()
    def negative_grad(self, v):
        var = self.get_var()
        var_mean = var.mean().item()
        if self.inference_method == 'Gibbs':
            samples = self.Gibbs_sampling_vh(v,
                                             num_steps=self.CD_step,
                                             burn_in=self.CD_burnin)
            v_neg = torch.cat([xx[0] for xx in samples], dim=0)
            h_neg = torch.cat([xx[1] for xx in samples], dim=0)
            grad = self.energy_grad_param(v_neg, h_neg)
        elif self.inference_method == 'Langevin':
            samples = self.Langevin_sampling_v(v,
                                               num_steps=self.CD_step,
                                               burn_in=self.CD_burnin,
                                               eta=self.Langevin_eta * var_mean,
                                               is_anneal=self.is_anneal_Langevin,
                                               adjust_step=self.Langevin_adjust_step)
            v_neg = torch.cat(samples, dim=0)
            grad = self.marginal_energy_grad_param(v_neg)

        elif self.inference_method == 'Gibbs-Langevin':
            samples = self.Gibbs_Langevin_sampling_vh(
                v,
                num_steps=self.CD_step,
                burn_in=self.CD_burnin,
                num_steps_Langevin=self.Langevin_step,
                eta=self.Langevin_eta * var_mean,
                is_anneal=self.is_anneal_Langevin,
                adjust_step=self.Langevin_adjust_step)
            v_neg = torch.cat([xx[0] for xx in samples], dim=0)
            h_neg = torch.cat([xx[1] for xx in samples], dim=0)
            grad = self.energy_grad_param(v_neg, h_neg)

        return grad

    @torch.no_grad()
    def CD_grad(self, v):
        v = v.view(v.shape[0], -1)

        # postive gradient
        grad_pos = self.positive_grad(v)

        # negative gradient
        v_neg = torch.randn_like(v)
        grad_neg = self.negative_grad(v_neg)

        # compute update
        for name, param in self.named_parameters():
            param.grad = grad_pos[name] - grad_neg[name]
