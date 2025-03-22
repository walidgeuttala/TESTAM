import torch.optim as optim
from model2 import *
import util

class trainer():
    def __init__(self, scaler, in_dim, out_dim, num_nodes, nhid, dropout, device, 
                 lr_mul = 1., n_warmup_steps = 2000, quantile = 0.7, is_quantile = False, warmup_epoch = 0,
                 use_uncertainty = False):
        self.model = TESTAM(num_nodes, dropout, in_dim=in_dim, out_dim=out_dim, hidden_size=nhid)
        self.model.to(device)
        # The learning rate setting below will not affct initial learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr = 1e-3, betas = (0.9, 0.98), eps = 1e-9)
        self.schedule = util.CosineWarmupScheduler(self.optimizer, d_model = nhid, n_warmup_steps = n_warmup_steps, lr_mul = lr_mul)
        self.loss = util.masked_mae
        
        self.scaler = scaler
        self.clip = 5
        self.n_warmup_steps = n_warmup_steps
        self.flag = is_quantile
        self.quantile = quantile
        self.cur_epoch = 0
        self.warmup_epoch = warmup_epoch
        self.threshold = 0.
        self.use_uncertainty = use_uncertainty


    def check_for_nan(self, tensor, var_name):
        if torch.isnan(tensor).any():
            raise ValueError(f"{var_name} contains NaN")

    def train(self, input, real, cur_epoch):
        self.model.train()
        self.schedule.zero_grad()

        output, gate, res = self.model(input)
        self.check_for_nan(output, "Output")
        self.check_for_nan(gate, "Gate")
        self.check_for_nan(res, "Residual")

        predict = self.scaler.inverse_transform(output)
        self.check_for_nan(predict, "Predict")

        ind_loss = self.loss(self.scaler.inverse_transform(res), real.permute(0,2,3,1).unsqueeze(-1), self.threshold, reduce=None)
        self.check_for_nan(ind_loss, "ind_loss")
        self.check_for_nan(real, "real")

        if self.flag:
            gated_loss = self.loss(predict, real, reduce=None).permute(0,2,3,1)
            self.check_for_nan(gated_loss, "gated_loss")
            l_worst_avoidance, l_best_choice = self.get_quantile_label(gated_loss, gate, real)
            self.check_for_nan(l_worst_avoidance, "l_worst_avoidance flag") 

        else:
            l_worst_avoidance, l_best_choice = self.get_label(ind_loss, gate, real)
            self.check_for_nan(l_worst_avoidance, "l_worst_avoidance else") 

        if self.use_uncertainty:
            uncertainty = self.get_uncertainty(real.permute(0,2,3,1), threshold=self.threshold)
            uncertainty = uncertainty.unsqueeze(dim=-1)
        else:
            uncertainty = torch.ones_like(gate)
        self.check_for_nan(uncertainty, "uncertainty") 

        epsilon = 1e-4
        self.check_for_nan(l_worst_avoidance, "l_worst_avoidance") 

        worst_avoidance = -.5 * l_worst_avoidance * torch.log(gate + epsilon) * (2 - uncertainty)
        best_choice = -.5 * l_best_choice * torch.log(gate + epsilon) * uncertainty
        self.check_for_nan(worst_avoidance, "worst_avoidance") 
        self.check_for_nan(best_choice, "best_choice") 

        if cur_epoch <= self.warmup_epoch:
            loss = ind_loss.mean()
        else:
            loss = ind_loss.mean() + worst_avoidance.mean() + best_choice.mean()

        self.check_for_nan(loss, "Loss")  # Check the final loss for NaN

        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.schedule.step_and_update_lr()
        mape = util.masked_mape(predict, real, self.threshold).item()
        rmse = util.masked_rmse(predict, real, self.threshold).item()

        return loss.item(), mape, rmse

    def eval(self, input, real):
        self.model.eval()
        output = self.model(input)
        #output = [batch_size,12,num_nodes,out_dim]
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, self.threshold)
        mape = util.masked_mape(predict,real,self.threshold).item()
        rmse = util.masked_rmse(predict,real,self.threshold).item()
        return loss.item(),mape,rmse

    def lb_loss(self, gate):
        n_experts = gate.size(-1)
        _, indices = torch.max(gate, dim = -1)
        counts = gate.new_tensor([len(torch.eq(indices, i).nonzero(as_tuple=True)[0]) for i in range(n_experts)])
        proxied_lb = (counts / counts.sum()) * gate.mean(dim = (1,2))
        lb_loss = proxied_lb.mean()
        return lb_loss

    def get_uncertainty(self, x, mode = 'psd', threshold = 0.0):

        def _acorr(x, dim = -1):
            size = x.size(dim)
            x_fft = torch.fft.fft(x, dim = dim)
            acorr = torch.fft.ifft(x_fft * x_fft.conj(), dim = dim).real
            return acorr / (size ** 2)

        def nanstd(x, dim, keepdim = False):
            return torch.sqrt(
                        torch.nanmean(
                                torch.pow(torch.abs(x - torch.nanmean(x, dim = dim, keepdim = True)), 2),
                                dim = dim, keepdim = keepdim
                            )
                    )

        with torch.no_grad():
            if mode == 'acorr':
                std = x.std(dim = -2, keepdim = True)
                corr = _acorr(x, dim = -2)
                x_noise = x + std * torch.randn((1,1,x.size(-2),1), device = x.device) / 2
                corr_w_noise = _acorr(x_noise, dim = -2)
                corr_changed = torch.abs(corr - corr_w_noise)
                uncertainty = torch.ones_like(corr_changed) * (corr_changed > corr_changed.quantile(1 - self.quantile))
            elif mode == 'psd':
                from copy import deepcopy as cp
                vals = cp(x)
                vals[vals <= threshold] = torch.nan
                diff = vals[:,:,1:] - vals[:,:,:-1]
                corr_changed = torch.nanmean(torch.abs(diff), dim = -2, keepdim = True) / (nanstd(diff, dim = -2, keepdim = True) + 1e-6)
                corr_changed[corr_changed != corr_changed] = 0.
                uncertainty = torch.ones_like(corr_changed) * (corr_changed < corr_changed.quantile(self.quantile))
            else:
                raise NotImplementedError
            return uncertainty


    def get_quantile_label(self, gated_loss, gate, real):
        gated_loss = gated_loss.unsqueeze(dim = -1)
        real = real.unsqueeze(dim = -1)
        max_quantile = gated_loss.quantile(self.quantile)
        min_quantile = gated_loss.quantile(1 - self.quantile)
        incorrect = (gated_loss > max_quantile).expand_as(gate)
        correct = ((gated_loss < min_quantile) & (real.permute(0,2,3,1,4) > self.threshold)).expand_as(gate)
        cur_expert = gate.argmax(dim = -1, keepdim = True)
        not_chosen = gate.topk(dim = -1, k = 2, largest = False).indices
        selected = torch.zeros_like(gate).scatter_(-1, cur_expert, 1.0)
        scaling = torch.zeros_like(gate).scatter_(-1, not_chosen, 0.5)
        selected[incorrect] = scaling[incorrect]
        l_worst_avoidance = selected.detach()
        selected = torch.zeros_like(gate).scatter(-1, cur_expert, 1.0) * correct
        l_best_choice = selected.detach()
        return l_worst_avoidance, l_best_choice

    def get_label(self, ind_loss, gate, real):
        self.check_for_nan(ind_loss, "ind_loss")  # Check the final loss for NaN
        self.check_for_nan(gate, "gate")
        self.check_for_nan(real, "real")

        n_experts = gate.size(-1)
        empty_val = (real.permute(0,2,3,1).unsqueeze(-1).expand_as(gate)) <= self.threshold
        self.check_for_nan(empty_val, "empty_val")
        max_error = ind_loss.argmax(dim = -1, keepdim = True)
        self.check_for_nan(max_error, "max_error")
        cur_expert = gate.argmax(dim = -1, keepdim = True)
        self.check_for_nan(cur_expert, "cur_expert")
        incorrect = max_error == cur_expert
        self.check_for_nan(incorrect, "incorrect")
        selected = torch.zeros_like(gate).scatter(-1, cur_expert, 1.0)
        self.check_for_nan(selected, "selected")
        scaling = torch.ones_like(gate) * ind_loss
        self.check_for_nan(scaling, "scaling")
        scaling = scaling.scatter(-1, max_error, 0.)
        self.check_for_nan(scaling, "scaling")
        # # Check if 'scaling' contains NaN values
        # assert not torch.isnan(scaling).any(), "scaling contains NaN values!"

        # # Check if 'selected' contains NaN values
        # assert not torch.isnan(selected).any(), "selected contains NaN values!"

        # # Check if the sum along the specified dimension is zero (which could cause division by zero)
        # scaling_sum = scaling.sum(dim=-1, keepdim=True)
        # assert (scaling_sum != 0).all(), f"scaling.sum(dim=-1) is zero, leading to division by zero: {scaling_sum}"

        # # Check if 'scaling' contains all zeros along a dimension (which would make the sum zero)
        # assert (scaling.sum(dim=-1, keepdim=True) != 0).all(), "scaling contains all zeros along a dimension, causing division by zero!"

        # # Check if the result of (1 - selected) has NaN values
        # result_1_minus_selected = 1 - selected
        # assert not torch.isnan(result_1_minus_selected).any(), "1 - selected contains NaN values!"

        # # Perform the original operation with a small epsilon to avoid division by zero
        # epsilon = 1e-8
        # scaling_normalized = scaling / (scaling_sum + epsilon) * (1 - selected)

        # # Check if the result contains NaN values after the operation
        # assert not torch.isnan(scaling_normalized).any(), "The result of the operation contains NaN values!"

        # print("All checks passed successfully!")
        # scaling = scaling / (scaling.sum(dim=-1, keepdim=True)) * (1 - selected)
        epsilon = 1e-8
        scaling = scaling / (scaling.sum(dim=-1, keepdim=True) + epsilon) * (1 - selected)


        self.check_for_nan(scaling, "scaling")
        l_worst_avoidance = torch.where(incorrect, scaling, selected)
        self.check_for_nan(l_worst_avoidance, "l_worst_avoidance")
        l_worst_avoidance = torch.where(empty_val, torch.zeros_like(gate), l_worst_avoidance)
        self.check_for_nan(l_worst_avoidance, "l_worst_avoidance")
        l_worst_avoidance = l_worst_avoidance.detach()
        min_error = ind_loss.argmin(dim = -1, keepdim = True)
        correct = min_error == cur_expert
        scaling = torch.zeros_like(gate)
        scaling = scaling.scatter(-1, min_error, 1.)
        l_best_choice = torch.where(correct, selected, scaling)
        l_best_choice = torch.where(empty_val, torch.zeros_like(gate), l_best_choice)
        l_best_choice = l_best_choice.detach()
        return l_worst_avoidance, l_best_choice
