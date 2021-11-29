import torch
from torch.nn import functional as F
from torch.nn.modules import loss


class KDLoss(loss._Loss):
    def forward(self, model_output, t_output):

        size_average = True

        model_output_log_prob = F.log_softmax(model_output, dim=1)
        t_output_soft = F.softmax(t_output, dim=1)
        del model_output, t_output

        t_output_soft = t_output_soft.unsqueeze(1)
        model_output_log_prob = model_output_log_prob.unsqueeze(2)

        cross_entropy_loss = -torch.bmm(t_output_soft, model_output_log_prob)
        if size_average:
             cross_entropy_loss = cross_entropy_loss.mean()
        else:
             cross_entropy_loss = cross_entropy_loss.sum()
        return cross_entropy_loss