import os

import torch
import torch.jit as jit
from torch.nn import Parameter

'''
class AR(jit.ScriptModule):
    def __init__(self, x_len, nfeatures):
        super(AR, self).__init__()

        self.w = Parameter(torch.randn(1, x_len, nfeatures))
        self.b = Parameter(torch.zeros(nfeatures))

    @jit.script_method
    def forward(self, inputs):
        weighted = torch.sum(inputs * self.w, dim=1) + self.b
        return weighted
'''

# AR_memory
class Model(torch.nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        # config
        self.config = config

        # auto-regressive layer for input & memory
        self.ar_iw = Parameter(torch.randn(1, self.config.x_len, self.config.nfeatures))
        self.ar_ib = Parameter(torch.zeros(self.config.nfeatures))
        self.ar_mw = Parameter(torch.randn(1, self.config.x_len + 1, self.config.nfeatures))
        self.ar_mb = Parameter(torch.zeros(self.config.nfeatures))

        # attention layer
        self.att_i = torch.nn.Linear(self.config.nfeatures, self.config.attention_size, bias=False)
        self.att_m = torch.nn.Linear(self.config.nfeatures, self.config.attention_size, bias=False)
        self.att_b = Parameter(torch.zeros(self.config.attention_size))
        self.att_comb = torch.nn.Linear(self.config.attention_size, 1)

        # fcnn for context & input
        self.pred = torch.nn.Linear(self.config.attention_size + self.config.nfeatures, self.config.nfeatures)
        self.loss_fn = torch.nn.MSELoss()


    def auto_regressive(self, inputs, w, b, ar_lambda):
        # y_t,d = sum_i(w_i * y_i,d) + b_d
        # weighted: [b, nf]
        weighted = torch.sum(inputs * w, dim=1) + b
        # ar_loss = ar_lambda * torch.sum(w ** 2)

        # return weighted, _ar_loss
        return weighted


    def attention(self, inputs, memories):
        # use MLP to compute attention score
        # given input, attend memories(m1, m2, m3, mnstep)

        # inputs : [b, nf] -> [b, 1, nf]
        query = torch.unsqueeze(inputs, dim=1)
        # [b, 1, attention_size]
        query = self.att_i(query)

        # memories : [b*m, nf] -> [b, m, nf]
        key = torch.reshape(memories, (-1, self.config.msteps, self.config.nfeatures))
        # [b, m, attention_size]
        key = self.att_i(key)

        # projection : [b, m, attention_size]
        projection = torch.tanh(query + key + self.att_b)

        # sim_matrix : [b, m, 1]
        sim_matrix = self.att_comb(projection)
        sim_matrix = torch.nn.functional.softmax(sim_matrix, dim=1)

        # context : [b, 1, attention_size] -> [b, attention_size]
        context = torch.matmul(torch.transpose(sim_matrix, 1, 2), key)
        context = torch.squeeze(context, dim=1)

        return context


    def forward (self, input_x, mem, targets):
        ##
        # get auto-regression
        # for input
        # input_ar : [b, nf]
        input_ar = self.auto_regressive(input_x, self.ar_iw , self.ar_ib, self.config.ar_lambda)

        # for memory
        # memory : [b, (n+1)*m, nf] -> [b*m, n+1, nf]
        memories = torch.cat(torch.split(mem, self.config.x_len + 1, dim=1), dim=0)
        # memory_ar : [b*m, nf]
        memory_ar = self.auto_regressive(memories, self.ar_mw, self.ar_mb, self.config.ar_lambda)

        ##
        # get attention
        # context: [b, nf]
        context = self.attention(input_ar, memory_ar)

        ##
        # fcnn
        linear_inputs = torch.cat((input_ar, context), dim=1)  # [b, 2nf]
        self.predictions = self.pred(linear_inputs)
        self.predictions = torch.tanh(self.predictions)

        self.loss = self.loss_fn(self.predictions, targets)

        ##
        # metric
        error = torch.sum((targets - self.predictions)**2) ** 0.5
        denom = torch.sum((targets - torch.mean(targets))**2) ** 0.5
        self.rse = error / denom
        self.mape = torch.mean(torch.abs((targets - self.predictions)/targets))
        self.smape = torch.mean(2*torch.abs(targets-self.predictions)/(torch.abs(targets)+torch.abs(self.predictions)))
        self.mae = torch.mean(torch.abs(targets - self.predictions))

        return self.predictions, self.loss, self.rse, self.smape, self.mae


if __name__ == "__main__":
    from config import Config
    config = Config()
    model = Model(config)
    print("done")