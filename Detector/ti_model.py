import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Model(nn.Module):   
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        self.lamda = args.lamda
    
    def forward(self, input_ids=None, labels=None):
        input_ids = input_ids.view(-1, self.args.block_size)
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        outputs = (outputs * input_ids.ne(1)[:, :, None]).sum(1) / input_ids.ne(1).sum(1)[:, None]
        outputs = outputs.reshape(-1, 2, outputs.size(-1))
        outputs = torch.nn.functional.normalize(outputs, p=2, dim=-1)
        
        # Ensure the output shape is compatible with classifier
        outputs = outputs.view(-1, outputs.size(-1) * 2)
        logits = self.classifier(outputs)
        cos_sim = (outputs[:, 0] * outputs[:, 1]).sum(-1)/(outputs[:, 0].norm()*outputs[:, 1].norm())

        if labels is not None:
            loss_fct = CrossEntropyLoss()  
            classification_loss = loss_fct(logits.view(-1,2), labels.view(-1))  
            
            cos_sim = cos_sim
            loss = classification_loss - self.lamda * cos_sim + self.lamda
            return loss, logits
        else:
            return logits
        
