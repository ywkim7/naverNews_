import torch.nn as nn
from transformers import ElectraForPreTraining

class naverModel(nn.Module):
    def __init__(self, num_classes, dr_rate=None):
        super(naverModel, self).__init__()
        self.electra = ElectraForPreTraining.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(512, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, ids, masks):
        outputs = self.electra(ids, masks)['logits']
        if self.dr_rate:
            outputs = self.dropout(outputs)
        return self.classifier(outputs)
    


# @misc{park2020koelectra,
# author = {Park, Jangwon},
# title = {KoELECTRA: Pretrained ELECTRA Model for Korean},
# year = {2020},
# publisher = {GitHub},
# journal = {GitHub repository},
# howpublished = {\url{https://github.com/monologg/KoELECTRA}}
# }
