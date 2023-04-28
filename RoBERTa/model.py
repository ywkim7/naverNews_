import torch.nn as nn
from transformers import AutoModel

class naverModel(nn.Module):
    def __init__(self, num_classes, dr_rate=None):
        super(naverModel, self).__init__()
        self.roberta = AutoModel.from_pretrained("klue/roberta-large")
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(1024, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, ids, masks):
        outputs = self.roberta(ids, masks)['pooler_output']
        if self.dr_rate:
            outputs = self.dropout(outputs)
        return self.classifier(outputs)
    


# @misc{park2021klue,
#       title={KLUE: Korean Language Understanding Evaluation},
#       author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jungwoo Ha and Kyunghyun Cho},
#       year={2021},
#       eprint={2105.09680},
#       archivePrefix={arXiv},
#       primaryClass={cs.CL}
# }