import torch.nn as nn

class MyBERT(nn.Module):
    def __init__(self, bert):
        super(MyBERT, self).__init__()
        
        self.bert = bert
        self.custom_layer1 = MyBERT.custom_layer(768, 512)
        self.custom_layer2 = MyBERT.custom_layer(512, 64)

        self.classifier = nn.Sequential(
            nn.Linear(64, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
      
        x = self.custom_layer1(cls_hs)
        x = self.custom_layer2(x)
        x = self.classifier(x)

        return x
    
    @staticmethod
    def custom_layer(in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Dropout(0.1),
            nn.Tanh()
        )