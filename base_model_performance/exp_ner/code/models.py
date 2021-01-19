import torch
import torch.nn as nn
import torch.nn.functional as F


def create_base_model(embedding_vector_size, hidden_size, dense1_out_size, dense_layer1_activation, num_labels):
    feature_extractor = FeatureExtractor(embedding_vector_size, hidden_size)
    base_model = BaseModel(feature_extractor, hidden_size, dense1_out_size, dense_layer1_activation, num_labels)

    return base_model, feature_extractor


class BaseModel(nn.Module):
    def __init__(self, Feature_Extractor, hidden_size, dense1_out_size, dense_layer1_activation, num_labels):
        super(BaseModel, self).__init__()
        self.Feature_Extractor = Feature_Extractor
        self.dense_layer1 = nn.Linear(hidden_size*2, dense1_out_size)
        if dense_layer1_activation == "relu":
            self.dense_layer1_activation = nn.ReLU()
        else:
            self.dense_layer1_activation = nn.Sigmoid()

        self.dense_layer2 = nn.Linear(dense1_out_size, num_labels)

    def forward(self, x):
        feat = self.Feature_Extractor(x)
        out = self.dense_layer1(feat)
        out = self.dense_layer1_activation(out)
        out = self.dense_layer2(out)
        return out


class FeatureExtractor(nn.Module):
    # A wrapper of LSTM for extracting text features
    def __init__(self, embedding_vector_size, hidden_size):
        super(FeatureExtractor, self).__init__()
        self.LSTM = nn.LSTM(embedding_vector_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, x):
        _, (hn, _) = self.LSTM(x, None)
        feat = torch.cat((hn[0, ::], hn[1, ::]), dim=1)
        return feat


class Global_CM(nn.Module):
    def __init__(self, base_model, channel_weights, fix_transition, device):
        super(Global_CM, self).__init__()
        self.eps = 1e-7
        self.base_model = base_model
        assert fix_transition
        self.transition_mat = torch.tensor(channel_weights, requires_grad=False).float().to(device)

    def forward(self, x):
        out = self.base_model(x)
        out = F.softmax(out, dim=1)
        out = torch.matmul(out, self.transition_mat)
        out = torch.log(out+self.eps)
        return out

