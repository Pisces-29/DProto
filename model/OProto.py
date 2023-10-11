import torch
import torch.nn.functional as F
from torch import nn
from toolkit.framework import FewShotREModel


def l2norm(X):
    norm = torch.pow(X, 2).sum(dim=-1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


class OProto(FewShotREModel):
    def __init__(self, sentence_encoder, dot=False, theta=0.6):
        FewShotREModel.__init__(self, sentence_encoder)
        self.drop = nn.Dropout()
        self.dot = dot
        self.theta = theta
        self.M_1 = torch.tensor(0.4, dtype=torch.float32)
        self.M_2 = torch.tensor(0.8, dtype=torch.float32)
        self.alpha = torch.tensor(10., dtype=torch.float32)
        self.beta = torch.tensor(1.0, dtype=torch.float32)
        self.gamma = torch.tensor(1.0, dtype=torch.float32)
        if torch.cuda.is_available():
            self.M_1 = self.M_1.cuda()
            self.M_2 = self.M_2.cuda()
            self.alpha = self.alpha.cuda()
            self.beta = self.beta.cuda()
            self.gamma = self.gamma.cuda()

    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __cos_sim__(self, S, Q, dim):
        return F.cosine_similarity(S.unsqueeze(1), Q.unsqueeze(2), dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def loss(self, cos_similarity, label, N, K, Q):
        L_in = torch.tensor(0., dtype=torch.float32)
        L_ood = torch.tensor(0., dtype=torch.float32)
        L_gt = torch.tensor(0., dtype=torch.float32)
        if torch.cuda.is_available():
            L_in = L_in.cuda()
            L_ood = L_ood.cuda()
            L_gt = L_gt.cuda()
        batch_size = cos_similarity.shape[0]

        for i in range(batch_size):
            batch_label = label[i, :]
            batch_cos_sim = cos_similarity[i, :, :]
            nota_index = torch.nonzero(batch_label == N)
            if len(nota_index) == 0:
                nota_index = len(batch_label)
            else:
                nota_index = nota_index[0, 0].item()
            batch_cos_sim_pos = batch_cos_sim[:nota_index, :]
            batch_cos_sim_nota = batch_cos_sim[nota_index:, :]
            target_pos = batch_label[:nota_index]
            num_nota = batch_cos_sim_nota.shape[0]
            num_pos = batch_cos_sim_pos.shape[0]

            if num_pos != 0:
                L_in += self.cost(batch_cos_sim_pos.contiguous().view(-1, N) * self.alpha,
                                  target_pos.contiguous().view(-1))
                target_pos = target_pos.view(-1, 1)
                pos_true_cos = torch.gather(batch_cos_sim_pos, dim=-1, index=target_pos)
                if torch.cuda.is_available():
                    L_gt += torch.sum(torch.max(torch.tensor(0).cuda(), self.M_2 - pos_true_cos)) / num_pos
                else:
                    L_gt += torch.sum(torch.max(torch.tensor(0), self.M_2 - pos_true_cos)) / num_pos

            if num_nota != 0:
                nota_cos_max, _ = torch.max(batch_cos_sim_nota, dim=-1)
                if torch.cuda.is_available():
                    L_ood += torch.sum(torch.max(torch.tensor(0).cuda(), nota_cos_max - self.M_1)) / num_nota
                else:
                    L_ood += torch.sum(torch.max(torch.tensor(0), nota_cos_max - self.M_1)) / num_nota

        loss = L_in + self.beta * L_ood + self.gamma * L_gt
        return loss

    def forward(self, support, query, N, K, Q):
        """
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        """
        support_emb = self.sentence_encoder(support)  # (B * N * K, D), where D is the hidden size
        query_emb = self.sentence_encoder(query)  # (B * total_Q, D)
        hidden_size = support_emb.size(-1)
        support_emb = support_emb.view(-1, N, K, hidden_size)  # (B, N, K, D)
        query_emb = query_emb.view(-1, Q, hidden_size)  # (B, total_Q, D)

        support_mean = torch.mean(support_emb, 2)  # Calculate prototype for each class
        cos_similarity = self.__cos_sim__(support_mean, query_emb, dim=3)  # (B, total, N)

        cos_similarity_max, pred = torch.max(cos_similarity, dim=-1)
        pred[cos_similarity_max < self.theta] = N

        return cos_similarity, pred
