import torch
from toolkit.framework import FewShotREModel


class MNAV(FewShotREModel):
    def __init__(self, sentence_encoder, NOTA_vectors, dot=True):
        FewShotREModel.__init__(self, sentence_encoder)
        self.dot = dot
        self.multiple_vector = torch.tensor(NOTA_vectors, requires_grad=False)  # (num, hidden_size)
        if torch.cuda.is_available():
            self.multiple_vector = self.multiple_vector.cuda()
        self.multiple_vector = torch.nn.Parameter(self.multiple_vector, requires_grad=True)

    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

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
        support = support_emb.view(-1, N, K, hidden_size)  # (B, N, K, D)
        query = query_emb.view(-1, Q, hidden_size)  # (B, total_Q, D)

        # Prototypical Networks
        support = torch.mean(support, 2)  # Calculate prototype for each class
        logits = self.__batch_dist__(support, query)  # (B, total_Q, N)
        logits_nota = self.__batch_dist__(self.multiple_vector.unsqueeze(0), query)  # (B, totalQ, num)
        logits_nota, _ = torch.max(logits_nota, dim=-1, keepdim=True)  # (B, totalQ, 1)
        logits = torch.cat([logits, logits_nota], 2)  # (B, total_Q, N + 1)
        _, pred = torch.max(logits, dim=-1)
        return logits, pred
