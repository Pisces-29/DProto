import sys
import torch
from toolkit.framework import FewShotREModel
from sklearn.neighbors import LocalOutlierFactor


class DProto(FewShotREModel):
    def __init__(self, sentence_encoder, gamma=1e-5, threshold=0.4, temperature=1):
        FewShotREModel.__init__(self, sentence_encoder)
        self.temperature_pos = torch.tensor(float(temperature), requires_grad=True)
        self.temperature_nota = torch.tensor(float(temperature), requires_grad=True)
        self.threshold = threshold
        self.gamma = torch.tensor(gamma, dtype=torch.float32)
        if torch.cuda.is_available():
            self.temperature_pos = self.temperature_pos.cuda()
            self.temperature_nota = self.temperature_nota.cuda()
            self.gamma = self.gamma.cuda()

    def __dist__(self, x, y, dim):
        return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def loss(self, logist, embedding, label, N, K, Q):
        loss_pos = torch.tensor(0., dtype=torch.float32)
        loss_nota = torch.tensor(0., dtype=torch.float32)
        if torch.cuda.is_available():
            loss_pos = loss_pos.cuda()
            loss_nota = loss_nota.cuda()
        batch_size = embedding.shape[0]
        for i in range(batch_size):
            batch_embedding = embedding[i, :, :]
            batch_label = label[i, :]
            nota_index = torch.nonzero(batch_label == N)
            if len(nota_index) == 0:
                nota_index = len(batch_label)
            else:
                nota_index = nota_index[0, 0].item()
            embedding_pos = batch_embedding[:nota_index, :]
            embedding_nota = batch_embedding[nota_index:, :]
            target_pos = batch_label[:nota_index]

            # non-NOTA loss
            n, d = embedding_pos.shape
            eye = torch.eye(target_pos.shape[0])
            if torch.cuda.is_available():
                eye = eye.cuda()

            p_norm = torch.pow(torch.cdist(embedding_pos, embedding_pos), 2)
            p_norm[p_norm < 1e-12] = 1e-12
            dist = torch.exp(-1 * p_norm / self.temperature_pos)
            if torch.cuda.is_available():
                dist = dist.cuda()

            # create matrix identifying all positive pairs
            bool_matrix = target_pos[:, None] == target_pos[:, None].T
            if torch.cuda.is_available():
                positives_matrix = (torch.as_tensor(bool_matrix, dtype=torch.int16).cuda() - eye)
                negatives_matrix = torch.as_tensor(~bool_matrix, dtype=torch.int16).cuda()
            else:
                positives_matrix = (torch.as_tensor(bool_matrix, dtype=torch.int16) - eye)
                negatives_matrix = torch.as_tensor(~bool_matrix, dtype=torch.int16)

            denominators = torch.sum(dist * negatives_matrix, dim=0)
            numerators = torch.sum(dist * positives_matrix, dim=0)
            denominators[denominators < 1e-12] = 1e-12

            frac = numerators / (numerators + denominators)
            batch_loss_pos = -1 * torch.sum(torch.log(frac[frac >= 1e-12])) / n
            loss_pos += batch_loss_pos

            n, _ = embedding_nota.shape
            if n != 0:
                p_norm_nota = torch.pow(torch.cdist(embedding_pos.detach(), embedding_nota), 2)
                p_norm_nota[p_norm_nota < 1e-12] = 1e-12
                dist_nota = torch.exp(-1 * p_norm_nota / self.temperature_nota)
                if torch.cuda.is_available():
                    dist_nota = dist_nota.cuda()
                numerators_nota = torch.sum(dist_nota, dim=0)
                batch_loss_nota = -1 * torch.sum(torch.log(numerators_nota[numerators_nota >= 1e-12])) / n
                loss_nota += batch_loss_nota

        return loss_pos - self.gamma * loss_nota

    def forward(self, support, query, N, K, Q):
        support_emb = self.sentence_encoder(support)  # (B * N * K, D), where D is the hidden size
        query_emb = self.sentence_encoder(query)  # (B * total_Q, D)
        hidden_size = support_emb.size(-1)
        support = support_emb.view(-1, N, K, hidden_size)  # (B, N, K, D)
        query = query_emb.view(-1, Q, hidden_size)  # (B, total_Q, D)
        batch_size = support.shape[0]

        # for compute loss
        support_loss = support_emb.view(-1, N * K, hidden_size)  # (B, N * K, D)
        cat_embedding = torch.cat((support_loss, query), dim=1)  # (B, N * K + total_Q, D)

        # Prototypical Networks
        support_mean = torch.mean(support, 2)  # Calculate prototype for each
        # The distance between each query instance and the prototype
        logits = self.__batch_dist__(support_mean, query)  # (B, total_Q, N)
        _, pred = torch.max(logits.view(-1, N), 1)
        pred = pred.view(-1, Q)

        # for prediction
        # Add prototypes to the collection
        lof_embedding = torch.cat((support_loss, support_mean), dim=1)

        # Add degenerated prototypes to the collection
        for i in range(K):
            temp_embedding = torch.cat((support[:, :, :i, :], support[:, :, i + 1:, :]), dim=2)
            temp_embedding_mean = torch.mean(temp_embedding, dim=2)
            lof_embedding = torch.cat((lof_embedding, temp_embedding_mean), dim=1)

        # Add query instances to the collection
        lof_embedding = torch.cat((lof_embedding, query), dim=1)

        # compute query instance's base score
        distance_min, _ = torch.max(logits, dim=-1)
        distance_max, _ = torch.min(logits, dim=-1)
        base_score_query = distance_min / distance_max

        # compute LOF
        for i in range(batch_size):
            clf = LocalOutlierFactor(n_neighbors=K, p=2)
            clf.fit(lof_embedding[i].detach().cpu().numpy())
            lof_query = -clf.negative_outlier_factor_[-Q:]
            lof_query = torch.as_tensor(lof_query)
            if torch.cuda.is_available():
                lof_query = lof_query.cuda()

            base_nota_score = base_score_query[i]

            nota_score = lof_query * base_nota_score
            temp_pred = pred[i]
            temp_pred[nota_score > self.threshold] = N
            pred[i] = temp_pred

        return logits, cat_embedding, pred
