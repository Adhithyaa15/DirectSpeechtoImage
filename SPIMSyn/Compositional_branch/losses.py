import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    def forward(self, pred, target):
        return F.mse_loss(pred, target)


class CosineSimilarityLoss(nn.Module):
    def forward(self, pred, target):
        p = pred.reshape(-1, pred.size(-1))
        t = target.reshape(-1, target.size(-1))
        return (1 - F.cosine_similarity(p, t, dim=-1)).mean()


class ClusterClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, target_embeddings, centroids):
        t = target_embeddings.reshape(-1, target_embeddings.size(-1))
        c = centroids.to(t.device)
        dists = torch.cdist(t, c)
        labels = dists.argmin(dim=-1)
        lg = logits.reshape(-1, logits.size(-1))
        return self.ce(lg, labels)


class CentroidContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, pred, target_embeddings, centroids):
        p = pred.reshape(-1, pred.size(-1))
        t = target_embeddings.reshape(-1, target_embeddings.size(-1))
        c = centroids.to(p.device)
        dists_t = torch.cdist(t, c)
        pos_idx = dists_t.argmin(dim=-1)
        pos_centroids = c[pos_idx]
        pos_dist = F.mse_loss(p, pos_centroids)
        dists_p = torch.cdist(p, c)
        neg_idx = dists_p.argmin(dim=-1)
        neg_centroids = c[neg_idx]
        neg_dist = (pos_centroids - neg_centroids).pow(2).sum(dim=-1).sqrt().mean()
        return torch.clamp(pos_dist - neg_dist + self.margin, min=0)


class CombinedCentroidLoss(nn.Module):
    def __init__(self, mse_w=1.0, cosine_w=1.0, ce_w=0.5, contrastive_w=0.5):
        super().__init__()
        self.mse = MSELoss()
        self.cosine = CosineSimilarityLoss()
        self.ce_loss = ClusterClassificationLoss()
        self.contrastive = CentroidContrastiveLoss()
        self.mse_w = mse_w
        self.cosine_w = cosine_w
        self.ce_w = ce_w
        self.contrastive_w = contrastive_w

    def forward(self, pred, target, cluster_logits=None, centroids=None):
        loss = self.mse_w * self.mse(pred, target) + self.cosine_w * self.cosine(pred, target)
        if cluster_logits is not None and centroids is not None:
            loss += self.ce_w * self.ce_loss(cluster_logits, target, centroids)
            loss += self.contrastive_w * self.contrastive(pred, target, centroids)
        return loss
