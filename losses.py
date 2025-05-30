import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricCrossEntropyLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        pred = torch.nn.functional.softmax(logits, dim=1)
        target_onehot = torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float()
        rce = (-pred * torch.log(target_onehot + 1e-6)).sum(dim=1)
        return self.alpha * self.ce(logits, targets) + self.beta * rce.mean()


class NCODPlusLoss(nn.Module):
    def __init__(self, num_classes, emb_dim, lambda_c=1.0, lambda_b=1.0, lambda_elr=3.0, ema_momentum=0.9, num_samples=20000):
        super().__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.lambda_c = lambda_c
        self.lambda_b = lambda_b
        self.lambda_elr = lambda_elr
        self.ema_momentum = ema_momentum

        self.register_buffer("centroids", torch.zeros(num_classes, emb_dim))
        self.class_counts = torch.zeros(num_classes, dtype=torch.long)
        self.u = nn.Parameter(torch.ones(1))

        # ELR memory buffer
        self.register_buffer("target_probs", torch.zeros(num_samples, num_classes))

    def update_centroids(self, embeddings, targets):
        for i in range(self.num_classes):
            mask = (targets == i)
            if mask.any():
                self.centroids[i] = embeddings[mask].mean(dim=0).detach()
                self.class_counts[i] = mask.sum().item()

    def forward(self, embeddings, logits, targets, alt_logits=None, indices=None):
        device = logits.device
        ce_loss = F.cross_entropy(logits, targets)

        # NCOD: similarity-based soft labels
        sim = F.cosine_similarity(embeddings.unsqueeze(1), self.centroids.to(device).unsqueeze(0), dim=2)
        soft_labels = F.softmax(sim, dim=1)
        one_hot_targets = F.one_hot(targets, num_classes=self.num_classes).float().to(device)

        similarity_to_target = (soft_labels * one_hot_targets).sum(dim=1)
        discount = 1.0 - similarity_to_target
        reg_term = discount.mean() * self.u.to(device)

        mse_term = F.mse_loss(soft_labels, one_hot_targets.detach())
        loss = ce_loss + mse_term + reg_term

        # NCOD+ ensemble regularization
        if alt_logits is not None:
            p1 = F.softmax(logits, dim=1)
            p2 = F.softmax(alt_logits, dim=1)
            avg_pred = (p1 + p2) / 2
            uniform_dist = torch.full_like(avg_pred, 1.0 / self.num_classes).to(device)

            kl_c = F.kl_div(p1.log(), p2, reduction='batchmean') + F.kl_div(p2.log(), p1, reduction='batchmean')
            kl_b = F.kl_div(avg_pred.log(), uniform_dist, reduction='batchmean')
            loss += self.lambda_c * kl_c + self.lambda_b * kl_b

        # 🔁 ELR Regularization
        if indices is not None:
            indices_cpu = indices.cpu()  # Ensure safe indexing into CPU buffer

            with torch.no_grad():
                probs = F.softmax(logits.detach(), dim=1)
                self.target_probs[indices_cpu] = (
                    self.ema_momentum * self.target_probs[indices_cpu] +
                    (1 - self.ema_momentum) * probs.cpu()
                )

            q_i = self.target_probs[indices_cpu].to(device)  # Also index using CPU version
            elr_reg = -torch.mean(torch.sum(F.softmax(logits, dim=1) * torch.log(1.0 - q_i + 1e-6), dim=1))
            loss += self.lambda_elr * elr_reg

        return loss
