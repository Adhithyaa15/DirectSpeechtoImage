import torch
import torch.nn as nn
import torch.nn.functional as F


def assign_to_clusters(embeddings, centroids):
    """
    Find nearest cluster for each token using L2 distance
    
    Args:
        embeddings: [B, 77, 768] - predicted or ground truth embeddings
        centroids: [n_clusters, 768] - cluster centers
    
    Returns:
        cluster_ids: [B, 77] - nearest cluster index for each token
    """
    B, T, D = embeddings.shape
    
    # Reshape for batched distance computation
    embeddings_flat = embeddings.reshape(-1, D)  # [B*77, 768]
    
    # Compute L2 distance to all centroids
    # distances[i, j] = ||embedding[i] - centroid[j]||^2
    distances = torch.cdist(embeddings_flat, centroids)  # [B*77, n_clusters]
    
    # Find nearest cluster
    cluster_ids = distances.argmin(dim=1)  # [B*77]
    cluster_ids = cluster_ids.reshape(B, T)  # [B, 77]
    
    return cluster_ids


class MSELoss(nn.Module):
    """Mean Squared Error loss for embedding alignment"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, 77, 768]
            targets: [B, 77, 768]
        Returns:
            loss: scalar
            loss_dict: dictionary with loss information
        """
        loss = self.mse(predictions, targets)
        
        loss_dict = {
            'total_loss': loss.item(),
            'mse_loss': loss.item()
        }
        
        return loss, loss_dict


class CosineSimilarityLoss(nn.Module):
    """Cosine Similarity loss for embedding alignment
    
    Measures the cosine similarity between predicted and target embeddings.
    Loss = 1 - cosine_similarity (so 0 is perfect, 2 is worst)
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, 77, 768]
            targets: [B, 77, 768]
        Returns:
            loss: scalar
            loss_dict: dictionary with loss information
        """
        # Normalize embeddings to unit vectors
        predictions_norm = F.normalize(predictions, p=2, dim=-1)
        targets_norm = F.normalize(targets, p=2, dim=-1)
        
        # Compute cosine similarity (element-wise dot product)
        # Result: [B, 77], values in [-1, 1]
        cosine_sim = (predictions_norm * targets_norm).sum(dim=-1)
        
        # Convert to loss: 1 - cosine_similarity
        # Perfect alignment: cos_sim=1 → loss=0
        # Orthogonal: cos_sim=0 → loss=1
        # Opposite: cos_sim=-1 → loss=2
        loss = (1 - cosine_sim).mean()
        
        loss_dict = {
            'total_loss': loss.item(),
            'cosine_loss': loss.item(),
            'avg_cosine_sim': cosine_sim.mean().item()
        }
        
        return loss, loss_dict


class CombinedLoss(nn.Module):
    """Combined MSE and Cosine Similarity loss
    
    Total Loss = mse_weight * MSE + cosine_weight * (1 - CosineSimilarity)
    
    MSE captures magnitude differences (absolute values)
    Cosine captures directional alignment (angles between vectors)
    """
    def __init__(self, mse_weight=1.0, cosine_weight=1.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, 77, 768]
            targets: [B, 77, 768]
        Returns:
            loss: scalar
            loss_dict: dictionary with loss information
        """
        # Compute MSE loss
        mse_loss = self.mse(predictions, targets)
        
        # Compute Cosine Similarity loss
        predictions_norm = F.normalize(predictions, p=2, dim=-1)
        targets_norm = F.normalize(targets, p=2, dim=-1)
        cosine_sim = (predictions_norm * targets_norm).sum(dim=-1)
        cosine_loss = (1 - cosine_sim).mean()
        
        # Combined loss
        total_loss = self.mse_weight * mse_loss + self.cosine_weight * cosine_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'cosine_loss': cosine_loss.item(),
            'avg_cosine_sim': cosine_sim.mean().item()
        }
        
        return total_loss, loss_dict

class ClusterClassificationLoss(nn.Module):
    """
    Cross-Entropy loss for cluster classification
    Predicts which of n_clusters centroids each token belongs to
    """
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, cluster_logits, ground_truth_embeddings, centroids):
        """
        Args:
            cluster_logits: [B, 77, n_clusters] - predicted cluster scores from model
            ground_truth_embeddings: [B, 77, 768] - ground truth CLIP-L embeddings
            centroids: [n_clusters, 768] - cluster centers
        
        Returns:
            loss: scalar cross-entropy loss
            loss_dict: dictionary with loss information
        """
        # Find which cluster each ground truth token belongs to
        with torch.no_grad():
            cluster_labels = assign_to_clusters(
                ground_truth_embeddings, centroids
            )  # [B, 77]
        
        # Reshape for cross entropy
        B, T, C = cluster_logits.shape
        logits_flat = cluster_logits.reshape(B * T, C)  # [B*77, n_clusters]
        labels_flat = cluster_labels.reshape(-1)  # [B*77]
        
        # Compute cross-entropy loss
        loss = self.ce_loss(logits_flat, labels_flat)
        
        # Compute accuracy for monitoring
        with torch.no_grad():
            predictions = logits_flat.argmax(dim=1)
            accuracy = (predictions == labels_flat).float().mean().item()
        
        loss_dict = {
            'ce_loss': loss.item(),
            'cluster_accuracy': accuracy
        }
        
        return loss, loss_dict


class CentroidContrastiveLoss(nn.Module):
    """
    Distance-based contrastive loss between predicted embeddings and centroids
    Uses L2 distance on normalized vectors: L_i = -log(exp(-||z_i - c_y||²/τ) / Σ_k exp(-||z_i - c_k||²/τ))
    Normalizing keeps distances bounded [0, 2] for numerical stability
    Pulls predictions close to correct centroid, pushes away from others
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, predictions, ground_truth_embeddings, centroids):
        """
        Args:
            predictions: [B, 77, 768] - predicted embeddings from decoder
            ground_truth_embeddings: [B, 77, 768] - ground truth CLIP-L embeddings
            centroids: [n_clusters, 768] - cluster centers
        
        Returns:
            loss: scalar contrastive loss
            loss_dict: dictionary with loss information
        """
        B, T, D = predictions.shape
        
        # Find ground truth cluster assignments
        with torch.no_grad():
            cluster_labels = assign_to_clusters(
                ground_truth_embeddings, centroids
            )  # [B, 77]
        
        # Normalize embeddings and centroids to unit vectors (bounds distances to [0, 2])
        pred_norm = F.normalize(predictions, p=2, dim=-1)  # [B, 77, 768]
        centroids_norm = F.normalize(centroids, p=2, dim=-1)  # [n_clusters, 768]
        
        # Compute L2 distances to all centroids (now bounded)
        pred_flat = pred_norm.reshape(-1, D)  # [B*77, 768]
        distances = torch.cdist(pred_flat, centroids_norm)  # [B*77, n_clusters], range [0, 2]
        
        # Convert distances to logits: -||z_i - c_k||²/τ
        # This matches the formula: exp(-||z_i - c_k||²/τ)
        # With normalized vectors, distance² ∈ [0, 4], so logits ∈ [-57, 0] for τ=0.07
        logits = -(distances ** 2) / self.temperature  # [B*77, n_clusters]
        
        # Labels are the assigned clusters
        labels = cluster_labels.reshape(-1)  # [B*77]
        
        # Cross entropy: minimize distance to correct cluster, maximize distance to others
        # Loss = -log(exp(-||z_i - c_correct||²/τ) / Σ_k exp(-||z_i - c_k||²/τ))
        loss = F.cross_entropy(logits, labels)
        
        # Compute metrics for monitoring
        with torch.no_grad():
            # Average distance to correct cluster (lower is better)
            correct_distances = torch.gather(
                distances, 1, labels.unsqueeze(1)
            ).squeeze(1)
            avg_positive_dist = correct_distances.mean().item()
            
            # Average distance to incorrect clusters (higher is better)
            mask = torch.ones_like(distances).scatter_(1, labels.unsqueeze(1), 0)
            avg_negative_dist = (distances * mask).sum() / mask.sum()
            avg_negative_dist = avg_negative_dist.item()
            
            # Distance margin: how much farther are wrong centroids vs correct one
            dist_margin = avg_negative_dist - avg_positive_dist
        
        loss_dict = {
            'contrastive_loss': loss.item(),
            'avg_positive_sim': -avg_positive_dist,  # Negative distance for backward compatibility
            'avg_negative_sim': -avg_negative_dist,  # Negative distance for backward compatibility
            'sim_margin': dist_margin  # Now represents distance margin (positive is good)
        }
        
        return loss, loss_dict


class CombinedCentroidLoss(nn.Module):
    """
    Combined loss: MSE + Cosine Similarity + Cluster Classification + Contrastive
    
    Total Loss = mse_weight * MSE 
               + cosine_weight * (1 - CosineSimilarity)
               + ce_weight * CrossEntropy(cluster_logits, cluster_labels)
               + contrastive_weight * Contrastive(predictions, centroids)
    """
    def __init__(self, mse_weight=1.0, cosine_weight=1.0, ce_weight=0.5, contrastive_weight=0.5):
        super().__init__()
        self.mse_loss = MSELoss()
        self.cosine_loss = CosineSimilarityLoss()
        self.ce_loss = ClusterClassificationLoss()
        self.contrastive_loss = CentroidContrastiveLoss()
        
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.ce_weight = ce_weight
        self.contrastive_weight = contrastive_weight
    
    def forward(self, predictions, cluster_logits, ground_truth, centroids):
        """
        Args:
            predictions: [B, 77, 768] - predicted embeddings from decoder
            cluster_logits: [B, 77, n_clusters] - cluster classification scores
            ground_truth: [B, 77, 768] - ground truth CLIP-L embeddings
            centroids: [n_clusters, 768] - cluster centers
        
        Returns:
            total_loss: weighted combination of all losses
            loss_dict: dictionary with all loss components
        """
        # MSE loss (token-level reconstruction)
        mse, mse_dict = self.mse_loss(predictions, ground_truth)
        
        # Cosine similarity loss (directional alignment)
        cosine, cosine_dict = self.cosine_loss(predictions, ground_truth)
        
        # CE loss (cluster classification head)
        ce, ce_dict = self.ce_loss(cluster_logits, ground_truth, centroids)
        
        # Contrastive loss (pull toward correct centroid)
        cont, cont_dict = self.contrastive_loss(predictions, ground_truth, centroids)
        
        # Combined weighted loss
        total_loss = (
            self.mse_weight * mse +
            self.cosine_weight * cosine +
            self.ce_weight * ce +
            self.contrastive_weight * cont
        )
        
        # Aggregate loss dictionary
        loss_dict = {
            'total_loss': total_loss.item(),
            'mse_loss': mse_dict['mse_loss'],
            'cosine_loss': cosine_dict['cosine_loss'],
            'avg_cosine_sim': cosine_dict['avg_cosine_sim'],
            'ce_loss': ce_dict['ce_loss'],
            'cluster_accuracy': ce_dict['cluster_accuracy'],
            'contrastive_loss': cont_dict['contrastive_loss'],
            'avg_positive_sim': cont_dict['avg_positive_sim'],
            'avg_negative_sim': cont_dict['avg_negative_sim'],
            'sim_margin': cont_dict['sim_margin']
        }
        
        return total_loss, loss_dict

def test_losses():
    """Test loss functions"""
    batch_size = 4
    seq_len = 77
    hidden_dim = 768
    
    # Create dummy data
    predictions = torch.randn(batch_size, seq_len, hidden_dim)
    targets = torch.randn(batch_size, seq_len, hidden_dim)
    
    print("Testing MSE Loss:")
    mse_loss_fn = MSELoss()
    total_loss, loss_dict = mse_loss_fn(predictions, targets)
    print(f"  Loss: {total_loss.item():.4f}")
    print(f"  Dict: {loss_dict}\n")
    
    print("Testing Cosine Similarity Loss:")
    cosine_loss_fn = CosineSimilarityLoss()
    total_loss, loss_dict = cosine_loss_fn(predictions, targets)
    print(f"  Loss: {total_loss.item():.4f}")
    print(f"  Dict: {loss_dict}\n")
    
    print("Testing Combined Loss (MSE + Cosine):")
    combined_loss_fn = CombinedLoss(mse_weight=1.0, cosine_weight=1.0)
    total_loss, loss_dict = combined_loss_fn(predictions, targets)
    print(f"  Loss: {total_loss.item():.4f}")
    print(f"  Dict: {loss_dict}")


if __name__ == "__main__":
    test_losses()
