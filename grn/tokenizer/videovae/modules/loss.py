import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

def get_disc_loss(disc_loss_type):
    if disc_loss_type == 'vanilla':
        disc_loss = vanilla_d_loss
    elif disc_loss_type == 'hinge': # this way
        disc_loss = hinge_d_loss
    return disc_loss

def adopt_weight(global_step, threshold=0, value=0., warmup=0):
    if global_step < threshold or threshold < 0:
        weight = value
    else:
        weight = 1
        if global_step - threshold < warmup:
            weight = min((global_step - threshold) / warmup, 1)
    return weight

def gradient_penalty(discriminator, real_data, fake_data, device):
    alpha = torch.rand(real_data.size(0), 1, device=device)
    alpha = alpha.expand_as(real_data)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    d_interpolates = discriminator(interpolates)
    gradients = grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, features_prime: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]

        # Normalize feature vectors
        features = F.normalize(features, dim=1)
        features_prime = F.normalize(features_prime, dim=1)
        
        # Concatenate features and features_prime
        combined_features = torch.cat([features, features_prime], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(combined_features, combined_features.T) / self.temperature
        
        # Mask to exclude self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(features.device)
        similarity_matrix.masked_fill_(mask, float('-inf'))

        # Create labels for contrastive loss
        labels = torch.arange(batch_size).to(features.device)
        labels = torch.cat([labels, labels], dim=0)
        
        # Compute logits (separate positive and negative pairs)
        positives_logits = torch.cat([similarity_matrix[:batch_size, batch_size:], similarity_matrix[batch_size:, :batch_size]], dim=0)
        
        # The labels are like: [0, 1, 2, ..., batch_size-1, 0, 1, 2, ..., batch_size-1]
        loss = F.cross_entropy(positives_logits, labels)
        
        return loss