class WeightedPairwiseHinge(nn.Module):
    def __init__(self, margin=0.1, p=1.0, eps=1e-6, initial_max_loss_coeff=0.0):
        super().__init__()
        self.margin = margin
        self.p      = p
        self.eps    = eps
        self.max_loss_coeff = initial_max_loss_coeff

    def forward(self, o1, o2, y1, y2, target):
        diff   = o1 - o2
        label_diff = torch.abs(y1 - y2)
        
        # Calculate dynamic margin
        dynamic_margin = self.margin * label_diff
    	
    	# Calculate the Hinge
        hinge = F.relu(dynamic_margin - target * diff)
        
        # Calculate the importance weight
        weight = label_diff.pow(self.p)

        weighted_hinge = weight * hinge
        mean_loss = weighted_hinge.sum() / (weight.sum() + self.eps)
	
	# Calculate the optional max loss term
        max_weighted_hinge = weighted_hinge.max()
        
        # Combine mean and max loss using the current max_loss_coeff
        loss = (1-self.max_loss_coeff) * mean_loss + self.max_loss_coeff * max_weighted_hinge

        # Calculate the accuracy
        preds = (diff > 0).float()
        labels = (target > 0).float()
        acc = (preds == labels).float().mean()

        return loss, acc