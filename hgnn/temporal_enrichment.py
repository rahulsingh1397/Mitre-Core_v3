import torch
import torch.nn as nn
import torch.nn.functional as F

class EntityMemoryModule(nn.Module):
    """TGN-style GRU memory per entity (src_ip, host, user) with fixes:
    - Separate GRUs per entity type (prevents convergence)
    - LayerNorm on outputs (prevents explosion)
    - Residual connection with learned alpha (prevents noise)
    """

    def __init__(self, alert_feat_dim: int = 6, mem_dim: int = 64, out_dim: int = 6):
        super().__init__()
        self.mem_dim = mem_dim
        
        # Separate GRUs per entity type (Fix A: prevents convergence)
        self.gru_ip = nn.GRUCell(alert_feat_dim, mem_dim)
        self.gru_host = nn.GRUCell(alert_feat_dim, mem_dim)
        self.gru_user = nn.GRUCell(alert_feat_dim, mem_dim)
        
        # LayerNorm on outputs (Fix B: prevents explosion)
        self.norm = nn.LayerNorm(out_dim)
        
        # Residual connection with learned alpha (Fix C: prevents noise)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
        # Projection from concatenated memories to output (same as input for residual)
        self.proj = nn.Linear(3 * mem_dim, out_dim)

    @torch.no_grad()
    def compute_contexts(self, feats, ip_ids, host_ids, user_ids):
        """Inference-time (no grad): returns [N, out_dim] temporal contexts."""
        n_ip   = int(ip_ids.max())   + 1 if len(ip_ids) > 0 else 1
        n_host = int(host_ids.max()) + 1 if len(host_ids) > 0 else 1
        n_user = int(user_ids.max()) + 1 if len(user_ids) > 0 else 1
        
        # Separate memories per entity type
        mem_ip   = feats.new_zeros(n_ip,   self.mem_dim)
        mem_host = feats.new_zeros(n_host, self.mem_dim)
        mem_user = feats.new_zeros(n_user, self.mem_dim)
        
        ctxs = []
        for i in range(len(feats)):
            f, ip, h, u = feats[i], ip_ids[i], host_ids[i], user_ids[i]
            
            # Read current contexts
            ctx = torch.cat([mem_ip[ip], mem_host[h], mem_user[u]])
            ctxs.append(ctx)
            
            # Update memories with separate GRUs
            mem_ip[ip]   = self.gru_ip(f.unsqueeze(0), mem_ip[ip].unsqueeze(0)).squeeze(0)
            mem_host[h]  = self.gru_host(f.unsqueeze(0), mem_host[h].unsqueeze(0)).squeeze(0)
            mem_user[u]  = self.gru_user(f.unsqueeze(0), mem_user[u].unsqueeze(0)).squeeze(0)
            
        if not ctxs:
            return feats.new_zeros(0, self.proj.out_features)
            
        # Apply projection and LayerNorm
        contexts = self.proj(torch.stack(ctxs))  # [N, out_dim]
        return self.norm(contexts)

    def forward(self, feats, ip_ids, host_ids, user_ids):
        """Training-time (with grad): same logic but grad-enabled."""
        n_ip   = int(ip_ids.max())   + 1 if len(ip_ids) > 0 else 1
        n_host = int(host_ids.max()) + 1 if len(host_ids) > 0 else 1
        n_user = int(user_ids.max()) + 1 if len(user_ids) > 0 else 1
        
        # Separate memories per entity type (clone to prevent inplace issues)
        mem_ip   = feats.new_zeros(n_ip,   self.mem_dim).clone()
        mem_host = feats.new_zeros(n_host, self.mem_dim).clone()
        mem_user = feats.new_zeros(n_user, self.mem_dim).clone()
        
        ctxs = []
        for i in range(len(feats)):
            f, ip, h, u = feats[i], ip_ids[i], host_ids[i], user_ids[i]
            
            # Read current contexts
            ctx = torch.cat([mem_ip[ip], mem_host[h], mem_user[u]])
            ctxs.append(ctx)
            
            # Update memories with separate GRUs (use clone to prevent inplace)
            mem_ip[ip]   = self.gru_ip(f.unsqueeze(0), mem_ip[ip].unsqueeze(0)).squeeze(0).clone()
            mem_host[h]  = self.gru_host(f.unsqueeze(0), mem_host[h].unsqueeze(0)).squeeze(0).clone()
            mem_user[u]  = self.gru_user(f.unsqueeze(0), mem_user[u].unsqueeze(0)).squeeze(0).clone()
            
        if not ctxs:
            return feats.new_zeros(0, self.proj.out_features)
            
        # Apply projection and LayerNorm
        contexts = self.proj(torch.stack(ctxs))  # [N, out_dim]
        return self.norm(contexts)
