import torch
import torch.nn as nn


class ClickAttention(nn.Module):
    def __init__(self,
                 feature_dim,
                 use_attention_q,
                 use_attention_k,
                 use_attention_v,
                 residual_attention,
                 scale_attention):
        super(ClickAttention, self).__init__()
        
        # Initialize click attention components based on the specified parameters
        if use_attention_q:
            self.W_Q = nn.Linear(feature_dim, feature_dim)
        else:
            self.W_Q = self.identity

        if use_attention_k:
            self.W_Kp = nn.Linear(feature_dim, feature_dim)
            self.W_Kn = nn.Linear(feature_dim, feature_dim)
        else:
            self.W_Kp = self.identity
            self.W_Kn = self.identity

        if use_attention_v:
            self.W_Vp = nn.Linear(feature_dim, feature_dim)
            self.W_Vn = nn.Linear(feature_dim, feature_dim)
        else:
            self.W_Vp = self.identity
            self.W_Vn = self.identity

        if residual_attention:
            self.update_feature = lambda x, y: x + y
        else:
            self.update_feature = lambda x, y: x
        
        if scale_attention:
            self.scale = feature_dim ** 0.5
        else:
            self.scale = 1.0

        self.softmax = nn.Softmax(dim=-1)
    
    def identity(self, x):
        # Helper function to act as an identity operation in the absence of certain attention components
        return x
    
    def get_click_features(self, feature_field, click_inds):
        # Extract features based on click indices
        batch_size, num_verts, feature_size = feature_field.shape
        num_clicks = click_inds.shape[-1]
        ind_shift = torch.arange(0, batch_size, device=click_inds.device).view(-1, 1) * num_verts
        ind_batch = click_inds + ind_shift
        click_feats = feature_field.reshape(batch_size * num_verts, -1)[ind_batch]
        return click_feats
    
    def forward(self, feature_field, click_inds):
        # Process input features and apply attention mechanisms
        batch_size, click_num = click_inds.shape
        feature_size = feature_field.shape[-1]
        
        sel = self.get_click_features(feature_field, torch.abs(click_inds))
        sel = sel.reshape(-1, feature_size)
        pos_idx = torch.where(click_inds.reshape(-1, 1) >= 0)[0]
        neg_idx = torch.where(click_inds.reshape(-1, 1) < 0)[0]
        sel_p = sel[pos_idx]
        sel_n = sel[neg_idx]

        Q = self.W_Q(feature_field)
        Q = self.update_feature(Q, feature_field)
        
        K = torch.empty_like(sel)
        K[pos_idx] = self.W_Kp(sel_p)
        K[neg_idx] = self.W_Kn(sel_n)
        K = self.update_feature(K, sel)
        K = K.reshape(batch_size, click_num, feature_size)

        V = torch.empty_like(sel)
        V[pos_idx] = self.W_Vp(sel_p)
        V[neg_idx] = self.W_Vn(sel_n)
        V = self.update_feature(V, sel)
        V = V.reshape(batch_size, click_num, feature_size)

        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attention_weights = self.softmax(attention_scores)
        weighted_vals = torch.bmm(attention_weights, V)

        return weighted_vals
