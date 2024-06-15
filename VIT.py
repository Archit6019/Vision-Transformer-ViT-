# Vision transformer model from (https://github.com/lucidrains/vit-pytorch)

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), #Layer normalization normalizes the activations of the neurons in a layer
            nn.Linear(dim, hidden_dim), #Adds a linear transformation layer that maps the input from dimension dim to hidden_dim
            nn.GELU(), #Adds a GELU (Gaussian Error Linear Unit) activation function
            nn.Dropout(dropout), ##Dropout randomly sets a fraction of input units to 0 at each update during trainin
            nn.Linear(hidden_dim, dim), #linear transformation layer that maps hidden_dim back to the original input dimension dim
            nn.Dropout(dropout)  #Adds another dropout layer with the same dropout rate as before
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads #Total dimension after concatenating the output of all attention heads
        project_out = not (heads == 1 and dim_head == dim) # A boolean indicating whether the final projection is needed.
        self.heads = heads #Number of heads
        self.scale = dim_head ** -0.5 #Scaling factor

        self.norm = nn.LayerNorm(dim) #Layer normalization applied to the input.
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) # project the input x into query (q), key (k), and value (v) vectors.
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x) #Applies layer normalization to the input x.
        qkv = self.to_qkv(x).chunk(3, dim = -1) ##Projects the normalized input x to query, key, and value vectors
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) ##Reorganizes q, k, and v from shape (batch_size, seq_len, inner_dim) to (batch_size, heads, seq_len, dim_head)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale ##Computes the scaled dot product of q and k
        attn = self.attend(dots) #Gets attention weights from attention scores
        attn = self.dropout(attn)
        out = torch.matmul(attn, v) #Get weighted sum by multiplying attention scores with value
        out = rearrange(out, 'b h n d -> b n (h d)') #Reorganizes out back to shape (batch_size, seq_len, inner_dim).
        return self.to_out(out) #Applies the final projection

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim) #Layer normalization applied to the input and final output of the transformer.
        self.layers = nn.ModuleList([]) #Empty list
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x #Applies the attention module to the input x and adds the original input x (residual connection)
            x = ff(x) + x #Applies the feed-forward module to the updated x and adds the original input x (residual connection)
        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size) #tuple
        patch_height, patch_width = pair(patch_size) #tuple

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.' #assertion condition

        num_patches = (image_height // patch_height) * (image_width // patch_width) #Number of patches the image will be divided into
        patch_dim = channels * patch_height * patch_width #Dimensionality of each patch
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential( #Sequential module to convert the image patches to embeddings
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) #Learnable position embeddings added to the patch embeddings.
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) #Learnable class token prepended to the sequence of patch embeddings.
        self.dropout = nn.Dropout(emb_dropout) #Dropout layer for the embeddings.
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout) #Transformer module

        self.pool = pool
        self.to_latent = nn.Identity() #Identity layer for the latent representation.
        self.mlp_head = nn.Linear(dim, num_classes) #Project transformer outputs to desired number of classes

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)