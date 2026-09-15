import torch
import torch.nn.functional as F
import time

def standard_softmax_attention(q, k, v):
    """
    Standard Transformer Softmax Attention:
    O(L^2) compute and memory.
    """
    # q, k, v: [batch, nheads, seq_len, headdim]
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    # Causal mask
    seq_len = q.size(-2)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
    scores = torch.where(mask, scores, -torch.inf)
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out

def linear_decay_attention(q, k, v, decay_rate=0.99):
    """
    Decayed Linear Attention (Mamba/SSD/GLA paradigm):
    O(L) compute, O(1) constant-memory state during generation.
    y_t = q_t @ S_t, where S_t = decay * S_{t-1} + k_t^T @ v_t
    """
    # q, k, v: [batch, nheads, seq_len, headdim]
    # Map q and k through non-negative feature map (e.g. 1 + elu(x) or silu(x))
    q_feat = F.silu(q)
    k_feat = F.silu(k)

    batch, nheads, seq_len, headdim = q.shape
    # Recurrent accumulation (or associative scan)
    # S has fixed shape [batch, nheads, headdim, headdim] INDEPENDENT of seq_len!
    S = torch.zeros(batch, nheads, headdim, headdim, device=q.device, dtype=q.dtype)
    out = torch.empty_like(v)

    for t in range(seq_len):
        # Update KV state: S = decay * S + k_t^T @ v_t
        kt = k_feat[:, :, t].unsqueeze(-1) # [B, H, D, 1]
        vt = v[:, :, t].unsqueeze(-2)      # [B, H, 1, D]
        S = decay_rate * S + torch.matmul(kt, vt) # [B, H, D, D]
        # Query output: y_t = q_t @ S
        qt = q_feat[:, :, t].unsqueeze(-2) # [B, H, 1, D]
        out[:, :, t] = torch.matmul(qt, S).squeeze(-2)
    return out

# Memory & FLOP scaling comparison
seq_lengths = [512, 1024, 2048, 4096]
batch = 1
nheads = 8
headdim = 64

print(f"{'Seq Len':<10} | {'Softmax Attn Time':<20} | {'Linear SSD Time':<20} | {'Softmax Mem (Floats)':<22} | {'Linear State Mem (Floats)'}")
print("-" * 105)

for L in seq_lengths:
    q = torch.randn(batch, nheads, L, headdim)
    k = torch.randn(batch, nheads, L, headdim)
    v = torch.randn(batch, nheads, L, headdim)

    # Time Softmax
    t0 = time.perf_counter()
    _ = standard_softmax_attention(q, k, v)
    t_soft = (time.perf_counter() - t0) * 1000

    # Time Linear
    t0 = time.perf_counter()
    _ = linear_decay_attention(q, k, v)
    t_lin = (time.perf_counter() - t0) * 1000

    softmax_mem = batch * nheads * L * L # Attention matrix elements
    linear_mem = batch * nheads * headdim * headdim # Fixed state elements

    print(f"{L:<10} | {t_soft:<20.2f} ms | {t_lin:<20.2f} ms | {softmax_mem:<22,d} | {linear_mem:<25,d}")
