import torch
import torch.nn as nn
import torch.nn.functional as F

dropout = 0.1


torch.manual_seed(1337)




class Spike(nn.Module):
    def __init__(self, dim):
        super(Spike, self).__init__()
        self.ln = nn.LayerNorm(dim)
    def forward(self, x):
        x = self.ln(x)
        return (x>0.0)*x*torch.tanh(x)



class FeedForward(nn.Module):

    def __init__(self, f_in, f_out):
        super().__init__()


        self.net = nn.Sequential(
            nn.Linear(f_in, f_in),
            nn.ReLU(),
            nn.Linear(f_in, f_out),
            nn.Dropout(dropout)
        )

    def forward(self, input):
        return self.net(input)


class FourierTransform(nn.Module):
    def __init__(self, device, time_intervals, f_in, f_out):
        super().__init__()

        self.value = nn.Linear(f_in, f_in, bias=False)
        self.fft = nn.Linear(time_intervals, time_intervals, bias=False)
        self.lnx = nn.LayerNorm(f_in)
        self.lnt = nn.LayerNorm(time_intervals)

        
        self.project = nn.Linear(f_in, f_out, bias=False)
        self.tril = torch.tril(torch.ones((time_intervals, time_intervals))).to(device)
        self.tril_W = self.tril/self.tril.sum(dim=1, keepdim=True)
        self.step = 0

    
    def forward(self, input):
        B,T,E = input.shape
        x = self.value(input)
        if self.step%2!=0: x = self.lnx(x)
        x = x.reshape(B, E, T)
        if self.step%2==0: x = self.lnt(x)
        x = nn.functional.linear(x, self.tril_W[:T,:T] * self.fft.weight[:T,:T], None)
        x = x.reshape(B, T, E)
        self.step += 1
        return self.project(x)



class Block(nn.Module):
    def __init__(self, device, time_intervals, n_embed, n_head):
        super().__init__()
        head_size = n_embed//n_head
        self.heads = nn.ModuleList([FourierTransform(device, time_intervals, n_embed, head_size) for i in range(n_head)])
        self.ffw = FeedForward(n_embed, n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, input):
        x = self.ln1(input)
        x = x + torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.ln2(x)
        out = x + self.ffw(x)
        return out




class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, time_intervals, vocab_embed, n_embed, n_head, n_layers, device="cpu"):
        super().__init__()
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_embed)
        self.position_embedding_table = nn.Embedding(time_intervals, vocab_embed)

        self.ln_in = nn.LayerNorm(vocab_embed)
        self.uniform = nn.Linear(vocab_embed, n_embed)

        self.blocks = nn.Sequential(*[Block(device, time_intervals, n_embed, n_head) for _ in range(n_layers)])

        
        self.ln_out = nn.LayerNorm(n_embed)
        
        self.linear_head = nn.Linear(n_embed, vocab_size)

        self.time_intervals = time_intervals

        self.cdist = torch.distributions.categorical



    def forward(self, idx, targets=None):
        
        B, T = idx.shape
            
        tok_emb = self.token_embedding_table(idx) # B, T, E
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))

        x = tok_emb + pos_emb

        x = self.uniform(self.ln_in(x))

        embed  = self.ln_out(self.blocks(x))
        logits = self.linear_head(embed)
        if targets is None:
            loss = None
        else:
            B, T, V = logits.shape
            logits = logits.view(B*T, V)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return embed, logits, loss
    
    
    def decode(self, idx):
        with torch.no_grad():
            _, logits, _ = self(idx)
            probs = F.softmax(logits, dim=-1)
            m = self.cdist.Categorical(probs)
            idx = m.sample()
            return idx

    def generate(self, idx, max_new_tokens, LLM=None):
        #idx is (B, T) array of indices in the current context
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # crop idx to the last block_size tokens
                idx_cond = idx[:, -self.time_intervals:]
                # get the predictions
                idx_cond_next = LLM.decode(idx_cond) if LLM != None else idx_cond
                _, logits, _ = self(idx_cond_next)
                #focus only on the last time step
                logits = logits[:, -1, :] #become (B, C)
                # apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1) #(B, C)
                # sample from distribution
                idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
                # append sampled index to the running sequence
                idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
