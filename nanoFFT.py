import torch
import torch.nn as nn
import torch.nn.functional as F

dropout = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

torch.manual_seed(1337)




class FeedForward(nn.Module):

    def __init__(self, f_in, f_out):
        super().__init__()


        self.net = nn.Sequential(
            nn.LayerNorm(f_in),
            nn.Linear(f_in, f_in),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(f_in, f_out),
            nn.Dropout(dropout)
        )

    def forward(self, input):
        return self.net(input)


class FourierTransform(nn.Module):
    def __init__(self, time_intervals, f_in, f_out):
        super().__init__()

        self.value = nn.Linear(f_in, f_in, bias=False)
        self.fft = nn.Linear(time_intervals, time_intervals, bias=False)
        self.project = nn.Linear(f_in, f_out, bias=False)

        self.tril = torch.tril(torch.ones((time_intervals, time_intervals))).to(device)
        self.tril_W = self.tril/self.tril.sum(dim=1, keepdim=True)

        
    def forward(self, input):

        B,T,E = input.shape
        x = self.value(input)
        x =x.reshape(B, E, T)
        x = nn.functional.linear(x, self.tril_W[:T,:T] * self.fft.weight, self.fft.bias)
        x = x.reshape(B, T, E)
        return self.project(x)



class Block(nn.Module):
    def __init__(self,  time_intervals, n_embed, n_head):
        super().__init__()
        head_size = n_embed//n_head
        self.heads = nn.ModuleList([FourierTransform(time_intervals, n_embed, head_size) for i in range(n_head)])
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
    def __init__(self, vocab_size, time_intervals, n_embed, n_head, n_layers):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(time_intervals, n_embed)

        self.blocks = nn.Sequential(*[Block(time_intervals, n_embed, n_head) for _ in range(n_layers)])

        self.normalize = nn.LayerNorm(n_embed)
        
        self.linear_head = nn.Linear(n_embed, vocab_size)

        self.time_intervals = time_intervals

    def etoi(self, logits):
        probs = F.softmax(logits, dim=-1)
        idxs = torch.multinomial(probs, num_samples=1).squeeze() #B,T
        return idxs

    def forward(self, idx, ext_logits=None, ext_encoder=False, ext_decoder=False, targets=None):
        
        B, T = idx.shape

        if ext_decoder: ext_tok_emb = self.token_embedding_table(self.etoi(ext_logits))

        tok_emb = self.token_embedding_table(idx) # B, T, E
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))

        
        x = tok_emb + pos_emb
        if ext_encoder: x += ext_logits
        if ext_decoder: x += ext_tok_emb

        x = self.blocks(x)
        x = self.normalize(x)
        logits = self.linear_head(x)
        if targets is None:
            loss = None
        else:
            B, T, V = logits.shape
            logits = logits.view(B*T, V)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    


    def generate(self, idx, max_new_tokens, ext_logits=None, ext_encoder=False, ext_decoder=False):
        #idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.time_intervals:]
            # get the predictions
            logits, loss = self(idx_cond, ext_logits, ext_encoder, ext_decoder)
            #focus only on the last time step
            logits = logits[:, -1, :] #become (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) #(B, C)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx