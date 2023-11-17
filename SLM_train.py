import torch
import random
import copy
from nanoFFT import BigramLanguageModel
import pickle

torch.manual_seed(1337)


batch_size = 64
time_intervals = 128
max_iter = 100000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 30


n_embed_LLM=150
n_head_LLM=15
n_layers_LLM = 15
n_embed_SLM=100
n_head_SLM=10
n_layers_SLM = 10



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print(device)



with open('./input/tokens.pkl', 'rb') as f:
    tokens  = pickle.load(f)

with open('./input/alice.pkl', 'rb') as f:
    input_tokens  = pickle.load(f)


vocab_size = len(tokens)
print('vocab token size: ', len(tokens))
print('input text token size: ', len(input_tokens))

stoi = {ch:i for i,ch in enumerate(tokens)}
itos = {i:ch for i,ch in enumerate(tokens)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])



data = torch.tensor(encode(input_tokens), dtype=torch.long)


torch.manual_seed(1337)


def get_batch():
    var_time = time_intervals
    ix = torch.randint(len(data) - var_time, (batch_size, ))
    x = torch.stack([data[i:i+var_time] for i in ix])
    y = torch.stack([data[i+1:i+var_time+1] for i in ix])
    return x.to(device), y.to(device)


def get_random_block():
    i = random.randint(0, len(data) - time_intervals)
    block = data[i:i+time_intervals].reshape(1, -1).to(device)
    return block.to(device)

@torch.no_grad()
def estimate_loss():
    SLM.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch()
        X_ = LLM.decode(X)
        _, _, loss = SLM(X_, targets=Y)
        losses[k] = loss.item()
    out = losses.mean()
    SLM.train()
    return out




LLM = BigramLanguageModel(vocab_size, time_intervals, vocab_embed=n_embed_LLM, n_embed=n_embed_LLM, n_head=n_head_LLM, n_layers=n_layers_LLM, device=device).to(device)
SLM = BigramLanguageModel(vocab_size, time_intervals, vocab_embed=n_embed_SLM, n_embed=n_embed_SLM, n_head=n_head_SLM, n_layers=n_layers_SLM, device=device).to(device)
optimizer = torch.optim.Adam(SLM.parameters(), lr=learning_rate)

try:
    SLM.load_state_dict(torch.load('SLM_dec_model.pt'))
    print("SLM loaded")
except:
    print("no SLM")

LLM.load_state_dict(torch.load('LLM_model.pt'))
LLM.to(device)
LLM.eval()
SLM.train()


#SLM.token_embedding_table = copy.deepcopy(LLM.token_embedding_table)
#SLM.position_embedding_table = copy.deepcopy(LLM.position_embedding_table)

pytorch_total_params = sum(p.numel() for p in LLM.parameters())
print('LLM parameters: ', pytorch_total_params)

pytorch_total_params = sum(p.numel() for p in SLM.parameters())
print('SLM parameters: ', pytorch_total_params)


context = get_random_block()
print("LLM alone:")
print("###########################################")
print(decode(LLM.generate(context, max_new_tokens=250)[0].tolist())[-250:])

print("###########################################")
print()
print("LLM with SLM:")
print("###########################################")
print(decode(SLM.generate(context, max_new_tokens=250, LLM=LLM)[0].tolist())[-250:])

print("###########################################")
print()

for iter in range(max_iter):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        context = get_random_block()
        text = decode(SLM.generate(context, max_new_tokens=50, LLM=LLM)[0].tolist())[-50:]
        text = text.replace("\n", " <new line> ")
        print(f"step {iter}, val loss: {losses:.4f}, text: {text}")
        if iter>=1000:
            try:
                torch.save(SLM.state_dict(), 'SLM_dec_model.pt')
            except:
                print("problem during saving SLM")

        if iter>=10000 and iter%10000==0:
            context = get_random_block()
            print(decode(context[0].tolist()))
            print("###########################################")
            print("###########################################")
            print(decode(SLM.generate(context, max_new_tokens=500, LLM=LLM)[0].tolist()))
            print("###########################################")
            print("###########################################")
    #sample batch of data
    xb, yb = get_batch()

    #evaluate the loss
    with torch.no_grad():
        xb_ = LLM.decode(xb)
    _, _, loss = SLM(xb_, targets=yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()



context = get_random_block()

print(decode(context[0].tolist()))

print("###########################################")
print("###########################################")
print("###########################################")



print(decode(SLM.generate(context, max_new_tokens=500, LLM=LLM)[0].tolist()))


