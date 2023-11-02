import torch
import random
from nanoFFT import BigramLanguageModel

batch_size = 64
time_intervals = 256
max_iter = 100000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 30


import tiktoken
enc = tiktoken.get_encoding("gpt2")
print('vocab size: ', enc.n_vocab)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


print("text length: ", len(text))




data = torch.tensor(enc.encode(text), dtype=torch.long)
n = int(0.7*len(data))

train_data = data[:n]
val_data = data[n:]


torch.manual_seed(1337)


def get_batch(split_type):
    data = train_data if split_type == "train" else val_data
    ix = torch.randint(len(data) - time_intervals, (batch_size, ))
    x = torch.stack([data[i:i+time_intervals] for i in ix])
    y = torch.stack([data[i+1:i+time_intervals+1] for i in ix])
    return x.to(device), y.to(device)


def get_random_block():
    data = train_data
    i = random.randint(0, len(data) - time_intervals)
    block = data[i:i+time_intervals].reshape(1, -1).to(device)
    return block

@torch.no_grad()
def estimate_loss():
    out = {}
    LLM.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = LLM(X, targets=Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    LLM.train()
    return out



LLM = BigramLanguageModel(enc.n_vocab, time_intervals, n_embed=210, n_head=30, n_layers=7).to(device)
optimizer = torch.optim.Adam(LLM.parameters(), lr=learning_rate)

pytorch_total_params = sum(p.numel() for p in LLM.parameters())
print('LLM parameters: ', pytorch_total_params)


try:
    LLM.load_state_dict(torch.load('nanoFFT_model.pt'))
    context = get_random_block()
    print(enc.decode(LLM.generate(context, max_new_tokens=500)[0].tolist()))
except:
    print("no LLM")

for iter in range(max_iter):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        context = get_random_block()
        text = enc.decode(LLM.generate(context, max_new_tokens=50)[0].tolist())[-50:]
        text = text.replace("\n", " <new line> ")
        print(f"step {iter}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}, text: {text}")
        if iter>=2000:
            try:
                torch.save(LLM.state_dict(), 'nanoFFT_model.pt')
            except:
                print("problem during saving LLM")

        if iter>=10000 and iter%10000==0:
            context = get_random_block()
            print(enc.decode(context[0].tolist()))
            print("###########################################")
            print("###########################################")
            print(enc.decode(LLM.generate(context, max_new_tokens=500)[0].tolist()))
            print("###########################################")
            print("###########################################")
    #sample batch of data
    xb, yb = get_batch("train")

    #evaluate the loss
    logits, loss = LLM(xb, targets=yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate from the LLM
#context = torch.ones((1,1), dtype=torch.long, device=device)

context = get_random_block()

print(enc.decode(context[0].tolist()))

print("###########################################")
print("###########################################")
print("###########################################")



print(enc.decode(LLM.generate(context, max_new_tokens=500)[0].tolist()))


