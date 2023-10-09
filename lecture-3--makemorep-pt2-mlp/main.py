import os
import random
import string
from types import SimpleNamespace

import torch
import torch.nn.functional as F


NAMES_FILEPATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'names.txt'
)

START_STOP_TOKEN = '.'

BLOCK_SIZE = 3 # context length, number of chars used to predict the next one
EMB_DIM_SIZE = 8
EMB_DIM_SIZE = 4
HIDDEN_LAYER_SIZE = 100
MINIBATCH_SIZE = 64

# @dataclass
# class Params:
#     BLOCK_SIZE: int
#     BATCH_SIZE: int
#     HIDDEN_LAYER_SIZE: int


def main():
    words = open(NAMES_FILEPATH, 'r').read().splitlines()

    tokens = setup_tokens()
    trainset, devset, testset = setup_datasets(words, tokens)

    print(f'trainset: {len(trainset.X):,}')
    print(f'devset  : {len(devset.X):,}')
    print(f'testset : {len(testset.X):,}')

    g = torch.Generator().manual_seed(2147483647)
    model = setup_model(tokens, generator=g)
    train_model(model, trainset)

    train_loss = eval_model(model, trainset)
    dev_loss = eval_model(model, devset)
    print(f'train loss: {train_loss.item():.4f}')
    print(f'dev loss: {dev_loss.item():.4f}')

    print()
    print('--- Generating words ---')
    g = torch.Generator().manual_seed(2147483647 + 10)
    for _ in range(20):
        word = generate_word(model, tokens, generator=g)
        print(word)


def setup_model(tokens, generator=None):
    randn = lambda *shape: torch.randn(shape, generator=generator)

    num_tokens = tokens.num_tokens
    C = randn(num_tokens, EMB_DIM_SIZE)
    W1 = randn(BLOCK_SIZE * EMB_DIM_SIZE, HIDDEN_LAYER_SIZE)
    b1 = randn(HIDDEN_LAYER_SIZE)
    W2 = torch.randn(HIDDEN_LAYER_SIZE, num_tokens)
    b2 = torch.randn(num_tokens)
    parameters = [C, W1, b1, W2, b2]

    for p in parameters:
        p.requires_grad = True

    return SimpleNamespace(
        C=C,
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
        parameters=parameters,
    )


def train_model(model, trainset):
    print()
    print('Training model...')
    print('-------------------------------')
    print('Context length:', BLOCK_SIZE)
    print('Number of embedding dims:', EMB_DIM_SIZE)
    print('Minibatch size:', MINIBATCH_SIZE)
    print('Hidden layer size:', HIDDEN_LAYER_SIZE)
    print('Number of params:', sum(p.nelement() for p in model.parameters))
    print('-------------------------------')
    print()

    X, Y = trainset.X, trainset.Y

    # NUM_ITERS = 200_000
    NUM_ITERS = 40_000
    print(f'Training for {NUM_ITERS:,} iterations...')

    for i in range(NUM_ITERS):
        # -- get minibactch --
        ix = torch.randint(0, X.shape[0], (MINIBATCH_SIZE,))
        x, y = X[ix], Y[ix]
        
        # -- forward pass --
        emb = model.C[x]
        h = torch.tanh(
            emb.view(-1, BLOCK_SIZE * EMB_DIM_SIZE) @ model.W1 +
            model.b1
        )
        logits = h @ model.W2 + model.b2
        loss = F.cross_entropy(logits, y)
        
        # -- backward pass --
        for p in model.parameters:
            p.grad = None
        loss.backward()
        
        # -- update --
        if i < (NUM_ITERS * 0.7):
            lr = 0.1
        else:
            lr = 0.01
        for p in model.parameters:
            p.data += -lr * p.grad

        if (i+1) % int(NUM_ITERS / 20) == 0:
            print(f'{100*(i / NUM_ITERS):3.0f}% -- loss: {loss.item():.4f}')


def generate_word(model, tokens, generator=None):
    out = []
    context = [0] * BLOCK_SIZE
    while True:
        emb = model.C[torch.tensor([context])] # (1, block_size, d)
        h = torch.tanh(emb.view(1, -1) @ model.W1 + model.b1)
        logits = h @ model.W2 + model.b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=generator).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break

    return ''.join(tokens.itos[i] for i in out)


def eval_model(model, dataset):
    X, Y = dataset.X, dataset.Y
    emb = model.C[X]
    h = torch.tanh(
        emb.view(-1, BLOCK_SIZE * EMB_DIM_SIZE) @ model.W1 +
        model.b1
    )
    logits = h @ model.W2 + model.b2
    loss = F.cross_entropy(logits, Y)
    return loss


def setup_tokens():
    tokens = [START_STOP_TOKEN, *string.ascii_lowercase]
    stoi = { s: i for i, s in enumerate(tokens) }
    itos = { i: s for s, i in stoi.items() }
    num_tokens = len(tokens)
    return SimpleNamespace(
        tokens=tokens,
        stoi=stoi,
        itos=itos,
        num_tokens=num_tokens,
    )


def setup_datasets(words, tokens):
    random.seed(42)
    shuffled_words = words[:]
    random.shuffle(shuffled_words)

    n1 = int(0.8*len(shuffled_words))
    n2 = int(0.9*len(shuffled_words))

    Xtr, Ytr = build_dataset(shuffled_words[:n1], tokens)
    Xdev, Ydev = build_dataset(shuffled_words[n1:n2], tokens)
    Xte, Yte = build_dataset(shuffled_words[n2:], tokens)
    return (
        SimpleNamespace(X=Xtr, Y=Ytr),
        SimpleNamespace(X=Xdev, Y=Ydev),
        SimpleNamespace(X=Xte, Y=Yte),
    )


def build_dataset(words, tokens):
    X, Y = [], []
    for w in words:
        context = [0] * BLOCK_SIZE
        for ch in w + START_STOP_TOKEN:
            ix = tokens.stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


# ------------------------------------------------------------

if __name__ == '__main__':
    main()
