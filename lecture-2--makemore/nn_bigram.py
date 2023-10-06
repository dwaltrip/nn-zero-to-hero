import string
from types import SimpleNamespace

import torch.nn.functional as F
import torch


START_STOP_TOKEN = '.'


class NeuralNetBigram:

    def __init__(self, words, generator=None):
        self.words = words
        self.model = self.init_model(generator=generator)
        self.trainset = self.compile_trainset(words)

    def compile_trainset(self, words):
        xs, ys = [], []
        for ch1, ch2 in _iter_all_char_pairs(words):
            ix1 = self.model.stoi[ch1]
            ix2 = self.model.stoi[ch2]
            xs.append(ix1)
            ys.append(ix2)
        print('number of examples:', len(xs))
        return SimpleNamespace(
            xs=torch.tensor(xs),
            ys=torch.tensor(ys),
        )

    # init the neural network
    def init_model(self, generator=None):
        tokens = [START_STOP_TOKEN, *string.ascii_lowercase]
        stoi = { s: i for i, s in enumerate(tokens) }
        itos = { i: s for s, i in stoi.items() }

        W = torch.randn((27, 27), generator=generator, requires_grad=True)

        return SimpleNamespace(
            num_tokens=len(tokens),
            W=W,
            stoi=stoi,
            itos=itos,
            is_trained=False,
        )
    
    is_trained = property(lambda self: self.model.is_trained)

    def train(self, regularization=False, verbose=False):
        W = self.model.W
        num_tokens = self.model.num_tokens
        xs, ys = self.trainset.xs, self.trainset.ys
        xenc = F.one_hot(xs, num_classes=num_tokens).float()
        num_examples = xs.nelement()

        for i in range(100):
            # forward pass
            logits = xenc @ W # predict log-counts
            counts = logits.exp() # analagous to N in hand-coded bigram
            probs = counts / counts.sum(1, keepdim=True) # probs for next char
            loss = -probs[torch.arange(num_examples), ys].log().mean()
            if regularization:
                reg_loss = 0.001 * (W**2).mean() # regularization loss
                loss += reg_loss
            if verbose:
                print(f'epoch {i} - loss:', loss.item())

            # backward pass
            W.grad = None # set grads to zero
            loss.backward()

            # update
            W.data += -20 * W.grad
        
        self.model.is_trained = True
        print()
        print(f'---- Final loss: {loss.item()} ----')

    def gen_word(self, generator=None):
        assert self.is_trained, 'Model is not trained yet.'
        ix = 0
        out = []

        while True:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
            logits = xenc @ self.model.W # predict log-counts
            counts = logits.exp() # analagous to N in hand-coded bigram
            p = counts / counts.sum(1, keepdim=True) # probs for next char
            ix = torch.multinomial(
                p,
                num_samples=1,
                replacement=True,
                generator=generator
            ).item()
            ch = self.model.itos[ix]
            if ch == START_STOP_TOKEN:
                break
            out.append(ch)

        return ''.join(out)


def _iter_all_char_pairs(words):
    for raw_word in words:
        chars = f'{START_STOP_TOKEN}{raw_word}{START_STOP_TOKEN}'
        for ch1, ch2 in zip(chars, chars[1:]):
            yield ch1, ch2


def main():
    words = open('names.txt', 'r').read().splitlines()
    bigram = NeuralNetBigram(words)

    bigram.train()

    g = torch.Generator().manual_seed(2147483647 + 1)
    print()
    print('Generating a few words:')
    for i in range(10):
        print('\t' + bigram.gen_word(generator=g))


if __name__ == '__main__':
    main()
