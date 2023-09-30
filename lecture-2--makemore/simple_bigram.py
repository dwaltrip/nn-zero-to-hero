from types import SimpleNamespace

import torch
import matplotlib.pyplot as plt


START_STOP_TOKEN = '.'

class BigramModel:
    def __init__(self, words):
        self.words = words
        self.model = None

    @property
    def is_trained(self):
        return (self.model is not None) and (self.model.P is not None)

    def init_model(self):
        all_chars = sorted(list(set(''.join(self.words))))
        tokens = [START_STOP_TOKEN, *all_chars]
        num_tokens = len(tokens)

        stoi = { s: i for i, s in enumerate(tokens) }
        itos = { i: s for s, i in stoi.items() }

        N = torch.zeros((num_tokens, num_tokens), dtype=torch.int32)

        self.model = SimpleNamespace(
            num_tokens=num_tokens,
            N=N,
            P=None,
            stoi=stoi,
            itos=itos,
        )
    
    def train(self):
        self.init_model()
        model = self.model

        for ch1, ch2 in self.iter_all_char_pairs():
            ix1 = model.stoi[ch1]
            ix2 = model.stoi[ch2]
            model.N[ix1, ix2] += 1
        
        P = model.N.float()
        P += 1 # model smoothing, prevent inf loss for never seen char pairs
        P /= P.sum(1, keepdim=True)
        model.P = P
    
    def calc_training_loss(self):
        assert self.is_trained, 'Model is not trained yet.'
        log_likelihood = 0.0

        for ch1, ch2 in self.iter_all_char_pairs():
            ix1 = self.model.stoi[ch1]
            ix2 = self.model.stoi[ch2]
            log_likelihood += torch.log(self.model.P[ix1, ix2])

        n = len(list(self.iter_all_char_pairs()))
        nnll = -log_likelihood / n # normalized negative log likelihood
        return nnll
    
    def gen_word(self, generator=None):
        assert self.is_trained, 'Model is not trained yet.'
        ix = 0
        out = []

        while True:
            p = self.model.P[ix]
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

    def iter_all_char_pairs(self):
        for raw_word in self.words:
            chars = f'{START_STOP_TOKEN}{raw_word}{START_STOP_TOKEN}'
            for ch1, ch2 in zip(chars, chars[1:]):
                yield ch1, ch2

    def plot_pair_counts(self):
        assert self.is_trained, 'Model is not trained yet.'

        plt.figure(figsize=(16,16))
        plt.imshow(self.model.N, cmap='Blues')
        for i in range(self.model.num_tokens):
            for j in range(self.model.num_tokens):
                chstr = self.model.itos[i] + self.model.itos[j]
                plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
                plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')
        plt.axis('off')
        plt.show()


def main():
    words = open('names.txt', 'r').read().splitlines()
    bigram = BigramModel(words)

    bigram.train()
    loss = bigram.calc_training_loss()
    print(f'Bigram Model -- training loss: {loss.item():.4f}')
    print('----------------------------')

    g = torch.Generator().manual_seed(2147483647)
    print()
    print('Generating a few words:')
    for i in range(10):
        print('\t' + bigram.gen_word(generator=g))


if __name__ == '__main__':
    main()
