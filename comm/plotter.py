import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
        self.loss_train = []
        self.loss_valid = []
        self.perplexity_train = []
        self.perplexity_valid = []

    def update(self, train_loss, valid_loss, train_ppl, valid_ppl):
        self.loss_train.append(train_loss)
        self.loss_valid.append(valid_loss)
        self.perplexity_train.append(train_ppl)
        self.perplexity_valid.append(valid_ppl)

    def plot(self, save_path='./training_validation_curves.png'):
        epochs = range(1, self.num_epochs + 1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.loss_train, label='Train Loss')
        plt.plot(epochs, self.loss_valid, label='Valid Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.perplexity_train, label='Train Perplexity')
        plt.plot(epochs, self.perplexity_valid, label='Valid Perplexity')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.title('Perplexity Curve')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Training and validation curves saved to {save_path}")
        plt.show()