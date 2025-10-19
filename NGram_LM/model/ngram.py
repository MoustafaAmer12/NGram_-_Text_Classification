import nltk
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt

# Download required NLTK data
nltk.download("gutenberg")
nltk.download("punkt")


class NGramLanguageModel:
    """
    N-Gram Language Model with Laplace (Add-1) Smoothing
    """

    def __init__(self, n, laplace_smoothing=True):
        """
        Initialize N-gram model

        Args:
            n: The n in n-gram (1 for unigram, 2 for bigram, etc.)
            laplace_smoothing: Whether to apply Laplace smoothing
        """
        self.n = n
        self.laplace_smoothing = laplace_smoothing
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocabulary = set()
        self.vocab_size = 0

    def train(self, tokens):
        """
        Train the n-gram model on a list of tokens

        Args:
            tokens: List of preprocessed tokens
        """
        # Add special start and end tokens
        tokens = ["<START>"] * (self.n - 1) + tokens + ["<END>"]

        # Build vocabulary
        self.vocabulary = set(tokens)
        self.vocab_size = len(self.vocabulary)

        # Count n-grams and contexts
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i : i + self.n])
            context = ngram[:-1]  # All words except the last
            word = ngram[-1]  # The last word

            self.ngram_counts[ngram] += 1
            if self.n > 1:
                self.context_counts[context] += 1

        # For unigram, context is empty
        if self.n == 1:
            self.context_counts[()] = sum(self.ngram_counts.values())

        print(f"{self.n}-gram model trained successfully!")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Unique {self.n}-grams: {len(self.ngram_counts)}")

    def get_probability(self, ngram):
        """
        Calculate probability of an n-gram using Laplace smoothing

        Args:
            ngram: Tuple of n words

        Returns:
            Probability of the n-gram
        """
        if len(ngram) != self.n:
            raise ValueError(f"Expected {self.n}-gram, got {len(ngram)}-gram")

        context = ngram[:-1]

        if self.laplace_smoothing:
            # Laplace smoothing: P(w|context) = (count(context, w) + 1) / (count(context) + V)
            numerator = self.ngram_counts[ngram] + 1
            denominator = self.context_counts[context] + self.vocab_size
        else:
            # MLE without smoothing
            numerator = self.ngram_counts[ngram]
            denominator = self.context_counts[context]

            if denominator == 0:
                return 0

        return numerator / denominator

    def calculate_perplexity(self, tokens):
        """
        Calculate perplexity on a test corpus

        Args:
            tokens: List of test tokens

        Returns:
            Perplexity value
        """
        # Add special tokens
        tokens = ["<START>"] * (self.n - 1) + tokens + ["<END>"]

        log_probability = 0
        ngram_count = 0

        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i : i + self.n])
            prob = self.get_probability(ngram)

            if prob > 0:
                log_probability += np.log2(prob)
            else:
                # Assign very small probability for unseen n-grams
                log_probability += np.log2(1e-10)

            ngram_count += 1

        # Perplexity = 2^(-1/N * sum(log2(P(wi))))
        perplexity = 2 ** (-log_probability / ngram_count)

        return perplexity

    def generate_text(self, num_words=20, seed_text=None):
        """
        Generate text using the trained n-gram model

        Args:
            num_words: Number of words to generate
            seed_text: Optional seed text to start generation

        Returns:
            Generated text as a string
        """
        if seed_text:
            generated = seed_text.lower().split()
        else:
            generated = ["<START>"] * (self.n - 1)

        for _ in range(num_words):
            # Get context (last n-1 words)
            context = tuple(generated[-(self.n - 1) :]) if self.n > 1 else ()

            # Find all possible next words given the context
            candidates = []
            probabilities = []

            for ngram, count in self.ngram_counts.items():
                if self.n == 1 or ngram[:-1] == context:
                    word = ngram[-1]
                    if word != "<END>":
                        candidates.append(word)
                        probabilities.append(self.get_probability(ngram))

            if not candidates:
                break

            # Normalize probabilities
            probabilities = np.array(probabilities)
            probabilities = probabilities / probabilities.sum()

            # Sample next word
            next_word = np.random.choice(candidates, p=probabilities)

            if next_word == "<END>":
                break

            generated.append(next_word)

        # Remove special tokens and join
        generated = [word for word in generated if word != "<START>"]
        return " ".join(generated)


if __name__ == "__main__":
    from NGram_LM.utils.preprocess import load_data

    train_tokens, eval_tokens = load_data()
    print("Training N-gram models from n=1 to n=10...\n")

    models = {}
    for n in range(1, 11):
        print(f"Training {n}-gram model...")
        model = NGramLanguageModel(n=n, laplace_smoothing=True)
        model.train(train_tokens)
        models[n] = model
        print()

    print("All models trained successfully!")

    print("Evaluating models on test corpus...\n")

    perplexities = {}
    for n in range(1, 11):
        print(f"Calculating perplexity for {n}-gram model...")
        perplexity = models[n].calculate_perplexity(eval_tokens)
        perplexities[n] = perplexity
        print(f"{n}-gram Perplexity: {perplexity:.2f}\n")

    # Display results in a table
    print("\n" + "=" * 50)
    print("PERPLEXITY COMPARISON")
    print("=" * 50)
    print(f"{'N-Gram Size':<15} {'Perplexity':<15}")
    print("-" * 50)
    for n in range(1, 11):
        print(f"{n:<15} {perplexities[n]:<15.2f}")
    print("=" * 50)

    # Plot perplexity vs n-gram size
    plt.figure(figsize=(12, 6))
    ns = list(perplexities.keys())
    perps = list(perplexities.values())

    plt.plot(ns, perps, marker="o", linewidth=2, markersize=8)
    plt.xlabel("N-Gram Size", fontsize=12)
    plt.ylabel("Perplexity", fontsize=12)
    plt.title("Perplexity vs N-Gram Size", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.xticks(ns)

    # Find best model
    best_n = min(perplexities, key=perplexities.get)
    plt.axvline(
        x=best_n, color="r", linestyle="--", alpha=0.5, label=f"Best Model (n={best_n})"
    )
    plt.legend()

    plt.tight_layout()
    plt.savefig("NGram_LM/results/perplexity_vs_n_gram_size.png")
    plt.show()

    print(
        f"\nBest performing model: {best_n}-gram (Perplexity: {perplexities[best_n]:.2f})"
    )

    print("=" * 50)
    print("TEXT GENERATION EXAMPLES")
    print("=" * 50)

    # Generate text with different n-gram sizes
    for n in [1, 2, 3, 4, 5]:
        print(f"\n{n}-gram generated text:")
        print("-" * 50)
        for i in range(3):
            text = models[n].generate_text(num_words=15)
            print(f"{i+1}. {text}")

    # Generate with seed text
    print("\n" + "=" * 50)
    print("GENERATION WITH SEED TEXT")
    print("=" * 50)

    seed = "the king"
    for n in [2, 3, 4]:
        print(f"\n{n}-gram with seed '{seed}':")
        text = models[n].generate_text(num_words=20, seed_text=seed)
        print(f"   {text}")

    print("\n" + "=" * 50)
    print("MODEL ANALYSIS")
    print("=" * 50)

    print("\n1. Vocabulary Statistics:")
    print(f"   - Total unique words: {models[1].vocab_size}")
    print(f"   - Most common words in training:")
    word_counts = Counter(train_tokens)
    for word, count in word_counts.most_common(10):
        print(f"     • {word}: {count}")

    print("\n2. Model Complexity:")
    for n in [1, 2, 3, 4, 5]:
        print(f"   - {n}-gram: {len(models[n].ngram_counts)} unique n-grams")

    print("\n3. Key Observations:")
    print(f"   - Lowest perplexity: {min(perplexities.values()):.2f} (n={best_n})")
    print(
        f"   - Highest perplexity: {max(perplexities.values()):.2f} (n={max(perplexities, key=perplexities.get)})"
    )
    print("   - As n increases, the model captures more context but may overfit")
    print("   - Laplace smoothing helps handle unseen n-grams gracefully")
