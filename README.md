# Performance-first Tokenizer implementation in PHP

```
composer require neuron-core/tokeinzer
```

## BPE (Byte-Pair Encoding) Tokenizer

This implementation requires an already trained vocabulary and merges files. It's focused on performance
with cache strategy and priority queue.

```php
$tokenizer = new BPETokenizer();

// Initialize pre-trained tokenizer
$tokenizer->loadFrom(
    __DIR__.'/../dataset/bpe-vocabulary.json',
    __DIR__.'/../dataset/bpe-merges.txt'
);

// Tokenize text
$tokens = $tokenizer->encode('Hello world!');

var_dump($tokens);
```

Here you can find the already trained GPT-2 datasets:

- Vocabulary: https://huggingface.co/gpt2/resolve/main/vocab.json
- Merges: https://huggingface.co/gpt2/resolve/main/merges.txt