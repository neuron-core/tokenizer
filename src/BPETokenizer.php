<?php

declare(strict_types=1);

namespace NeuronAI\Tokenizer;

class BPETokenizer implements TokenizerInterface
{
    protected array $vocabulary = [];
    protected array $merges = [];
    protected array $tokenToId = [];
    protected array $idToToken = [];
    protected int $vocabularySize = 0;
    protected array $mergePriorities = [];
    protected array $preTokenizeCache = [];

    protected const BOS_TOKEN = '<|startoftext|>';
    protected const EOS_TOKEN = '<|endoftext|>';
    protected const UNK_TOKEN = '<|unknown|>';
    protected const PAD_TOKEN = '<|pad|>';

    public function __construct()
    {
        $this->initializeSpecialTokens();
    }

    private function initializeSpecialTokens(): void
    {
        $specialTokens = [
            self::PAD_TOKEN,
            self::BOS_TOKEN,
            self::EOS_TOKEN,
            self::UNK_TOKEN,
        ];

        foreach ($specialTokens as $token) {
            $this->addToken($token);
        }
    }

    private function addToken(string $token): void
    {
        if (!isset($this->tokenToId[$token])) {
            $id = $this->vocabularySize;
            $this->tokenToId[$token] = $id;
            $this->idToToken[$id] = $token;
            $this->vocabulary[] = $token;
            ++$this->vocabularySize;
        }
    }

    private function preTokenize(string $text): array
    {
        if ($text === '') {
            return [];
        }

        // Check cache first for frequently used texts
        $hash = \hash('xxh3', $text);
        if (isset($this->preTokenizeCache[$hash])) {
            return $this->preTokenizeCache[$hash];
        }

        // Tiktoken-style regex pattern for high-quality tokenization
        $pattern = "/(?:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/u";

        \preg_match_all($pattern, $text, $matches);

        // Filter out empty matches
        $result = \array_filter($matches[0], fn (string $token): bool => $token !== '');

        // Cache result with size limit
        if (\count($this->preTokenizeCache) < 1000) {
            $this->preTokenizeCache[$hash] = $result;
        }

        return $result;
    }

    private function splitWord(string $word): array
    {
        if ($word === '') {
            return [];
        }

        // Split word into characters
        $symbols = \mb_str_split($word);
        if (\count($symbols) < 2) {
            return $symbols;
        }

        // Use optimized merge algorithm with position tracking
        return $this->optimizedMerge($symbols);
    }

    private function optimizedMerge(array $symbols): array
    {
        $count = \count($symbols);
        if ($count < 2) {
            return $symbols;
        }

        // Track positions and sizes for efficient merging
        $positions = \range(0, $count - 1);
        $sizes = \array_fill(0, $count, 1);

        // Find all possible merge candidates with priorities
        $mergeCandidates = new \SplPriorityQueue();
        for ($i = 0; $i < $count - 1; ++$i) {
            $pair = $symbols[$i] . $symbols[$i + 1];
            if (isset($this->mergePriorities[$pair])) {
                // Higher priority = lower number, so negate for max-heap
                $mergeCandidates->insert(['pos' => $i, 'pair' => $pair], -$this->mergePriorities[$pair]);
            }
        }

        // Apply merges in priority order
        while (!$mergeCandidates->isEmpty()) {
            $candidate = $mergeCandidates->extract();
            $pos = $candidate['pos'];
            $pair = $candidate['pair'];

            // Check if this merge position is still valid
            if ($sizes[$pos] === 0 || $sizes[$pos + 1] === 0) {
                continue;
            }

            // Verify the pair still matches at this position
            if ($symbols[$positions[$pos]] . $symbols[$positions[$pos + 1]] !== $pair) {
                continue;
            }

            // Perform the merge
            $symbols[$positions[$pos]] = $pair;
            $sizes[$pos + 1] = 0; // Mark second position as merged

            // Add new merge candidates if possible
            if ($pos > 0 && $sizes[$pos - 1] > 0) {
                $newPair = $symbols[$positions[$pos - 1]] . $pair;
                if (isset($this->mergePriorities[$newPair])) {
                    $mergeCandidates->insert(['pos' => $pos - 1, 'pair' => $newPair], -$this->mergePriorities[$newPair]);
                }
            }
            if ($pos + 2 < $count && $sizes[$pos + 2] > 0) {
                $newPair = $pair . $symbols[$positions[$pos + 2]];
                if (isset($this->mergePriorities[$newPair])) {
                    $mergeCandidates->insert(['pos' => $pos, 'pair' => $newPair], -$this->mergePriorities[$newPair]);
                }
            }
        }

        // Extract final result, skipping merged positions
        $result = [];
        for ($i = 0; $i < $count; ++$i) {
            if ($sizes[$i] > 0) {
                $result[] = $symbols[$positions[$i]];
            }
        }

        return $result;
    }

    public function encode(string $text): array
    {
        if ($text === '') {
            return [];
        }

        $words = $this->preTokenize($text);
        $tokenIds = [];

        foreach ($words as $word) {
            $symbols = $this->splitWord($word);

            foreach ($symbols as $symbol) {
                $tokenId = $this->getTokenId($symbol);
                $tokenIds[] = $tokenId ?? $this->getTokenId(self::UNK_TOKEN);
            }
        }

        return \array_filter($tokenIds, fn (?int $id): bool => $id !== null);
    }

    public function decode(array $tokens): string
    {
        $list = [];

        foreach ($tokens as $tokenId) {
            $token = $this->getToken($tokenId);
            if ($token !== null &&
                $token !== self::BOS_TOKEN &&
                $token !== self::EOS_TOKEN &&
                $token !== self::PAD_TOKEN) {
                $list[] = $token;
            }
        }

        $text = \implode('', $list);

        // Replace GPT-2 space marker (Ġ = \u0120) with actual spaces
        return \str_replace("\u{0120}", ' ', $text);
    }

    public function getVocabularySize(): int
    {
        return $this->vocabularySize;
    }

    public function getTokenId(string $token): ?int
    {
        return $this->tokenToId[$token] ?? null;
    }

    public function getToken(int $tokenId): ?string
    {
        $token = $this->idToToken[$tokenId] ?? null;
        return $token !== null ? (string) $token : null;
    }

    public function getBosTokenId(): int
    {
        // For GPT-2 style tokenizers, use endoftext as BOS token if startoftext doesn't exist
        if (isset($this->tokenToId[self::BOS_TOKEN])) {
            return $this->tokenToId[self::BOS_TOKEN];
        }

        // Fallback to endoftext token (common in GPT-2)
        if (isset($this->tokenToId[self::EOS_TOKEN])) {
            return $this->tokenToId[self::EOS_TOKEN];
        }

        throw new \RuntimeException('No BOS token found in vocabulary');
    }

    public function getEosTokenId(): int
    {
        if (isset($this->tokenToId[self::EOS_TOKEN])) {
            return $this->tokenToId[self::EOS_TOKEN];
        }

        throw new \RuntimeException('No EOS token found in vocabulary');
    }

    /**
     * @throws \JsonException
     */
    public function saveTo(string $vocabularyFile, ?string $mergesFile = null): void
    {
        if (!$this->isInitialized()) {
            throw new \RuntimeException('Tokenizer must be loaded from a pre-trained model before saving');
        }

        $result = \file_put_contents($vocabularyFile, \json_encode($this->tokenToId, \JSON_PRETTY_PRINT | \JSON_THROW_ON_ERROR));
        if ($result === false) {
            throw new \RuntimeException("Failed to save tokenizer to: {$vocabularyFile}");
        }

        $result = \file_put_contents($mergesFile, \json_encode($this->merges, \JSON_PRETTY_PRINT | \JSON_THROW_ON_ERROR));
        if ($result === false) {
            throw new \RuntimeException("Failed to save tokenizer to: {$mergesFile}");
        }
    }

    public function loadFrom(string $vocabularyFile, ?string $mergesFile = null): void
    {
        if (!\file_exists($vocabularyFile)) {
            throw new \InvalidArgumentException("Vocabulary file not found: {$vocabularyFile}");
        }

        if (!\is_null($mergesFile) && !\file_exists($mergesFile)) {
            throw new \InvalidArgumentException("Merges file not found: {$mergesFile}");
        }

        $vocabulary = \file_get_contents($vocabularyFile);
        if ($vocabulary === false) {
            throw new \InvalidArgumentException("Failed to read vocabulary file: {$vocabularyFile}");
        }

        try {
            $data = \json_decode($vocabulary, true, 512, \JSON_THROW_ON_ERROR);
        } catch (\JsonException $e) {
            throw new \InvalidArgumentException("Invalid JSON in tokenizer file: {$e->getMessage()}", $e->getCode(), $e);
        }

        $this->tokenToId = $data;
        $this->idToToken = \array_flip($this->tokenToId);
        $this->vocabulary = \array_keys($this->tokenToId);
        $this->vocabularySize = \count($this->tokenToId);

        // Ensure we have the necessary special tokens, add them if missing
        $this->ensureSpecialTokensExist();

        if (!\is_null($mergesFile)) {
            $this->merges = $this->parseMergeRules($mergesFile);
        }
    }

    protected function parseMergeRules(string $mergesFile): array
    {
        $mergesContent = \file_get_contents($mergesFile);
        if ($mergesContent === false) {
            throw new \InvalidArgumentException("Failed to read merges file: {$mergesFile}");
        }

        $lines = \explode("\n", $mergesContent);
        $merges = [];
        $priority = 0;

        foreach ($lines as $line) {
            $line = \trim($line);
            // Skip empty lines and comments
            if ($line === '') {
                continue;
            }
            if (\str_starts_with($line, '#')) {
                continue;
            }

            $parts = \explode(' ', $line);
            if (\count($parts) === 2) {
                $first = \str_replace('Ġ', ' ', $parts[0]);
                $second = \str_replace('Ġ', ' ', $parts[1]);
                $merges[] = [$first, $second];
                // Store priority for quick lookup (lower index = higher priority)
                $this->mergePriorities[$first . $second] = $priority++;
            }
        }

        return $merges;
    }

    private function ensureSpecialTokensExist(): void
    {
        $specialTokens = [
            self::PAD_TOKEN,
            self::UNK_TOKEN,
        ];

        foreach ($specialTokens as $token) {
            if (!isset($this->tokenToId[$token])) {
                $this->addToken($token);
            }
        }

        // For BOS and EOS tokens, don't add them if they don't exist
        // The getBosTokenId() and getEosTokenId() methods will handle fallback logic
    }

    public function isInitialized(): bool
    {
        return $this->vocabularySize > 4; // More than just special tokens
    }
}
