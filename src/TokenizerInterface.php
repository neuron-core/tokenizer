<?php

declare(strict_types=1);

namespace NeuronAI\LLM\Tokenizer;

interface TokenizerInterface
{
    public function encode(string $text): array;

    public function decode(array $tokens): string;

    public function getVocabularySize(): int;

    public function getTokenId(string $token): ?int;

    public function getToken(int $tokenId): ?string;

    public function getBosTokenId(): int;

    public function getEosTokenId(): int;

    /**
     * @throws \InvalidArgumentException
     */
    public function loadFrom(string $vocabularyFile, ?string $mergesFile = null): void;

    /**
     * @throws \InvalidArgumentException
     */
    public function saveTo(string $vocabularyFile, ?string $mergesFile = null): void;
}
