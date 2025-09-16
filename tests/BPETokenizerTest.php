<?php

declare(strict_types=1);

namespace NeuronAI\LLM\Tests\Tokenizer;

use NeuronAI\LLM\Tokenizer\BPETokenizer;
use PHPUnit\Framework\TestCase;

class BPETokenizerTest extends TestCase
{
    private string $vocabularyFilePath;
    private string $mergesFilePath;

    protected function setUp(): void
    {
        $this->vocabularyFilePath = \sys_get_temp_dir() . '/test_vocabulary_bpe_tokenizer.json';
        if (\file_exists($this->vocabularyFilePath)) {
            \unlink($this->vocabularyFilePath);
        }

        $this->mergesFilePath = \sys_get_temp_dir() . '/test_merges_bpe_tokenizer.json';
        if (\file_exists($this->mergesFilePath)) {
            \unlink($this->mergesFilePath);
        }
    }

    protected function tearDown(): void
    {
        if (\file_exists($this->vocabularyFilePath)) {
            \unlink($this->vocabularyFilePath);
        }

        if (\file_exists($this->mergesFilePath)) {
            \unlink($this->mergesFilePath);
        }
    }

    public function testIsNotInitializedOnConstruction(): void
    {
        $tokenizer = new BPETokenizer();
        $this->assertFalse($tokenizer->isInitialized());
        $this->assertEquals(4, $tokenizer->getVocabularySize()); // Only special tokens
    }

    /**
     * @throws \JsonException
     */
    public function testSaveToFileRequiresInitialization(): void
    {
        $tokenizer = new BPETokenizer();

        $this->expectException(\RuntimeException::class);
        $this->expectExceptionMessage('Tokenizer must be loaded from a pre-trained model before saving');

        $tokenizer->saveTo($this->vocabularyFilePath, $this->mergesFilePath);
    }


    public function testLoadFromFile(): void
    {
        // Create a pre-trained tokenizer file
        $tokenizerData = [
            '<|pad|>' => 0,
            '<|startoftext|>' => 1,
            '<|endoftext|>' => 2,
            '<|unknown|>' => 3,
            'h' => 4,
            'e' => 5,
            'l' => 6,
            'o' => 7,
            ' ' => 8,
            'w' => 9,
            'r' => 10,
            'd' => 11
        ];

        \file_put_contents($this->vocabularyFilePath, \json_encode($tokenizerData));
        \file_put_contents($this->mergesFilePath, \json_encode([]));

        // Create a new tokenizer and load from a file
        $loadedTokenizer = new BPETokenizer();

        $loadedTokenizer->loadFrom($this->vocabularyFilePath, $this->mergesFilePath);

        $this->assertEquals(12, $loadedTokenizer->getVocabularySize());
    }

    public function testLoadFromNonexistentFile(): void
    {
        $tokenizer = new BPETokenizer();

        $this->expectException(\InvalidArgumentException::class);

        $tokenizer->loadFrom(__DIR__.'/file.json', __DIR__.'/file.json');
    }

    public function testEncodeDecodeConsistencyWithBasicText(): void
    {
        $tokenizer = $this->createInitializedTokenizer();
        $text = "hello world";

        $tokens = $tokenizer->encode($text);
        $decoded = $tokenizer->decode($tokens);

        $this->assertNotEmpty($tokens);
        $this->assertEquals($text, $decoded);
    }

    public function testEncodeDecodeConsistencyWithComplexText(): void
    {
        $tokenizer = $this->createInitializedTokenizer();
        $text = "The quick brown fox jumps over the lazy dog! It's amazing.";

        $tokens = $tokenizer->encode($text);
        $decoded = $tokenizer->decode($tokens);

        $this->assertNotEmpty($tokens);
        $this->assertEquals($text, $decoded);
    }

    public function testEncodeDecodeConsistencyWithEmptyText(): void
    {
        $tokenizer = $this->createInitializedTokenizer();
        $text = "";

        $tokens = $tokenizer->encode($text);
        $decoded = $tokenizer->decode($tokens);

        $this->assertEmpty($tokens);
        $this->assertEquals($text, $decoded);
    }

    private function createInitializedTokenizer(): BPETokenizer
    {
        $tokenizerData = [
            '<|pad|>' => 0,
            '<|startoftext|>' => 1,
            '<|endoftext|>' => 2,
            '<|unknown|>' => 3,
            'h' => 4,
            'e' => 5,
            'l' => 6,
            'o' => 7,
            ' ' => 8,
            'w' => 9,
            'r' => 10,
            'd' => 11,
            'T' => 12,
            'q' => 13,
            'u' => 14,
            'i' => 15,
            'c' => 16,
            'k' => 17,
            'b' => 18,
            'n' => 19,
            'f' => 20,
            'x' => 21,
            'j' => 22,
            'm' => 23,
            'p' => 24,
            's' => 25,
            'v' => 26,
            't' => 27,
            'a' => 28,
            'z' => 29,
            'y' => 30,
            'g' => 31,
            '!' => 32,
            'I' => 33,
            "'" => 34,
            '.' => 35
        ];

        \file_put_contents($this->vocabularyFilePath, \json_encode($tokenizerData));
        \file_put_contents($this->mergesFilePath, \json_encode([]));

        $tokenizer = new BPETokenizer();
        $tokenizer->loadFrom($this->vocabularyFilePath, $this->mergesFilePath);

        return $tokenizer;
    }
}
