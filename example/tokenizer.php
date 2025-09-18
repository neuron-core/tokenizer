<?php

declare(strict_types=1);

use NeuronAI\Tokenizer\BPETokenizer;

\ini_set("memory_limit", "1024M");

require_once __DIR__ . '/../vendor/autoload.php';

function generateString($length = 150)
{
    $characters = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ';
    $charactersLength = \strlen($characters);
    $randomString = '';

    for ($i = 0; $i < $length; $i++) {
        $randomString .= $characters[\random_int(0, $charactersLength - 1)];
    }

    return $randomString;
}

$tokenizer = new BPETokenizer();
$tokenizer->loadFrom(
    __DIR__.'/../bpe-vocabulary.json',
    __DIR__.'/../bpe-merges.txt',
);


$contents = [];
for ($i = 1; $i <= 2000; $i++) {
    $contents[] = generateString();
}

$durations = [];
foreach ($contents as $i => $content) {
    $start = \microtime(true);
    $tokens = $tokenizer->encode($content);
    if ($i === 0) {
        echo "Tokens: ".implode(", ", $tokens)."\n";
    }
    $durations[] = \microtime(true) - $start;
}

$avg = \array_sum($durations) / \count($durations);
echo ($avg/1000).'ms';
