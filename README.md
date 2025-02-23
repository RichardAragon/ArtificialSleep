# Artificial Sleep

## Overview
Artificial Sleep is an adaptive self-repairing mechanism for large language models (LLMs) that prevents performance degradation over time. Inspired by biological sleep cycles, this approach enables models to detect degradation, apply structured optimizations, and reset when necessary to maintain stable output quality.

## Features
- **Perplexity-Based Degradation Detection**: Monitors model drift and triggers intervention when necessary.
- **Gradual Weight Adjustment**: Prevents overfitting drift while maintaining learned representations.
- **Attention Cleanup Mechanism**: Normalizes extreme attention weights to avoid runaway activations.
- **Memory Consolidation**: Retains successful outputs while discarding degraded responses.
- **Emergency Model Reset**: Restores model to initial state if degradation becomes extreme.

## Usage
To run the **Artificial Sleep** experiment:
```sh
python artificial_sleep.py
```

This will initiate multiple rounds of text generation, monitoring perplexity and applying self-repair mechanisms as needed.

## Example Output
```
Iteration 1
Prompt: Explain quantum mechanics
Response: Quantum mechanics is the branch of physics that deals with...
Metrics: {'repetition_ratio': 0.72, 'perplexity': 4.5}

Iteration 6
Performance degradation detected - initiating sleep cycle
...
Iteration 12
Model stability restored
```

## Research
Artificial Sleep was developed as part of Moonshot Laboratories' research into **self-repairing neural networks**. The full research paper is available.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributors
- **Richard Aragon** - Lead Researcher
- **Moonshot Laboratories** - AI Research and Development

## Acknowledgments
Special thanks to the open-source AI community for advancing research in AI resilience and self-repair mechanisms.

