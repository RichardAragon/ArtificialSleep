import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import logging
import gc
from collections import deque
import torch.nn.functional as F

class ArtificialSleep:
    def __init__(self, model_name="HuggingFaceTB/SmolLM2-135M", memory_buffer_size=100):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.memory_buffer = deque(maxlen=memory_buffer_size)
        self.performance_history = []  
        self.setup_logging()

        # Initialize baseline perplexity with a dummy value (to prevent NoneType errors)
        self.baseline_perplexity = 10.0  # Default value to avoid NoneType issues
        self.degradation_threshold = 1.2  # Allow 20% degradation before intervention


    def setup_logging(self):
        logging.basicConfig(
            filename='ai_sleep_experiment.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def calculate_perplexity(self, text):
        """Calculate model perplexity on input text"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        labels = inputs["input_ids"].clone()
        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss

        return torch.exp(loss).item()

    def check_response_quality(self, response):
        """Check for signs of degeneration in responses"""
        words = response.split()
        word_set = set(words)
        repetition_ratio = len(word_set) / len(words)
        ends_properly = response.strip().endswith(('.', '!', '?'))
        perplexity = self.calculate_perplexity(response)

        return {
            'repetition_ratio': repetition_ratio,
            'ends_properly': ends_properly,
            'perplexity': perplexity
        }

    def gradual_weight_adjustment(self):
        """Less aggressive weight decay to prevent over-degradation."""
        with torch.no_grad():
            for param in self.model.parameters():
                param.data.mul_(0.9999)  # Increased from 0.999 to 0.9999

    def attention_based_cleanup(self, temperature=1.2):
        """Reduce attention cleanup impact to prevent performance drift."""
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if "attention" in name.lower() and hasattr(module, 'weight'):
                    attention_weights = module.weight.data
                    attention_weights = F.softmax(attention_weights / temperature, dim=-1)
                    
                    # Blend instead of replacing weights completely
                    module.weight.data = (module.weight.data * 0.9) + (attention_weights * 0.1)

    def memory_consolidation(self, top_k=3):
        """Reinforce successful patterns while removing high-perplexity responses."""
        if not self.memory_buffer:
            return

        # Remove degraded outputs (perplexity > 3x baseline)
        self.memory_buffer = deque(
            [entry for entry in self.memory_buffer if entry['metrics']['perplexity'] < self.baseline_perplexity * 3],
            maxlen=len(self.memory_buffer)
        )

        sorted_examples = sorted(self.memory_buffer, key=lambda x: x['metrics']['perplexity'])
        good_examples = sorted_examples[:1]  # Start with the best example

        for example in sorted_examples[1:]:
            if example['text'] not in [e['text'] for e in good_examples]:
                good_examples.append(example)
            if len(good_examples) >= top_k:
                break

        for example in good_examples:
            inputs = self.tokenizer(example['text'], return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                self.model(**inputs)

    def sleep_cycle(self):
        """Improved sleep cycle focusing on gentle optimization & reset mechanism."""
        logging.info("Starting sleep cycle")
        torch.cuda.empty_cache()
        gc.collect()

        # Check if model degradation is severe
        if len(self.performance_history) > 5:
            last_perplexities = [p['perplexity'] for p in self.performance_history[-5:]]
            if max(last_perplexities) > self.baseline_perplexity * 5:
                logging.warning("Extreme degradation detected. Resetting model to initial state.")
                self.model = AutoModelForCausalLM.from_pretrained(self.model.config._name_or_path)
                return

        self.gradual_weight_adjustment()
        self.attention_based_cleanup(temperature=1.05)
        self.memory_consolidation(top_k=5)

        logging.info("Sleep cycle completed")

    def generate_response(self, prompt):
        """Generate response with quality checks."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                no_repeat_ngram_size=3
            )

        response = self.tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True)
        metrics = self.check_response_quality(response)

        # Store valid responses in memory buffer
        if metrics['perplexity'] < self.baseline_perplexity * 3:
            self.memory_buffer.append({'text': response, 'metrics': metrics})

        return response, metrics

    def run_experiment(self, num_iterations=20, sleep_interval=5):
        """Run experiment with adaptive sleep scheduling."""
        prompts = [
            "Explain quantum mechanics",
            "Tell me a story",
            "What is the meaning of life?"
        ]

        # Establish baseline performance
        baseline_prompt = "Explain a simple concept"
        _, baseline_metrics = self.generate_response(baseline_prompt)
        self.baseline_perplexity = baseline_metrics['perplexity']

        for i in range(num_iterations):
            prompt = prompts[i % len(prompts)]
            response, metrics = self.generate_response(prompt)

            print(f"\nIteration {i+1}")
            print(f"Prompt: {prompt}")
            print(f"Response: {response}")
            print(f"Metrics: {metrics}")

            if (metrics['perplexity'] > self.baseline_perplexity * self.degradation_threshold or
                    metrics['repetition_ratio'] < 0.4):
                print("Performance degradation detected - initiating sleep cycle")
                self.sleep_cycle()
            elif (i + 1) % sleep_interval == 0:
                self.sleep_cycle()

            self.performance_history.append(metrics)

        self.visualize_results()

    def visualize_results(self):
        """Visualize multiple performance metrics."""
        plt.figure(figsize=(15, 5))

        perplexities = [p['perplexity'] for p in self.performance_history]
        repetition_ratios = [p['repetition_ratio'] for p in self.performance_history]

        plt.subplot(1, 2, 1)
        plt.plot(perplexities, label='Perplexity', marker='o')
        plt.axhline(y=self.baseline_perplexity, color='r', linestyle='--', label='Baseline')
        plt.title('Perplexity Over Time')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(repetition_ratios, label='Repetition Ratio', marker='o')
        plt.title('Repetition Ratio Over Time')
        plt.legend()

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    ai_sleep = ArtificialSleep()
    ai_sleep.run_experiment()
