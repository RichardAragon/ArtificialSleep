import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import logging
import gc
import torch.nn.functional as F
from collections import deque

# Logging Setup
logging.basicConfig(filename='swarm_ai_sleep.log', level=logging.INFO)

class SwarmArtificialSleep:
    def __init__(self, model_name="HuggingFaceTB/SmolLM2-135M", memory_buffer_size=100):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.memory_buffer = deque(maxlen=memory_buffer_size)
        self.swarm_agents = {name: {"perplexity": [], "status": "awake"} 
                             for name, _ in self.model.named_modules() if isinstance(_, torch.nn.Linear)}
        
        self.baseline_perplexity = 10.0
        self.degradation_threshold = 1.2  # 20% increase triggers sleep

    def calculate_perplexity(self, text):
        """Calculate perplexity of the model's output."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        labels = inputs["input_ids"].clone()
        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
        return torch.exp(loss).item()

    def check_response_quality(self, response):
        """Assess degeneration metrics like repetition and perplexity."""
        words = response.split()
        word_set = set(words)
        repetition_ratio = len(word_set) / len(words)
        perplexity = self.calculate_perplexity(response)

        return {
            'repetition_ratio': repetition_ratio,
            'perplexity': perplexity
        }

    def swarm_monitor(self, text):
        """Swarm agents assess each layer's performance."""
        metrics = self.check_response_quality(text)
        degradation_detected = False

        for name in self.swarm_agents:
            self.swarm_agents[name]["perplexity"].append(metrics["perplexity"])
            
            if len(self.swarm_agents[name]["perplexity"]) > 5:  # Track last 5 responses
                avg_perplexity = np.mean(self.swarm_agents[name]["perplexity"][-5:])
                
                if avg_perplexity > self.baseline_perplexity * self.degradation_threshold:
                    self.swarm_agents[name]["status"] = "sleeping"
                    degradation_detected = True
                else:
                    self.swarm_agents[name]["status"] = "awake"

        return degradation_detected

    def localized_sleep_cycle(self):
        """Targeted self-repair for affected layers only."""
        logging.info("Starting targeted swarm-based sleep cycle")
        torch.cuda.empty_cache()
        gc.collect()

        with torch.no_grad():
            for name, module in self.model.named_modules():
                if name in self.swarm_agents and self.swarm_agents[name]["status"] == "sleeping":
                    logging.info(f"Swarm layer {name} undergoing artificial sleep.")

                    # Normalize weights to reduce entropy
                    module.weight.data -= module.weight.data.mean()

                    # Attention-based cleanup to soften activations
                    if "attention" in name.lower():
                        attention_weights = module.weight.data
                        attention_weights = F.softmax(attention_weights / 1.05, dim=-1)
                        module.weight.data = attention_weights

                    self.swarm_agents[name]["status"] = "recovering"

        logging.info("Swarm sleep cycle completed")

    def generate_response(self, prompt):
        """Generate response and trigger swarm monitoring."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_length=200, do_sample=True, temperature=0.7, 
                top_p=0.95, top_k=50, no_repeat_ngram_size=3
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Monitor swarm status
        if self.swarm_monitor(response):
            print("Degradation detected! Swarm initiating localized artificial sleep.")
            self.localized_sleep_cycle()

        return response

    def run_experiment(self, num_iterations=20):
        """Run the swarm-based self-healing AI sleep experiment."""
        prompts = ["Explain quantum mechanics", "Tell me a story", "What is the meaning of life?"]
        for i in range(num_iterations):
            prompt = prompts[i % len(prompts)]
            response = self.generate_response(prompt)

            print(f"\nIteration {i+1}")
            print(f"Prompt: {prompt}")
            print(f"Response: {response}")
            print(f"Swarm Layer Status: { {k: v['status'] for k, v in self.swarm_agents.items()} }")

if __name__ == "__main__":
    ai_swarm_sleep = SwarmArtificialSleep()
    ai_swarm_sleep.run_experiment()
