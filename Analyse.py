import time
import tiktoken
from typing import Dict, Any

class AgentAnalytics:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Approximate pricing per 1K tokens for gpt-4o-mini
        # Input: $0.15 / 1M tokens -> $0.00015 / 1K tokens -> $0.00000015 / token
        # Output: $0.60 / 1M tokens -> $0.0006 / 1K tokens -> $0.0000006 / token
        self.price_per_input_token = 0.00000015
        self.price_per_output_token = 0.0000006

    def start_timer(self) -> float:
        return time.perf_counter()

    def end_timer(self, start_time: float) -> float:
        """Returns elapsed time in milliseconds."""
        return (time.perf_counter() - start_time) * 1000

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoding.encode(text))
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.price_per_input_token) + (output_tokens * self.price_per_output_token)

    def generate_stats(self, latency_ms: float, prompt_text: str, response_text: str) -> Dict[str, Any]:
        input_tokens = self.count_tokens(prompt_text)
        output_tokens = self.count_tokens(response_text)
        total_tokens = input_tokens + output_tokens
        cost = self.calculate_cost(input_tokens, output_tokens)
        
        return {
            "latency_ms": round(latency_ms, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost_usd": round(cost, 6)
        }

analytics_agent = AgentAnalytics()
