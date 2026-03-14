#!/usr/bin/env python3
"""
ClawGate Integration Example
Demonstrates how to integrate with ThunderLLAMA KV Cache Strategy API
"""

import requests
import time
from typing import Dict, Any


class ThunderLLAMAClient:
    """Client for ThunderLLAMA KV Cache Strategy API"""

    def __init__(self, base_url: str = "http://localhost:30000"):
        self.base_url = base_url
        self.strategy_endpoint = f"{base_url}/thunder/kv-strategy"

    def get_status(self) -> Dict[str, Any]:
        """Get current strategy and metrics"""
        response = requests.get(self.strategy_endpoint)
        response.raise_for_status()
        return response.json()

    def set_strategy(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Set a new strategy"""
        payload = {
            "name": name,
            "params": params,
            "version": 1
        }
        response = requests.post(self.strategy_endpoint, json=payload)
        response.raise_for_status()
        return response.json()

    def evaluate(self) -> Dict[str, Any]:
        """Dry-run evaluation (preview decision without applying)"""
        response = requests.get(f"{self.strategy_endpoint}/evaluate")
        response.raise_for_status()
        return response.json()

    def get_available_strategies(self) -> list:
        """Get list of available strategies"""
        response = requests.get(f"{self.strategy_endpoint}/available")
        response.raise_for_status()
        return response.json()


class ClawGateController:
    """ClawGate controller for intelligent KV cache management"""

    def __init__(self, client: ThunderLLAMAClient):
        self.client = client

    def monitor_and_optimize(self):
        """
        Pattern 1: Monitor and Override
        ClawGate monitors default strategy and overrides when needed
        """
        status = self.client.get_status()
        metrics = status['metrics']
        current_level = status['current_level']

        print(f"Current level: {current_level}")
        print(f"Memory pressure: {metrics['memory_pressure']:.4f}")
        print(f"Context utilization: {metrics['ctx_utilization']:.2%}")

        # Emergency override: switch to q4_0 if memory pressure is critical
        if metrics['memory_pressure'] > 0.9:
            print("🚨 Critical memory pressure detected! Switching to q4_0...")
            result = self.client.set_strategy("fixed", {"level": "q4_0"})
            print(f"✅ Switched to q4_0 (rebuild: {result['rebuild_success']})")

        # Optimize for medium load: switch to q8_0
        elif metrics['memory_pressure'] > 0.6 and metrics['ctx_utilization'] > 0.5:
            print("⚡ Medium load detected. Switching to q8_0...")
            result = self.client.set_strategy("fixed", {"level": "q8_0"})
            print(f"✅ Switched to q8_0 (rebuild: {result['rebuild_success']})")

        # Low load: use full precision
        elif metrics['memory_pressure'] < 0.3 and metrics['ctx_utilization'] < 0.2:
            print("✨ Low load. Switching to f16...")
            result = self.client.set_strategy("fixed", {"level": "f16"})
            print(f"✅ Switched to f16 (rebuild: {result['rebuild_success']})")

    def configure_threshold_strategy(self):
        """
        Pattern 2: Configure and Forget
        Set custom threshold strategy and let ThunderLLAMA manage autonomously
        """
        print("Setting custom threshold strategy...")
        result = self.client.set_strategy("threshold", {
            "thresholds": [
                {"ctx_utilization": 0.3, "level": "f16"},
                {"ctx_utilization": 0.6, "level": "q8_0"},
                {"ctx_utilization": 0.8, "level": "q4_0"}
            ],
            "hysteresis": 0.05
        })

        print(f"✅ Threshold strategy configured")
        print(f"   Current level: {result['current_level']}")
        print(f"   Source: {result['strategy']['source']}")
        print("   Strategy will auto-adjust based on context utilization")

    def preview_and_apply(self):
        """
        Pattern 3: Full Control
        Preview decision first, then decide whether to apply
        """
        # Preview decision
        eval_result = self.client.evaluate()
        proposed_level = eval_result['decision']['level']
        reason = eval_result['decision']['reason']

        print(f"Proposed decision: {proposed_level}")
        print(f"Reason: {reason}")

        # Decide whether to apply based on business logic
        if self.should_apply(proposed_level):
            print("Applying decision...")
            result = self.client.set_strategy("fixed", {"level": proposed_level})
            print(f"✅ Applied: {result['current_level']}")
        else:
            print("❌ Decision not applied (business rule)")

    def should_apply(self, proposed_level: str) -> bool:
        """Custom business logic to decide whether to apply a decision"""
        # Example: never downgrade to q4_0 during business hours
        current_hour = time.localtime().tm_hour
        if 9 <= current_hour <= 18 and proposed_level == "q4_0":
            return False
        return True


def main():
    """Main demo"""
    client = ThunderLLAMAClient("http://localhost:30000")
    controller = ClawGateController(client)

    print("=== ClawGate Integration Demo ===\n")

    # Pattern 1: Monitor and Override
    print("--- Pattern 1: Monitor and Override ---")
    controller.monitor_and_optimize()
    print()

    # Pattern 2: Configure and Forget
    print("--- Pattern 2: Configure and Forget ---")
    controller.configure_threshold_strategy()
    print()

    # Pattern 3: Full Control
    print("--- Pattern 3: Preview and Apply ---")
    controller.preview_and_apply()
    print()

    # Show available strategies
    print("--- Available Strategies ---")
    strategies = client.get_available_strategies()
    for strategy in strategies:
        print(f"  - {strategy['name']}")
    print()

    print("✅ Demo complete!")


if __name__ == "__main__":
    main()
