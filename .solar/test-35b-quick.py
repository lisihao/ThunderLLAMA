#!/usr/bin/env python3
"""35B 模型快速测试"""

import requests

URL = "http://localhost:8090/completion"

prompt = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is"

response = requests.post(URL, json={
    "prompt": prompt,
    "n_predict": 100,
    "temperature": 0.0,
    "stream": False
}, timeout=60)

if response.status_code == 200:
    data = response.json()
    content = data['content']
    print(f"✅ 35B 模型输出:")
    print(f"  {repr(content[:200])}...")
    print(f"\n完整输出 (前 500 字符):")
    print(content[:500])
else:
    print(f"❌ Error: {response.status_code}")
