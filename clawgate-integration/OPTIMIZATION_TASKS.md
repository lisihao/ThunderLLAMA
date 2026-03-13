# ThunderLLAMA + ClawGate 优化任务列表

> **创建时间**: 2026-03-12
> **当前进度**: Phase 2.1 部分完成（Task 2.1.1, 2.1.2 ✅）

---

## 📊 Phase 2: 监控与优化（剩余任务）

### ✅ Task 2.1.1: ClawGate Prometheus Metrics（已完成）
- metrics.py (5 metric types)
- metrics_server.py (port 9090)
- Integrated into context_optimizer.py

### ✅ Task 2.1.2: ThunderLLAMA Real Stats（已完成）
- llama_get_lmcache_stats() API
- /lmcache/stats endpoint with real data
- Statistics tracking in llama_context

---

### ⏳ Task 2.1.3: Prometheus Scraping 配置

**预估时间**: 30 分钟
**依赖**: Task 2.1.1, 2.1.2
**优先级**: 🔴 高（立即执行）

**实现步骤**:
1. 安装 Prometheus:
   ```bash
   brew install prometheus
   ```

2. 创建配置文件:
   ```bash
   mkdir -p ~/.openclaw/monitoring
   vim ~/.openclaw/monitoring/prometheus.yml
   ```

3. 配置内容:
   ```yaml
   global:
     scrape_interval: 15s
     evaluation_interval: 15s

   scrape_configs:
     - job_name: 'clawgate'
       static_configs:
         - targets: ['localhost:9090']
       metrics_path: '/metrics'

     - job_name: 'thunderllama'
       static_configs:
         - targets: ['localhost:30000']
       metrics_path: '/lmcache/stats'
       metric_relabel_configs:
         - source_labels: [__name__]
           regex: '(total_prefills|skip_count|skip_rate|l2_hit_rate)'
           action: keep
   ```

4. 启动 Prometheus:
   ```bash
   prometheus --config.file=~/.openclaw/monitoring/prometheus.yml \
              --storage.tsdb.path=~/.openclaw/monitoring/data
   ```

5. 验证数据采集:
   ```bash
   curl http://localhost:9090/api/v1/query?query=clawgate_force_prefill_total
   ```

**验收标准**:
- [ ] Prometheus 成功启动
- [ ] 两个 job 都是 UP 状态
- [ ] 可以查询到 clawgate_* 和 thunderllama_* 指标
- [ ] 数据每 15 秒更新一次

**预期收益**:
- 历史数据持久化
- 支持 PromQL 查询
- 为 Grafana 提供数据源

---

### ⏳ Task 2.1.4: Grafana Dashboard

**预估时间**: 2 小时
**依赖**: Task 2.1.3
**优先级**: 🔴 高（本周完成）

**实现步骤**:
1. 安装 Grafana:
   ```bash
   brew install grafana
   brew services start grafana
   # 访问 http://localhost:3000 (admin/admin)
   ```

2. 添加 Prometheus 数据源:
   - URL: `http://localhost:9090`
   - Access: Browser

3. 创建 Dashboard（5 个 Panel）:

   **Panel 1: Performance Trends**
   - Metric: `histogram_quantile(0.95, clawgate_request_duration_seconds_bucket)`
   - Type: Time series
   - Legend: P50, P95, P99

   **Panel 2: Decision Distribution**
   - Metrics:
     - `rate(clawgate_force_prefill_total[5m])`
     - `rate(clawgate_allow_cache_total[5m])`
   - Type: Stacked bar chart
   - Legend: Force Prefill, Allow Cache

   **Panel 3: LMCache Hit Rate**
   - Metric: `l2_hit_rate`
   - Type: Gauge
   - Thresholds: 0-0.5 (red), 0.5-0.8 (yellow), 0.8-1.0 (green)

   **Panel 4: Skip Logic Effectiveness**
   - Metrics:
     - `skip_count`
     - `total_prefills`
     - `skip_rate`
   - Type: Time series + Stat

   **Panel 5: Prefix Overlap Distribution**
   - Metric: `clawgate_prefix_overlap_bucket`
   - Type: Heatmap
   - X-axis: Time, Y-axis: Overlap ratio

4. 导出 Dashboard JSON:
   ```bash
   # 保存到 clawgate-integration/grafana-dashboard.json
   ```

**验收标准**:
- [ ] 5 个 Panel 全部显示数据
- [ ] 数据实时更新（15s 刷新）
- [ ] Dashboard JSON 已导出
- [ ] 截图保存到文档

**预期收益**:
- 可视化监控
- 快速发现性能瓶颈
- 支持决策阈值调优

---

### ⏳ Task 2.2: 决策阈值调优

**预估时间**: 1 天
**依赖**: Task 2.1.4
**优先级**: 🟡 中（下周）

**实现步骤**:
1. 收集基准数据（1000+ 请求）:
   ```python
   # benchmark_threshold_tuning.py
   import pandas as pd

   data = []
   for overlap in [0.3, 0.5, 0.7, 0.9]:
       for hit_rate in [0.5, 0.7, 0.9]:
           result = run_benchmark(overlap, hit_rate)
           data.append({
               'overlap': overlap,
               'hit_rate': hit_rate,
               'decision': result.decision,
               'latency': result.latency,
               'skip_triggered': result.skip_triggered
           })

   df = pd.DataFrame(data)
   df.to_csv('threshold_tuning_data.csv')
   ```

2. 分析最优阈值:
   ```python
   # 找到最低延迟的阈值组合
   optimal = df.groupby(['overlap', 'hit_rate'])['latency'].mean().idxmin()
   print(f"Optimal thresholds: overlap={optimal[0]}, hit_rate={optimal[1]}")
   ```

3. A/B 测试:
   ```python
   # 当前阈值: overlap=0.7, hit_rate=0.9
   current = run_load_test(overlap=0.7, hit_rate=0.9)

   # 新阈值: 基于数据分析
   new = run_load_test(overlap=optimal[0], hit_rate=optimal[1])

   improvement = (current.latency - new.latency) / current.latency
   print(f"Improvement: {improvement:.1%}")
   ```

4. 更新配置:
   ```python
   # context_optimizer.py
   OVERLAP_THRESHOLD = 0.65  # 从 0.7 降低
   HIT_RATE_THRESHOLD = 0.85  # 从 0.9 降低
   ```

**验收标准**:
- [ ] 收集 1000+ 数据点
- [ ] 生成阈值优化报告（CSV + 图表）
- [ ] A/B 测试显示性能提升
- [ ] 配置已更新并测试通过

**预期收益**:
- Skip 触发率提升 10-20%
- 平均延迟降低 5-10%

---

### ⏳ Task 2.3: ThunderChunkStorage 真实统计

**预估时间**: 1 天
**依赖**: Task 2.1.2
**优先级**: 🟡 中（下周）

**实现步骤**:
1. 添加统计字段到 `ThunderChunkStorage`:
   ```cpp
   // thunder-chunk-storage.h
   class ThunderChunkStorage {
   private:
       std::atomic<uint64_t> total_chunks{0};
       std::atomic<uint64_t> total_gets{0};
       std::atomic<uint64_t> total_hits{0};
       size_t current_usage_bytes = 0;

   public:
       uint64_t get_total_chunks() const { return total_chunks; }
       uint64_t get_usage_bytes() const { return current_usage_bytes; }
       double get_hit_rate() const {
           if (total_gets == 0) return 0.0;
           return (double)total_hits / (double)total_gets;
       }
   };
   ```

2. 在 `get()` 和 `put()` 中更新统计:
   ```cpp
   thunder_kv_chunk* ThunderChunkStorage::get(const thunder_kv_chunk_key& key) {
       total_gets++;
       auto it = cache.find(key);
       if (it != cache.end()) {
           total_hits++;
           return it->second;
       }
       return nullptr;
   }

   void ThunderChunkStorage::put(const thunder_kv_chunk_key& key, thunder_kv_chunk* chunk) {
       total_chunks++;
       current_usage_bytes += chunk->k_size + chunk->v_size;
       cache[key] = chunk;
   }
   ```

3. 添加 API 函数到 `llama.h`:
   ```cpp
   LLAMA_API void llama_get_chunk_storage_stats(
       const struct llama_context * ctx,
       uint64_t * total_chunks,
       uint64_t * usage_bytes,
       double * hit_rate
   );
   ```

4. 实现 API 函数:
   ```cpp
   // llama.cpp
   void llama_get_chunk_storage_stats(
       const struct llama_context * ctx,
       uint64_t * total_chunks,
       uint64_t * usage_bytes,
       double * hit_rate
   ) {
       if (!ctx || !ctx->lmcache_storage) {
           if (total_chunks) *total_chunks = 0;
           if (usage_bytes) *usage_bytes = 0;
           if (hit_rate) *hit_rate = 0.0;
           return;
       }

       if (total_chunks) *total_chunks = ctx->lmcache_storage->get_total_chunks();
       if (usage_bytes) *usage_bytes = ctx->lmcache_storage->get_usage_bytes();
       if (hit_rate) *hit_rate = ctx->lmcache_storage->get_hit_rate();
   }
   ```

5. 更新 `/lmcache/stats` 端点:
   ```cpp
   // server-context.cpp
   uint64_t l2_chunks = 0, l2_bytes = 0;
   double storage_hit_rate = 0.0;
   llama_get_chunk_storage_stats(ctx, &l2_chunks, &l2_bytes, &storage_hit_rate);

   res->ok({
       {"total_prefills", total_prefills},
       {"skip_count", skip_count},
       {"skip_rate", skip_rate},
       {"l2_hit_rate", storage_hit_rate},  // 真实值
       {"l2_chunks", l2_chunks},           // 真实值
       {"l2_usage_bytes", l2_bytes}        // 真实值
   });
   ```

**验收标准**:
- [ ] `get_total_chunks()` 返回正确值
- [ ] `get_usage_bytes()` 与磁盘文件大小一致
- [ ] `get_hit_rate()` 在 0-1 范围内
- [ ] `/lmcache/stats` 显示真实数据
- [ ] 单元测试覆盖所有 API

**预期收益**:
- 准确的缓存使用情况
- 真实的命中率监控
- 支持容量规划

---

## ⚡ 性能优化任务

### ⏳ Task 3.1: Skip Logic 覆盖率提升

**预估时间**: 2 天
**依赖**: Task 2.3
**优先级**: 🟡 中（2 周后）

**当前问题**:
- Skip 只在 `is_prefill=1 AND chunks_found == chunks_needed` 时触发
- 实际触发率 < 5%

**实现步骤**:
1. 支持部分命中时的智能跳过:
   ```cpp
   // llama-context.cpp
   const double PARTIAL_SKIP_THRESHOLD = 0.9;  // 90% 命中即可

   if (is_prefill && lmcache_chunks_needed > 0) {
       double hit_ratio = (double)lmcache_chunks_found / (double)lmcache_chunks_needed;

       if (hit_ratio >= 1.0) {
           // 100% 命中：完全跳过
           lmcache_can_skip_compute = true;
       } else if (hit_ratio >= PARTIAL_SKIP_THRESHOLD) {
           // 90%+ 命中：部分跳过
           lmcache_partial_skip = true;
           lmcache_skip_layers_start = 0;
           lmcache_skip_layers_end = (int)(n_layer * hit_ratio);
       }
   }
   ```

2. 实现分层跳过:
   ```cpp
   // 只计算缺失的层
   for (int il = 0; il < n_layer; il++) {
       if (lmcache_partial_skip &&
           il >= lmcache_skip_layers_start &&
           il < lmcache_skip_layers_end) {
           // 跳过这一层的计算
           continue;
       }
       // 正常计算
       compute_layer(il);
   }
   ```

3. 添加统计指标:
   ```cpp
   mutable uint64_t lmcache_full_skip_count = 0;    // 100% skip
   mutable uint64_t lmcache_partial_skip_count = 0; // 90%+ skip
   ```

4. 性能测试:
   ```python
   # 测试不同命中率的加速比
   for hit_ratio in [0.5, 0.7, 0.9, 0.95, 1.0]:
       result = benchmark_with_hit_ratio(hit_ratio)
       print(f"Hit {hit_ratio:.0%}: {result.speedup:.2f}x")
   ```

**验收标准**:
- [ ] 90% 命中时可触发部分跳过
- [ ] 部分跳过时有 5-10x 加速
- [ ] Full skip 保持 27.93x 性能
- [ ] 新指标 `partial_skip_count` 正常统计
- [ ] 性能测试报告生成

**预期收益**:
- Skip 触发率: 5% → 30%
- 平均延迟降低: 15-25%
- 吞吐量提升: 20-30%

---

### ⏳ Task 3.2: Hybrid Hashing 优化（Rolling Hash）

**预估时间**: 1 天
**依赖**: 无
**优先级**: 🟢 低（1 个月后）

**当前问题**:
- Content hash 计算开销: ~2ms per chunk
- 使用 SHA256 全量哈希

**实现步骤**:
1. 实现 Rolling Hash:
   ```cpp
   // thunder-chunk-hasher.h
   class RollingHasher {
   private:
       static constexpr uint64_t PRIME = 1000000007;
       static constexpr uint64_t BASE = 31;

       uint64_t hash_value = 0;
       int window_size = 0;

   public:
       void add_token(llama_token token) {
           hash_value = (hash_value * BASE + token) % PRIME;
           window_size++;
       }

       uint64_t get_hash() const { return hash_value; }
   };
   ```

2. 集成到 `ThunderChunkHasher`:
   ```cpp
   thunder_kv_chunk_key ThunderChunkHasher::make_key(
       const llama_token* tokens,
       int n_tokens,
       size_t chunk_start,
       int layer_idx
   ) {
       RollingHasher hasher;
       for (size_t i = chunk_start; i < chunk_start + CHUNK_SIZE && i < n_tokens; i++) {
           hasher.add_token(tokens[i]);
       }

       return {
           .content_hash = hasher.get_hash(),
           .layer_idx = layer_idx,
           .chunk_start = chunk_start
       };
   }
   ```

3. 性能对比测试:
   ```cpp
   // benchmark_hashing.cpp
   auto start = std::chrono::high_resolution_clock::now();

   // SHA256 (current)
   for (int i = 0; i < 1000; i++) {
       sha256_hash(tokens, n_tokens);
   }
   auto sha256_time = duration(start);

   // Rolling Hash (new)
   start = now();
   for (int i = 0; i < 1000; i++) {
       rolling_hash(tokens, n_tokens);
   }
   auto rolling_time = duration(start);

   printf("Speedup: %.2fx\n", sha256_time / rolling_time);
   ```

**验收标准**:
- [ ] Rolling Hash 实现正确
- [ ] 哈希冲突率 < 0.1%
- [ ] 哈希计算时间减少 50%+
- [ ] 端到端性能无回退
- [ ] 单元测试通过

**预期收益**:
- Hashing 开销: 2ms → 1ms
- Prefill 延迟降低: 5-10%

---

### ⏳ Task 3.3: Decode 阶段缓存优化

**预估时间**: 3-5 天
**依赖**: Task 3.1
**优先级**: 🟢 低（2 个月后，研究性质）

**当前问题**:
- Decode 阶段无法跳过（架构限制）
- 每次都要计算新 token 的 Q/K/V

**研究方向**:

**方向 1: Prefix KV Sharing**
```cpp
// 跨请求共享相同 prefix 的 KV cache
struct SharedPrefixCache {
    std::unordered_map<uint64_t, kv_cache_ptr> prefix_cache;

    kv_cache_ptr get_or_create(const llama_token* prefix, int len) {
        uint64_t hash = compute_hash(prefix, len);
        if (prefix_cache.count(hash)) {
            return prefix_cache[hash];  // 共享
        }
        auto kv = create_kv_cache(prefix, len);
        prefix_cache[hash] = kv;
        return kv;
    }
};
```

**方向 2: Attention Pattern Caching**
```cpp
// 缓存稀疏化的 attention patterns
// 仅适用于 position 0-N 的固定 pattern
if (is_repeated_prefix(tokens, n_tokens)) {
    // 使用缓存的 attention weights
    cached_attention = get_cached_pattern(prefix_hash);
    // 只计算新 token 的 attention
    new_attention = compute_attention(new_token, cached_attention);
}
```

**方向 3: Speculative Decoding**
```cpp
// 使用小模型预测，大模型验证
// 相同 prefix 时，预测准确率更高
small_model_predictions = speculative_decode(prefix, n_predict=5);
verified_tokens = verify_with_large_model(small_model_predictions);
```

**验收标准**:
- [ ] 完成可行性研究报告
- [ ] 原型实现至少一个方向
- [ ] 性能测试显示 5%+ 提升
- [ ] 代码 review 通过

**预期收益**:
- Decode 阶段: 10-20% 加速
- 总体吞吐: 5-10% 提升

---

## 🏗️ 架构优化任务

### ⏳ Task 4.1: Cache-Aware Batching

**预估时间**: 3 天
**依赖**: Task 2.2
**优先级**: 🟡 中（3 周后）

**目标**: 将相似 prefix 的请求分组批处理

**实现步骤**:
1. 添加请求队列管理器:
   ```python
   # clawgate-integration/request_batcher.py
   from sklearn.cluster import DBSCAN
   import numpy as np

   class RequestBatcher:
       def __init__(self, threshold=0.8):
           self.threshold = threshold
           self.pending_requests = []

       def add_request(self, request):
           self.pending_requests.append(request)

       def batch_by_similarity(self):
           if len(self.pending_requests) < 2:
               return [self.pending_requests]

           # 计算 prefix 相似度矩阵
           embeddings = [req.get_prefix_embedding() for req in self.pending_requests]
           similarity_matrix = compute_cosine_similarity(embeddings)

           # 聚类
           clustering = DBSCAN(eps=1-self.threshold, metric='precomputed')
           labels = clustering.fit_predict(1 - similarity_matrix)

           # 分组
           groups = {}
           for i, label in enumerate(labels):
               if label not in groups:
                   groups[label] = []
               groups[label].append(self.pending_requests[i])

           return list(groups.values())
   ```

2. 集成到 ClawGate:
   ```python
   # context_optimizer.py
   from request_batcher import RequestBatcher

   class ContextOptimizer:
       def __init__(self):
           self.batcher = RequestBatcher(threshold=0.8)

       async def process_batch(self, requests):
           groups = self.batcher.batch_by_similarity()

           for group in groups:
               # 第一个请求建立缓存
               await self.process_request(group[0], cache_prompt=True)

               # 后续请求触发 skip
               for req in group[1:]:
                   await self.process_request(req, cache_prompt=False)
   ```

3. 性能测试:
   ```python
   # benchmark_batching.py
   # 测试场景：10 个请求，5 个 agent，每个 agent 2 个请求
   requests = generate_multi_agent_requests(n_agents=5, requests_per_agent=2)

   # Without batching
   baseline = await process_sequential(requests)

   # With batching
   optimized = await process_with_batching(requests)

   improvement = (baseline.latency - optimized.latency) / baseline.latency
   print(f"Batching improvement: {improvement:.1%}")
   ```

**验收标准**:
- [ ] RequestBatcher 单元测试通过
- [ ] Multi-agent 场景吞吐提升 30%+
- [ ] 延迟不增加（或 < 5%）
- [ ] 性能测试报告生成

**预期收益**:
- Multi-agent 吞吐: +30-50%
- 缓存命中率: +10-15%

---

### ⏳ Task 4.2: Eviction-Aware Scheduling

**预估时间**: 2 天
**依赖**: Task 2.3
**优先级**: 🟢 低（1 个月后）

**目标**: LRU eviction 时通知 ClawGate 重新排序

**实现步骤**:
1. 添加 eviction 回调到 `ThunderChunkStorage`:
   ```cpp
   // thunder-chunk-storage.h
   using EvictionCallback = std::function<void(const thunder_kv_chunk_key&)>;

   class ThunderChunkStorage {
   private:
       EvictionCallback on_eviction;

   public:
       void set_eviction_callback(EvictionCallback cb) {
           on_eviction = cb;
       }

       void evict_lru() {
           auto key = lru_queue.front();
           lru_queue.pop_front();
           cache.erase(key);

           // 触发回调
           if (on_eviction) {
               on_eviction(key);
           }
       }
   };
   ```

2. 在 server 端注册回调:
   ```cpp
   // server-context.cpp
   ctx->lmcache_storage->set_eviction_callback([this](const auto& key) {
       // 通知 ClawGate
       json eviction_event = {
           {"type", "eviction"},
           {"key", {
               {"content_hash", key.content_hash},
               {"layer_idx", key.layer_idx},
               {"chunk_start", key.chunk_start}
           }},
           {"timestamp", time(nullptr)}
       };

       // HTTP POST to ClawGate
       post_to_clawgate("/events/eviction", eviction_event);
   });
   ```

3. ClawGate 处理 eviction 事件:
   ```python
   # clawgate-integration/eviction_handler.py
   from fastapi import FastAPI

   app = FastAPI()

   @app.post("/events/eviction")
   async def handle_eviction(event: dict):
       evicted_key = event['key']

       # 重新排序队列
       optimizer.reorder_queue_after_eviction(evicted_key)

       # 更新缓存预测模型
       cache_predictor.update_on_eviction(evicted_key)

       return {"status": "ok"}
   ```

4. 队列重排序逻辑:
   ```python
   def reorder_queue_after_eviction(self, evicted_key):
       # 降低依赖该 key 的请求优先级
       for req in self.pending_queue:
           if req.depends_on_cache(evicted_key):
               req.priority -= 10  # 降低优先级

       # 重新排序
       self.pending_queue.sort(key=lambda r: r.priority, reverse=True)
   ```

**验收标准**:
- [ ] Eviction 回调正确触发
- [ ] ClawGate 收到 eviction 事件
- [ ] 队列重排序逻辑测试通过
- [ ] 缓存命中率提升 5%+

**预期收益**:
- 缓存命中率: +5-10%
- 减少无效的 force_prefill 决策

---

### ⏳ Task 4.3: Multi-Level Cache Hierarchy

**预估时间**: 5 天
**依赖**: Task 2.3
**优先级**: 🟢 低（2 个月后）

**目标**: L1(内存) + L2(SSD) + L3(HDD) 三级缓存

**实现步骤**:
1. 设计缓存层级:
   ```cpp
   // multi_level_cache.h
   class MultiLevelCache {
   private:
       LRUCache<thunder_kv_chunk_key, thunder_kv_chunk*> l1_cache;  // 1GB
       DiskCache l2_cache;  // SSD, 10GB
       DiskCache l3_cache;  // HDD, 100GB

   public:
       thunder_kv_chunk* get(const thunder_kv_chunk_key& key) {
           // L1 hit
           if (auto chunk = l1_cache.get(key)) {
               return chunk;
           }

           // L2 hit
           if (auto chunk = l2_cache.get(key)) {
               l1_cache.put(key, chunk);  // 提升到 L1
               return chunk;
           }

           // L3 hit
           if (auto chunk = l3_cache.get(key)) {
               l2_cache.put(key, chunk);  // 提升到 L2
               l1_cache.put(key, chunk);  // 提升到 L1
               return chunk;
           }

           return nullptr;  // Miss
       }
   };
   ```

2. 实现 L2 SSD 缓存:
   ```cpp
   // L2: 使用 mmap 的 SSD 缓存
   class SSDCache {
   private:
       std::string cache_dir = "/Users/lisihao/.cache/thunderllama";

   public:
       thunder_kv_chunk* get(const thunder_kv_chunk_key& key) {
           std::string path = cache_dir + "/" + key_to_filename(key);
           if (!file_exists(path)) return nullptr;

           int fd = open(path.c_str(), O_RDONLY);
           void* data = mmap(nullptr, chunk_size, PROT_READ, MAP_PRIVATE, fd, 0);
           // ... 反序列化
           munmap(data, chunk_size);
           close(fd);
           return chunk;
       }
   };
   ```

3. 配置策略:
   ```yaml
   # ~/.openclaw/config.yaml
   lmcache:
     l1:
       type: memory
       size_mb: 1024
     l2:
       type: ssd
       path: /Users/lisihao/.cache/thunderllama
       size_mb: 10240
     l3:
       type: hdd
       path: /Volumes/toshiba/lmcache
       size_mb: 102400
   ```

**验收标准**:
- [ ] 三级缓存全部工作
- [ ] 缓存提升逻辑正确
- [ ] 冷启动性能提升 2x+
- [ ] 性能测试报告

**预期收益**:
- 冷启动: 2-3x 加速
- 缓存容量: 100GB+
- 命中率: +15-20%

---

### ⏳ Task 4.4: 动态 Chunk Size

**预估时间**: 2 天
**依赖**: Task 2.3
**优先级**: 🟢 低（3 个月后）

**当前问题**:
- 固定 256 tokens chunk size
- 短 prompt 浪费空间
- 长 prompt chunk 数量多

**实现步骤**:
1. 实现自适应 chunk size:
   ```cpp
   size_t adaptive_chunk_size(int prompt_length) {
       if (prompt_length < 512) {
           return 128;  // 短 prompt
       } else if (prompt_length < 2048) {
           return 256;  // 中等 prompt
       } else {
           return 512;  // 长 prompt
       }
   }
   ```

2. 更新 hashing 逻辑:
   ```cpp
   // 每个 chunk 可以有不同的 size
   struct thunder_kv_chunk_key {
       uint64_t content_hash;
       int layer_idx;
       size_t chunk_start;
       size_t chunk_size;  // 新增字段
   };
   ```

3. 性能测试:
   ```python
   for prompt_len in [256, 512, 1024, 2048, 4096]:
       # Fixed chunk size
       fixed_result = benchmark(prompt_len, chunk_size=256)

       # Adaptive chunk size
       adaptive_result = benchmark(prompt_len, chunk_size='adaptive')

       print(f"Len {prompt_len}: Fixed={fixed_result.latency}ms, "
             f"Adaptive={adaptive_result.latency}ms")
   ```

**验收标准**:
- [ ] 自适应算法实现正确
- [ ] 缓存空间利用率提升 20%+
- [ ] 性能无回退
- [ ] 单元测试通过

**预期收益**:
- 缓存空间节省: 15-25%
- Chunk 数量减少: 10-20%

---

## 📋 任务总览（按优先级）

### 🔴 高优先级（本周）
1. **Task 2.1.3**: Prometheus 配置（30 分钟）
2. **Task 2.1.4**: Grafana Dashboard（2 小时）

### 🟡 中优先级（下周）
3. **Task 2.2**: 决策阈值调优（1 天）
4. **Task 2.3**: ThunderChunkStorage 统计（1 天）

### 🟡 中优先级（2-3 周后）
5. **Task 3.1**: Skip Logic 覆盖率提升（2 天）
6. **Task 4.1**: Cache-Aware Batching（3 天）

### 🟢 低优先级（1-3 个月）
7. **Task 4.2**: Eviction-Aware Scheduling（2 天）
8. **Task 3.2**: Hybrid Hashing 优化（1 天）
9. **Task 4.3**: Multi-Level Cache（5 天）
10. **Task 3.3**: Decode 缓存优化（3-5 天，研究性质）
11. **Task 4.4**: 动态 Chunk Size（2 天）

---

## 📊 预期总收益

完成所有任务后的预期性能提升：

| 指标 | 当前 | 优化后 | 提升 |
|------|------|--------|------|
| **平均延迟** | 587ms | 350ms | **40%** |
| **Skip 触发率** | 5% | 35% | **7x** |
| **缓存命中率** | 85% | 95% | **12%** |
| **吞吐量** | 100 req/s | 150 req/s | **50%** |
| **Multi-agent 吞吐** | 50 req/s | 90 req/s | **80%** |

---

## 🎯 里程碑

- **Week 1**: Monitoring 完成（Task 2.1.3, 2.1.4）
- **Week 2**: Tuning 完成（Task 2.2, 2.3）
- **Week 4**: 性能优化 Phase 1（Task 3.1, 4.1）
- **Month 2**: 架构优化完成（Task 4.2, 4.3）
- **Month 3**: 全部任务完成

---

**创建者**: Claude Sonnet 4.5
**最后更新**: 2026-03-12
