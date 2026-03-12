# ThunderLLAMA 安全卸载指南

## 概述

ThunderLLAMA 现在支持安全的外部存储管理，专为 U盘/移动硬盘设计。

## 特性

### 1. 自动故障检测与降级

**启动时检测**：
- 如果磁盘损坏、无法挂载、或无权限
- 自动禁用 L3 (磁盘缓存)
- 回退到 L2 (8GB 内存缓存)
- **程序不会崩溃**

```
[ThunderChunkStorage] WARNING: Failed to open disk cache file
[ThunderChunkStorage] ⚠️  L3 disk cache DISABLED - running in memory-only mode
[ThunderChunkStorage] ℹ️  L2 cache (8GB memory) will continue to work
```

**运行时检测**：
- 如果 U盘 意外拔出
- 写入失败时自动停止使用 L3
- 保留 L2 内存缓存继续工作

---

### 2. 安全卸载机制

#### 方法 1：信号触发（推荐）

**步骤**：
1. 找到进程 PID：
   ```bash
   ps aux | grep llama-simple
   # 或者启动时查看日志：
   # [ThunderLMCache] ✓ Signal handler registered: kill -USR1 <PID>
   ```

2. 发送安全卸载信号：
   ```bash
   kill -USR1 <PID>
   ```

3. 等待确认消息：
   ```
   [ThunderChunkStorage] ⚠️  Received unmount request - preparing for external storage removal...
   [ThunderChunkStorage] ✓ Stopped accepting new writes to L3
   [ThunderChunkStorage] Syncing all pending data to disk...
   [ThunderChunkStorage] ✓ All data synced to disk
   [ThunderChunkStorage] ✓ Disk file unmapped
   [ThunderChunkStorage] ✓ Disk file closed
   [ThunderChunkStorage] ✅ Safe unmount complete - external storage can be safely removed
   [ThunderChunkStorage] ℹ️  L2 cache (8GB memory) will continue to work
   ```

4. 现在可以安全拔出 U盘/硬盘

**效果**：
- 立即停止新写入
- 同步所有数据到磁盘
- 关闭文件句柄
- 程序继续运行（使用内存缓存）

---

#### 方法 2：程序化调用

如果你在编写脚本或工具，可以：

```c
#include "thunder-lmcache-control.h"

// 在卸载前调用
thunder_lmcache_safe_unmount();
```

---

### 3. 检查 L3 状态

```c
#include "thunder-lmcache-control.h"

if (thunder_lmcache_is_l3_enabled()) {
    printf("L3 disk cache is operational\n");
} else {
    printf("L3 disabled, using memory-only mode\n");
}
```

---

## 使用场景

### 场景 1：日常使用（自动配置）

```bash
cd /Users/lisihao/ThunderLLAMA

# 自动检测 toshiba
source setup_cache_env.sh

# 启用缓存
export THUNDER_LMCACHE=1

# 正常运行
build/bin/llama-simple -m model.gguf -n 10 "your prompt"
```

**程序会自动**：
- 注册信号处理器
- 检测磁盘健康
- 如果磁盘不可用，降级到内存模式

---

### 场景 2：需要拔出 U盘

**步骤**：
1. 程序正在运行
2. 查看日志找到 PID：
   ```
   [ThunderLMCache] ✓ Signal handler registered: kill -USR1 12345
   ```
3. 发送卸载信号：
   ```bash
   kill -USR1 12345
   ```
4. 等待 "Safe unmount complete" 消息
5. 拔出 U盘
6. 程序继续运行（使用内存缓存）

---

### 场景 3：磁盘损坏/无法访问

**自动处理**：
```
[ThunderChunkStorage] WARNING: Disk health check failed
[ThunderChunkStorage] ⚠️  L3 disk cache DISABLED - running in memory-only mode
[ThunderChunkStorage] ℹ️  L2 cache (8GB memory) will continue to work
```

**结果**：
- ✅ 程序不崩溃
- ✅ L2 (8GB 内存) 继续缓存
- ✅ 性能略有下降（无跨进程持久化）

---

## 技术细节

### 信号处理

- **SIGUSR1**：触发安全卸载
- **线程安全**：使用内部锁保护
- **非阻塞**：信号处理器快速返回

### 降级策略

| 故障类型 | 行为 | L2 (内存) | L3 (磁盘) | 跨进程 |
|----------|------|-----------|-----------|--------|
| 磁盘正常 | 正常 | ✅ 8GB | ✅ 256GB | ✅ 支持 |
| 磁盘损坏 | 降级 | ✅ 8GB | ❌ 禁用 | ❌ 禁用 |
| 磁盘拔出 | 降级 | ✅ 8GB | ❌ 禁用 | ❌ 禁用 |

### 数据安全

**正常退出**：
- ✅ 所有数据同步到磁盘
- ✅ 下次启动可恢复

**异常崩溃**：
- ❌ 未同步的数据丢失
- ✅ 程序重启后自动重建缓存
- ℹ️  缓存是优化，丢失不影响正确性

**安全卸载**：
- ✅ 强制同步所有数据
- ✅ 关闭文件句柄
- ✅ 可以安全拔出硬盘

---

## 故障排查

### 问题：信号不起作用

**原因**：进程可能已经结束或 PID 错误

**解决**：
```bash
# 确认进程还在运行
ps -p <PID>

# 检查信号是否正确
kill -USR1 <PID>  # 不是 -SIGUSR1
```

---

### 问题：磁盘不可用但没有降级

**检查**：
```bash
# 查看启动日志
[ThunderChunkStorage] ✓ L3 disk cache initialized successfully
```

如果没有看到 "DISABLED"，说明磁盘初始化成功了。

---

### 问题：性能下降

**如果 L3 被禁用**：
- 使用 `thunder_lmcache_is_l3_enabled()` 检查
- 重启程序并修复磁盘问题

---

## 最佳实践

1. **启动时检查日志**：
   - 确认 "L3 disk cache initialized successfully"
   - 记录 PID 以备卸载

2. **拔出 U盘前**：
   - 发送 `kill -USR1 <PID>`
   - 等待 "Safe unmount complete"
   - 再拔出硬盘

3. **长期运行**：
   - 定期检查 L3 状态
   - 如果降级，考虑重启程序

4. **备份重要数据**：
   - 缓存会在异常时丢失
   - 不要依赖缓存存储重要数据

---

## API 参考

```c
// thunder-lmcache-control.h

// 注册信号处理器（启动时自动调用）
void thunder_lmcache_register_signals();

// 手动触发安全卸载
void thunder_lmcache_safe_unmount();

// 检查 L3 是否启用
int thunder_lmcache_is_l3_enabled();
```

---

## 总结

✅ **磁盘健康检查**：启动时自动检测，失败时降级
✅ **运行时保护**：写入失败时自动停止使用 L3
✅ **安全卸载**：`kill -USR1 <PID>` 触发，同步后可拔出
✅ **优雅降级**：L3 失败时 L2 继续工作，程序不崩溃
✅ **零干扰**：所有功能自动启用，无需手动配置

---

*最后更新：2026-03-12*
