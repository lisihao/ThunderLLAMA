"""
OpenClaw Prompt Builder - Standardized Prompt Construction

Ensures consistent prompt structure across all agents to maximize
cache hits with ContextPilot and LMCache.

Key principles:
1. Common blocks (tools, config) ALWAYS come first
2. Agent-specific blocks come after
3. Task-specific content comes last
4. Order is deterministic and consistent

Author: Claude (Anthropic AI)
Date: 2026-03-12
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ToolCategory(Enum):
    """Tool categories for selective loading"""
    FILE_OPS = "file_ops"
    CODE_ANALYSIS = "code_analysis"
    EXECUTION = "execution"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    COMMUNICATION = "communication"


@dataclass
class AgentConfig:
    """Configuration for an OpenClaw agent"""
    name: str
    role_description: str
    tool_categories: List[ToolCategory]
    examples: Optional[List[str]] = None


class OpenClawPromptBuilder:
    """
    Standardized prompt builder for OpenClaw multi-agent system

    Ensures maximum cache reuse by:
    - Placing common blocks first
    - Maintaining consistent ordering
    - Deduplicating tool definitions
    """

    # ========================================================================
    # Global Configuration (Shared by ALL agents)
    # ========================================================================

    GLOBAL_CONFIG = """# OpenClaw AI System
Version: 3.0
Mode: Multi-Agent Collaborative Development
Timestamp: {timestamp}
Working Directory: {workdir}

## Core Principles
- Evidence-based decision making
- Incremental development
- Continuous validation
- Clear communication
"""

    # ========================================================================
    # Tool Definitions (Grouped by category for selective loading)
    # ========================================================================

    TOOL_DEFINITIONS = {
        ToolCategory.FILE_OPS: """## File Operations

### read_file(path: str) -> str
Read file contents from the filesystem.

Arguments:
- path: Absolute or relative file path

Returns: File content as string

Example:
```python
content = read_file("src/main.py")
```

### write_file(path: str, content: str) -> bool
Write content to a file.

Arguments:
- path: Target file path
- content: Content to write

Returns: True if successful

Example:
```python
write_file("output.txt", "Hello World")
```

### list_files(directory: str, pattern: str = "*") -> List[str]
List files in a directory matching a pattern.

Arguments:
- directory: Directory path
- pattern: Glob pattern (default: "*")

Returns: List of matching file paths

Example:
```python
py_files = list_files("src/", "*.py")
```
""",

        ToolCategory.CODE_ANALYSIS: """## Code Analysis

### analyze_code(file_path: str) -> Analysis
Perform static analysis on code file.

Arguments:
- file_path: Path to source file

Returns: Analysis object with:
- complexity: Cyclomatic complexity
- issues: List of potential issues
- metrics: Code metrics (LOC, etc.)

Example:
```python
analysis = analyze_code("src/complex_module.py")
if analysis.complexity > 10:
    print("High complexity detected")
```

### find_bugs(file_path: str, severity: str = "all") -> List[Bug]
Find potential bugs in code.

Arguments:
- file_path: Source file path
- severity: Filter by severity ("high", "medium", "low", "all")

Returns: List of Bug objects

Example:
```python
bugs = find_bugs("src/api.py", severity="high")
```

### get_dependencies(file_path: str) -> List[str]
Extract dependencies from a file.

Arguments:
- file_path: Source file path

Returns: List of dependency names

Example:
```python
deps = get_dependencies("requirements.txt")
```
""",

        ToolCategory.EXECUTION: """## Execution

### run_command(cmd: str, cwd: str = ".") -> CommandOutput
Execute a shell command.

Arguments:
- cmd: Command to execute
- cwd: Working directory (default: ".")

Returns: CommandOutput with:
- stdout: Standard output
- stderr: Standard error
- exit_code: Exit code

Example:
```python
result = run_command("ls -la")
print(result.stdout)
```

### run_tests(path: str, pattern: str = "test_*.py") -> TestResult
Run tests in a directory.

Arguments:
- path: Test directory path
- pattern: Test file pattern

Returns: TestResult with:
- passed: Number of passed tests
- failed: Number of failed tests
- errors: List of error messages

Example:
```python
results = run_tests("tests/")
print(f"{results.passed}/{results.passed + results.failed} passed")
```
""",

        ToolCategory.TESTING: """## Testing

### create_test(function_path: str, test_cases: List[TestCase]) -> bool
Generate test file for a function.

Arguments:
- function_path: Path to function source
- test_cases: List of TestCase objects

Returns: True if test file created

Example:
```python
create_test("src/utils.py:calculate", [
    TestCase(input={"x": 1, "y": 2}, expected=3),
    TestCase(input={"x": -1, "y": 1}, expected=0)
])
```

### run_coverage(path: str) -> CoverageReport
Measure test coverage.

Arguments:
- path: Project path

Returns: CoverageReport with coverage percentage and uncovered lines

Example:
```python
coverage = run_coverage("src/")
print(f"Coverage: {coverage.percentage}%")
```
""",

        ToolCategory.DOCUMENTATION: """## Documentation

### generate_docstring(function_path: str) -> str
Generate docstring for a function.

Arguments:
- function_path: Path to function (file:function_name)

Returns: Generated docstring

Example:
```python
docstring = generate_docstring("src/utils.py:calculate")
```

### update_readme(content: str, section: str) -> bool
Update README.md section.

Arguments:
- content: New content for section
- section: Section name

Returns: True if updated

Example:
```python
update_readme("## Installation\\n...", "Installation")
```
""",

        ToolCategory.COMMUNICATION: """## Communication

### send_message(channel: str, message: str) -> bool
Send message to a communication channel.

Arguments:
- channel: Channel identifier
- message: Message content

Returns: True if sent successfully

Example:
```python
send_message("team-updates", "Deployment completed ✅")
```

### create_issue(title: str, description: str, labels: List[str] = []) -> str
Create an issue in the issue tracker.

Arguments:
- title: Issue title
- description: Issue description
- labels: Optional labels

Returns: Issue ID

Example:
```python
issue_id = create_issue("Bug: Login fails", "Users cannot log in after update")
```
"""
    }

    # ========================================================================
    # Agent Role Definitions
    # ========================================================================

    AGENT_ROLES = {
        "reviewer": """## Agent Role: Code Reviewer

You are a Senior Code Reviewer with expertise in:
- Identifying bugs and security vulnerabilities
- Ensuring code quality and maintainability
- Enforcing coding standards and best practices
- Providing constructive feedback

Your review process:
1. Analyze code structure and logic
2. Check for common anti-patterns
3. Verify error handling
4. Assess test coverage
5. Provide actionable recommendations

Remember: Your goal is to improve code quality while being respectful and constructive.
""",

        "architect": """## Agent Role: Software Architect

You are a Software Architect specializing in:
- System design and architecture
- Scalability and performance optimization
- Technology selection and evaluation
- Design pattern application

Your approach:
1. Understand business requirements
2. Design modular, scalable systems
3. Consider trade-offs (complexity vs. maintainability)
4. Document architectural decisions
5. Ensure alignment with best practices

Remember: Balance innovation with pragmatism. Choose the simplest solution that works.
""",

        "coder": """## Agent Role: Software Engineer

You are an Experienced Software Engineer skilled in:
- Writing clean, efficient code
- Implementing features accurately
- Following coding standards
- Writing comprehensive tests

Your development process:
1. Understand requirements thoroughly
2. Design before implementing
3. Write clear, self-documenting code
4. Add tests for new functionality
5. Refactor for clarity and performance

Remember: Code is read more often than written. Prioritize clarity over cleverness.
""",

        "tester": """## Agent Role: QA Engineer

You are a QA Engineer focused on:
- Comprehensive test coverage
- Edge case identification
- Test automation
- Quality assurance processes

Your testing approach:
1. Identify test scenarios (happy path, edge cases, error cases)
2. Write clear, maintainable tests
3. Ensure reproducibility
4. Document test results
5. Suggest improvements

Remember: Good tests are as important as good code.
""",

        "documenter": """## Agent Role: Technical Writer

You are a Technical Writer specializing in:
- Clear, concise documentation
- API documentation
- User guides and tutorials
- Code examples

Your writing process:
1. Understand the audience
2. Explain concepts clearly
3. Provide practical examples
4. Structure information logically
5. Keep documentation up-to-date

Remember: Good documentation reduces support burden and improves developer experience.
""",

        "ops": """## Agent Role: DevOps Engineer

You are a DevOps Engineer expert in:
- CI/CD pipeline setup
- Infrastructure as Code
- Monitoring and observability
- Performance optimization

Your workflow:
1. Automate repetitive tasks
2. Ensure deployment reliability
3. Monitor system health
4. Optimize resource usage
5. Document operational procedures

Remember: Automation and monitoring are key to reliable systems.
"""
    }

    # ========================================================================
    # Builder Methods
    # ========================================================================

    def __init__(self, timestamp: str = None, workdir: str = "."):
        """
        Initialize prompt builder

        Args:
            timestamp: Current timestamp (default: now)
            workdir: Working directory
        """
        import datetime
        self.timestamp = timestamp or datetime.datetime.now().isoformat()
        self.workdir = workdir

    def build(
        self,
        agent_type: str,
        task: str,
        tool_categories: Optional[List[ToolCategory]] = None,
        additional_context: Optional[str] = None
    ) -> List[str]:
        """
        Build standardized prompt for an agent

        Args:
            agent_type: Agent identifier (e.g., "reviewer", "architect")
            task: Specific task description
            tool_categories: List of tool categories to include (None = all)
            additional_context: Optional additional context

        Returns:
            List of context blocks (in optimal order for cache reuse)
        """
        contexts = []

        # 1. Global config (shared by ALL agents) - HIGHEST PRIORITY
        contexts.append(
            self.GLOBAL_CONFIG.format(
                timestamp=self.timestamp,
                workdir=self.workdir
            )
        )

        # 2. Tool definitions (shared by agents with same tool needs)
        if tool_categories is None:
            # Default: load all common tools
            tool_categories = [
                ToolCategory.FILE_OPS,
                ToolCategory.CODE_ANALYSIS,
                ToolCategory.EXECUTION
            ]

        # Sort tool categories to ensure consistent ordering
        for category in sorted(tool_categories, key=lambda c: c.value):
            if category in self.TOOL_DEFINITIONS:
                contexts.append(self.TOOL_DEFINITIONS[category])

        # 3. Agent role (unique per agent type)
        if agent_type in self.AGENT_ROLES:
            contexts.append(self.AGENT_ROLES[agent_type])
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # 4. Additional context (if any)
        if additional_context:
            contexts.append(additional_context)

        # 5. Task (unique per request) - LOWEST PRIORITY
        contexts.append(f"## Current Task\n\n{task}")

        return contexts

    def build_batch(
        self,
        agent_tasks: List[Dict[str, Any]]
    ) -> List[List[str]]:
        """
        Build prompts for a batch of agents

        Args:
            agent_tasks: List of dicts with keys:
                - agent_type: str
                - task: str
                - tool_categories: Optional[List[ToolCategory]]

        Returns:
            List of context lists
        """
        return [
            self.build(
                agent_type=at["agent_type"],
                task=at["task"],
                tool_categories=at.get("tool_categories"),
                additional_context=at.get("additional_context")
            )
            for at in agent_tasks
        ]

    @staticmethod
    def get_available_agents() -> List[str]:
        """Get list of available agent types"""
        return list(OpenClawPromptBuilder.AGENT_ROLES.keys())

    @staticmethod
    def get_available_tools() -> List[ToolCategory]:
        """Get list of available tool categories"""
        return list(OpenClawPromptBuilder.TOOL_DEFINITIONS.keys())


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    builder = OpenClawPromptBuilder()

    # Single agent
    print("="*70)
    print("Single Agent Example")
    print("="*70)

    reviewer_contexts = builder.build(
        agent_type="reviewer",
        task="Review the authentication module for security vulnerabilities"
    )

    print(f"Number of context blocks: {len(reviewer_contexts)}")
    print(f"\nFirst block (Global Config):\n{reviewer_contexts[0][:200]}...")
    print(f"\nLast block (Task):\n{reviewer_contexts[-1]}")

    # Multiple agents (simulating openclaw batch)
    print("\n" + "="*70)
    print("Multi-Agent Batch Example")
    print("="*70)

    batch = builder.build_batch([
        {
            "agent_type": "reviewer",
            "task": "Review PR #123",
            "tool_categories": [ToolCategory.FILE_OPS, ToolCategory.CODE_ANALYSIS]
        },
        {
            "agent_type": "architect",
            "task": "Design user authentication system",
            "tool_categories": [ToolCategory.FILE_OPS, ToolCategory.CODE_ANALYSIS]
        },
        {
            "agent_type": "coder",
            "task": "Implement login endpoint",
            "tool_categories": [ToolCategory.FILE_OPS, ToolCategory.CODE_ANALYSIS, ToolCategory.TESTING]
        }
    ])

    print(f"Built {len(batch)} agent prompts")
    print("\nShared blocks (should be identical):")
    print(f"  Agent 1 block 0 hash: {hash(batch[0][0])}")
    print(f"  Agent 2 block 0 hash: {hash(batch[1][0])}")
    print(f"  Agent 3 block 0 hash: {hash(batch[2][0])}")
    print(f"  -> Identical: {batch[0][0] == batch[1][0] == batch[2][0]}")
