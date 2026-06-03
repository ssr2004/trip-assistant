"""
任务执行器
执行规划好的任务
"""
from typing import Dict, List, Any
from tools.registry import ToolRegistry


class TaskExecutor:
    """任务执行器"""

    def __init__(self):
        """初始化执行器"""
        self.tool_registry = ToolRegistry()

    async def execute(self, tasks: List[Dict]) -> List[Dict]:
        """
        执行任务列表

        Args:
            tasks: 任务列表

        Returns:
            执行结果列表
        """
        results = []

        # 按优先级排序
        sorted_tasks = sorted(tasks, key=lambda x: x.get("priority", 999))

        for task in sorted_tasks:
            tool_name = task.get("tool")
            params = task.get("params", {})

            try:
                result = await self.tool_registry.execute(tool_name, params)
                results.append({
                    "tool": tool_name,
                    "params": params,
                    "result": result,
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "tool": tool_name,
                    "params": params,
                    "result": None,
                    "status": "error",
                    "error": str(e)
                })

        return results
