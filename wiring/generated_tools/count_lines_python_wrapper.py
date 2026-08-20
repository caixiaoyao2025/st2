"""Auto-generated function wrapper for tool: count_lines_python"""
from agent_connector.tool_runner import run_tool_spec, format_result

_TOOL_SPEC = {'name': 'count_lines_python', 'description': 'Count the lines and byte size of a text file using the Python execution runner.', 'output_control': {'intercept_large_output': False, 'max_preview_lines': 50}, 'execution': {'type': 'python', 'entry_point': 'tool_helpers:count_lines'}, 'inputs': {'file_path': {'type': 'string', 'description': 'Path to a text file inside /data.'}}}


def count_lines_python(file_path: str) -> str:
    """Count the lines and byte size of a text file using the Python execution runner."""
    return format_result(run_tool_spec(_TOOL_SPEC, {'file_path': file_path}))
