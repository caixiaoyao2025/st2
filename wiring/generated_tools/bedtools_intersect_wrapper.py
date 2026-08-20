"""Auto-generated function wrapper for tool: bedtools_intersect"""
from agent_connector.tool_runner import run_tool_spec, format_result

_TOOL_SPEC = {'name': 'bedtools_intersect', 'type': 'cli', 'command': 'bedtools intersect -a {a_bed_path} -b {b_bed_path} -wa -wb', 'description': 'Intersect two BED files and return overlapping records.', 'output_control': {'intercept_large_output': True, 'max_preview_lines': 100}, 'inputs': {'a_bed_path': {'type': 'string', 'description': 'Absolute path to the first BED file inside /data.'}, 'b_bed_path': {'type': 'string', 'description': 'Absolute path to the second BED file inside /data.'}}}


def bedtools_intersect(a_bed_path: str, b_bed_path: str) -> str:
    """Intersect two BED files and return overlapping records."""
    return format_result(run_tool_spec(_TOOL_SPEC, {'a_bed_path': a_bed_path, 'b_bed_path': b_bed_path}))
