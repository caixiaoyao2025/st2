"""Auto-generated function wrapper for tool: samtools_flagstat"""
from agent_connector.tool_runner import run_tool_spec, format_result

_TOOL_SPEC = {'name': 'samtools_flagstat', 'type': 'cli', 'command': 'samtools flagstat {bam_path}', 'description': 'Calculate alignment statistics from a BAM file.', 'output_control': {'intercept_large_output': True, 'max_preview_lines': 50}, 'inputs': {'bam_path': {'type': 'string', 'description': 'Absolute path to the BAM file inside /data.'}}}


def samtools_flagstat(bam_path: str) -> str:
    """Calculate alignment statistics from a BAM file."""
    return format_result(run_tool_spec(_TOOL_SPEC, {'bam_path': bam_path}))
