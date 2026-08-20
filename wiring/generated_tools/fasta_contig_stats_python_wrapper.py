"""Auto-generated function wrapper for tool: fasta_contig_stats_python"""
from agent_connector.tool_runner import run_tool_spec, format_result

_TOOL_SPEC = {'name': 'fasta_contig_stats_python', 'description': 'Count sequences and total bases in a FASTA file using the Python execution runner.', 'output_control': {'intercept_large_output': False, 'max_preview_lines': 50}, 'execution': {'type': 'python', 'entry_point': 'tool_helpers:fasta_contig_stats'}, 'inputs': {'fasta_path': {'type': 'string', 'description': 'Path to a FASTA file inside /data.'}}}


def fasta_contig_stats_python(fasta_path: str) -> str:
    """Count sequences and total bases in a FASTA file using the Python execution runner."""
    return format_result(run_tool_spec(_TOOL_SPEC, {'fasta_path': fasta_path}))
