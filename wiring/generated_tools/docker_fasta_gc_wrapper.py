"""Auto-generated function wrapper for tool: docker_fasta_gc"""
from agent_connector.tool_runner import run_tool_spec, format_result

_TOOL_SPEC = {'name': 'docker_fasta_gc', 'description': 'Compute GC content of a FASTA file inside the staphb/fastqc image using the Docker runner.', 'output_control': {'intercept_large_output': True, 'max_preview_lines': 50}, 'execution': {'type': 'docker', 'image': 'alpine:3.20', 'command': 'sh -c "grep -v \'^>\' {fasta_path} | tr -d \'\\n\' | awk \'{{gc=gsub(/[GgCc]/,\\"\\"); total=length($0); print \\"GC%=\\" (gc/total)*100}}\'"'}, 'inputs': {'fasta_path': {'type': 'string', 'description': 'Path to a FASTA file inside /data.'}}}


def docker_fasta_gc(fasta_path: str) -> str:
    """Compute GC content of a FASTA file inside the staphb/fastqc image using the Docker runner."""
    return format_result(run_tool_spec(_TOOL_SPEC, {'fasta_path': fasta_path}))
