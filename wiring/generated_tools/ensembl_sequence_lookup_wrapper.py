"""Auto-generated function wrapper for tool: ensembl_sequence_lookup"""
from agent_connector.tool_runner import run_tool_spec, format_result

_TOOL_SPEC = {'name': 'ensembl_sequence_lookup', 'description': 'Fetch a DNA sequence by Ensembl gene/transcript ID via the Ensembl REST API.', 'output_control': {'intercept_large_output': True, 'max_preview_lines': 20}, 'execution': {'type': 'api', 'method': 'GET', 'endpoint': 'https://rest.ensembl.org/sequence/id/{id}'}, 'inputs': {'id': {'type': 'string', 'description': 'Ensembl stable ID such as ENSG00000141510.'}}}


def ensembl_sequence_lookup(id: str) -> str:
    """Fetch a DNA sequence by Ensembl gene/transcript ID via the Ensembl REST API."""
    return format_result(run_tool_spec(_TOOL_SPEC, {'id': id}))
