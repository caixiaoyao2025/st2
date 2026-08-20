"""Auto-generated function wrapper for tool: blastn_tabular"""
from agent_connector.tool_runner import run_tool_spec, format_result

_TOOL_SPEC = {'name': 'blastn_tabular', 'type': 'cli', 'command': 'blastn -query {query_fasta_path} -db {blast_db_path} -outfmt 6', 'description': 'Run BLASTN and return tabular outfmt 6 hits.', 'output_control': {'intercept_large_output': True, 'max_preview_lines': 80}, 'inputs': {'query_fasta_path': {'type': 'string', 'description': 'Absolute path to the query FASTA file inside /data.'}, 'blast_db_path': {'type': 'string', 'description': 'Absolute path or BLAST database prefix inside /data.'}}}


def blastn_tabular(query_fasta_path: str, blast_db_path: str) -> str:
    """Run BLASTN and return tabular outfmt 6 hits."""
    return format_result(run_tool_spec(_TOOL_SPEC, {'query_fasta_path': query_fasta_path, 'blast_db_path': blast_db_path}))
