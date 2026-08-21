"""Auto-generated wrapper for tool: seqmagick_info"""
import json as _json
from pathlib import Path as _Path

from agent_connector.tool_runner import run_tool_spec, format_result

_TOOL_SPEC = {'name': 'seqmagick_info', 'type': 'cli', 'arg_style': 'subcommand', 'command': 'seqmagick info', 'callable_via': '', 'readme_examples': [], 'readme_usage': '', 'description': 'Summarize one or more sequence files - report the number of sequences, the total length (total number of bases), and per-sequence statistics. Use this to count sequences and compute the total length of a FASTA file.', 'output_control': {'intercept_large_output': True, 'max_preview_lines': 50}, 'inputs': {'sequence_files': {'type': 'path', 'description': 'Input sequence file(s)', 'required': True, 'source': 'help_parsed', 'positional': True, 'position': 0, 'artifact': 'fasta', 'artifact_info': {'extensions': ['.fasta', '.fa', '.fna', '.faa'], 'description': 'DNA/RNA/protein sequences in FASTA format', 'fixture_ext': '.fasta'}}, 'input_format': {'type': 'string', 'description': 'Input format. Overrides extension for all input files', 'required': False, 'source': 'help_parsed', 'flag': '--input-format', 'takes_value': True}, 'out_file': {'type': 'path', 'description': 'Output destination. Default STDOUT', 'required': False, 'source': 'help_parsed', 'flag': '--out-file', 'takes_value': True}, 'format': {'type': 'string', 'description': 'Output format (tab/csv/align)', 'required': False, 'source': 'help_parsed', 'flag': '--format', 'takes_value': True, 'choices': ['tab', 'csv', 'align']}, 'threads': {'type': 'int', 'description': 'Number of threads (CPUs)', 'required': False, 'source': 'help_parsed', 'flag': '--threads', 'takes_value': True}}, 'resources': {}, 'outputs': {'stdout': {'type': 'text', 'description': 'Summary statistics printed to stdout', 'source': 'inferred'}}, 'execution': {'type': 'cli', 'command': 'seqmagick info'}, 'subcommands': ['convert', 'info', 'mogrify', 'quality-filter', 'extract-ids', 'backtrans-align'], 'subcommand_details': {'convert': {'description': 'Reverse complement a FASTA file: produce the reverse complement of each sequence and write it to the output file. Also converts between sequence formats (FASTA/FASTQ/CLU/PHYLIP/NEXUS/EMBL). This is the pure-Python, pip-installable way to reverse complement sequences - no Rust/cargo toolchain required (unlike bqtools revcomp).', 'params': [{'name': 'source_file', 'type': 'path', 'description': 'Input sequence file', 'required': True, 'positional': True, 'position': 0}, {'name': 'dest_file', 'type': 'path', 'description': 'Output file', 'required': True, 'positional': True, 'position': 1}, {'name': '--input-format', 'type': 'string', 'description': 'Input file format (default: determine from extension)', 'required': False, 'takes_value': True}, {'name': '--output-format', 'type': 'string', 'description': 'Output file format (default: determine from extension)', 'required': False, 'takes_value': True}, {'name': '--alphabet', 'type': 'string', 'description': 'Input alphabet', 'required': False, 'takes_value': True, 'choices': ['dna', 'dna-ambiguous', 'rna', 'rna-ambiguous', 'protein']}, {'name': '--reverse-complement', 'type': 'boolean', 'description': 'Reverse complement the sequences (writes the reverse complement of each input sequence)', 'required': False, 'takes_value': False}], 'outputs': {'dest_file': {'type': 'file', 'description': 'Converted sequence output file', 'input': 'dest_file', 'source': 'help_parsed'}}}, 'info': {'description': 'Summarize one or more sequence files - report the number of sequences, the total length (total number of bases), and per-sequence statistics. Use this to count sequences and compute the total length of a FASTA file.', 'params': [{'name': 'sequence_files', 'type': 'path', 'description': 'Input sequence file(s)', 'required': True, 'positional': True, 'position': 0}, {'name': '--input-format', 'type': 'string', 'description': 'Input format. Overrides extension for all input files', 'required': False, 'takes_value': True}, {'name': '--out-file', 'type': 'path', 'description': 'Output destination. Default STDOUT', 'required': False, 'takes_value': True}, {'name': '--format', 'type': 'string', 'description': 'Output format (tab/csv/align)', 'required': False, 'takes_value': True, 'choices': ['tab', 'csv', 'align']}, {'name': '--threads', 'type': 'int', 'description': 'Number of threads (CPUs)', 'required': False, 'takes_value': True}], 'outputs': {'stdout': {'type': 'text', 'description': 'Summary statistics printed to stdout', 'source': 'inferred'}}}}, 'subcommand_discovery_complete': False, 'install': {'method': 'pip_pkg', 'command': 'pip install seqmagick', 'system_commands': [], 'python_packages': ['seqmagick'], 'declared_packages': [], 'missing_deps': [], 'venv_path': ''}, '_active_subcommand': 'info'}


class SeqmagickInfoWrapper:
    """Tool wrapper backed by a ToolSpec (registry.yaml entry)."""

    def __init__(self):
        self.name = "seqmagick_info"
        self.description = "Summarize one or more sequence files - report the number of sequences, the total length (total number of bases), and per-sequence statistics. Use this to count sequences and compute the total length of a FASTA file."

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name}>"

    def run(self, **kwargs):
        return format_result(run_tool_spec(_TOOL_SPEC, kwargs))

    def call_with_args(self, tool_input):
        if isinstance(tool_input, str):
            tool_input = _json.loads(tool_input)
        return self.run(**tool_input)

    __call__ = run
    func = run
