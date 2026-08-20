"""Auto-generated function wrapper for tool: fastp_qc"""
from agent_connector.tool_runner import run_tool_spec, format_result

_TOOL_SPEC = {'name': 'fastp_qc', 'type': 'cli', 'command': 'fastp -i {fq_path} -o {filtered_fq_path} -h {html_report_path} -j {json_report_path} -R MCP_fastp_report', 'description': 'Run FASTQ quality control with fastp. Example: fastp -i input.fq -o trimmed.fq -h report.html -j report.json. NOTE: -j is for JSON report, -o is for trimmed FASTQ output (not JSON).', 'output_control': {'intercept_large_output': True, 'max_preview_lines': 30}, 'inputs': {'fq_path': {'type': 'string', 'description': 'Path to input FASTQ file inside /data.'}, 'filtered_fq_path': {'type': 'string', 'description': 'Target path for the filtered FASTQ output inside /data.'}, 'html_report_path': {'type': 'string', 'description': 'Target path for the fastp HTML report inside /data.'}, 'json_report_path': {'type': 'string', 'description': 'Target path for the fastp JSON report inside /data.'}}, 'expected_outputs': [{'name': 'html_report_path', 'render_as': 'text'}, {'name': 'json_report_path', 'render_as': 'text'}]}


def fastp_qc(fq_path: str, filtered_fq_path: str, html_report_path: str, json_report_path: str) -> str:
    """Run FASTQ quality control with fastp. Example: fastp -i input.fq -o trimmed.fq -h report.html -j report.json. NOTE: -j is for JSON report, -o is for trimmed FASTQ output (not JSON)."""
    return format_result(run_tool_spec(_TOOL_SPEC, {'fq_path': fq_path, 'filtered_fq_path': filtered_fq_path, 'html_report_path': html_report_path, 'json_report_path': json_report_path}))
