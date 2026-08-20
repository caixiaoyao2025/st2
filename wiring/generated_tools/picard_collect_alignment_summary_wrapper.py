"""Auto-generated function wrapper for tool: picard_collect_alignment_summary"""
from agent_connector.tool_runner import run_tool_spec, format_result

_TOOL_SPEC = {'name': 'picard_collect_alignment_summary', 'type': 'java', 'command': 'java -jar {picard_jar_path} CollectAlignmentSummaryMetrics I={bam_path} O={metrics_tsv_path}', 'description': 'Run Picard CollectAlignmentSummaryMetrics from a user-supplied JAR and render the metrics table.', 'output_control': {'intercept_large_output': True, 'max_preview_lines': 60}, 'inputs': {'picard_jar_path': {'type': 'string', 'description': 'Path to picard.jar inside /data.'}, 'bam_path': {'type': 'string', 'description': 'Path to the input BAM file inside /data.'}, 'metrics_tsv_path': {'type': 'string', 'description': 'Target path for Picard metrics inside /data.'}}, 'expected_outputs': [{'name': 'metrics_tsv_path', 'render_as': 'dataframe', 'max_rows': 30}]}


def picard_collect_alignment_summary(picard_jar_path: str, bam_path: str, metrics_tsv_path: str) -> str:
    """Run Picard CollectAlignmentSummaryMetrics from a user-supplied JAR and render the metrics table."""
    return format_result(run_tool_spec(_TOOL_SPEC, {'picard_jar_path': picard_jar_path, 'bam_path': bam_path, 'metrics_tsv_path': metrics_tsv_path}))
