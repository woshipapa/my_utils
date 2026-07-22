"""
HTML布局构建器

用于构建完整的HTML报告布局
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
from enum import Enum
import time


class Severity(Enum):
    """严重程度"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Finding:
    """分析发现"""

    id: str
    title: str
    description: str
    severity: Severity
    category: str
    evidence: dict[str, Any]
    affected_components: list[str]
    metrics: dict[str, float]


@dataclass
class Recommendation:
    """优化建议"""

    id: str
    title: str
    description: str
    priority: int
    estimated_impact: str
    effort: str
    actions: list[str]
    references: list[str]


@dataclass
class AnalysisReport:
    """分析报告"""

    metadata: dict[str, Any]
    findings: list[Finding]
    recommendations: list[Recommendation]
    summary: str
    overall_score: float


@dataclass(frozen=True)
class PanelSpec:
    """可扩展的面板定义"""

    panel_id: str
    title: str
    description: str = ""
    order: int = 0
    enabled: bool = True
    min_height: str = "260px"


def default_timeline_panel_specs(include_metrics: bool = True) -> list[PanelSpec]:
    """默认时间线面板顺序（含可扩展新面板）"""
    specs = [
        PanelSpec(
            "gpu_metrics",
            "GPU Metrics In Window",
            "Per-metric/device curves",
            order=10,
            enabled=include_metrics,
            min_height="300px",
        ),
        PanelSpec(
            "rank_heatmap",
            "Rank Heatmap",
            "Cross-rank/device duration matrix",
            order=20,
        ),
        PanelSpec(
            "roofline_proxy",
            "Roofline Proxy",
            "Compute-vs-memory utilisation scatter",
            order=30,
        ),
        PanelSpec(
            "gil_lane", "Python GIL Lane", "Python runtime/GIL thread lanes", order=40
        ),
        PanelSpec(
            "kernel_category",
            "Kernel Category Breakdown",
            "Overlap-aware category mix",
            order=50,
        ),
        PanelSpec(
            "nvtx_stability",
            "Per-Matched-NVTX Category Stability",
            "Window-level category stability",
            order=60,
        ),
        PanelSpec(
            "all_streams",
            "All Streams Overlap + Metrics Alignment",
            "Merged stream timeline",
            order=70,
        ),
        PanelSpec(
            "nvtx_scopes", "Matched NVTX Scopes", "Matched NVTX range table", order=80
        ),
        PanelSpec(
            "kernel_timeline",
            "Kernel Timeline By Stream",
            "Per-stream kernel bars",
            order=90,
        ),
    ]
    return [item for item in sorted(specs, key=lambda x: x.order) if item.enabled]


class LayoutBuilder:
    """
    HTML布局构建器

    用于逐步构建HTML报告的各个部分
    """

    def __init__(self):
        self.sections = []
        self.scripts = []
        self.styles = []
        self.title = "Performance Analysis Report"
        self.subtitle = ""

    def set_title(self, title: str, subtitle: str = "") -> "LayoutBuilder":
        """设置报告标题"""
        self.title = title
        self.subtitle = subtitle
        return self

    def add_header(self, title: str = None, subtitle: str = None) -> "LayoutBuilder":
        """
        添加标题头部

        Args:
            title: 标题（None使用默认）
            subtitle: 副标题
        """
        if title is None:
            title = self.title

        if subtitle is None:
            subtitle = (
                self.subtitle or f"Generated at {time.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        self.sections.append(
            {
                "type": "header",
                "content": f"""
            <div class="report-header">
                <h1>{title}</h1>
                <p class="subtitle">{subtitle}</p>
            </div>
            """,
            }
        )
        return self

    def add_summary(
        self,
        summary: str,
        score: float,
        details: dict[str, Any] = None,
    ) -> "LayoutBuilder":
        """
        添加摘要部分

        Args:
            summary: 摘要文本
            score: 整体得分 (0-100)
            details: 详细信息字典
        """
        score_class = self._get_score_class(score)
        score_color = self._get_score_color(score)

        detail_items = ""
        if details:
            for key, value in details.items():
                detail_items += f'<div class="summary-detail"><span class="detail-label">{key}:</span> <span class="detail-value">{value}</span></div>'

        self.sections.append(
            {
                "type": "summary",
                "content": f"""
            <div class="summary-card score-{score_class}">
                <div class="score-circle" style="background: {score_color};">
                    {score:.0f}
                </div>
                <div class="summary-content">
                    <h3>Performance Score</h3>
                    <p class="summary-text">{summary}</p>
                    <div class="summary-details">{detail_items}</div>
                </div>
            </div>
            """,
            }
        )
        return self

    def add_metrics_grid(self, metrics: dict[str, Any]) -> "LayoutBuilder":
        """
        添加指标网格

        Args:
            metrics: 指标字典 {"name": value}
        """
        cards = ""
        for name, value in metrics.items():
            cards += f"""
            <div class="metric-card">
                <div class="metric-label">{name}</div>
                <div class="metric-value">{value}</div>
            </div>
            """

        self.sections.append(
            {
                "type": "metrics_grid",
                "content": f'<div class="metrics-grid">{cards}</div>',
            }
        )
        return self

    def add_chart(
        self,
        chart_html: str,
        title: str = "",
        width: str = None,
        height: str = "400px",
    ) -> "LayoutBuilder":
        """
        添加图表

        Args:
            chart_html: 图表HTML代码
            title: 图表标题
            width: 宽度
            height: 高度
        """
        style = ""
        if height:
            style += f"height: {height};"
        if width:
            style += f"width: {width};"

        title_html = f"<h3>{title}</h3>" if title else ""

        self.sections.append(
            {
                "type": "chart",
                "content": f"""
            <div class="chart-section" style="{style}">
                {title_html}
                <div class="chart-wrapper">
                    {chart_html}
                </div>
            </div>
            """,
            }
        )
        return self

    def add_two_column_charts(
        self,
        left_chart_html: str,
        right_chart_html: str,
        left_title: str = "",
        right_title: str = "",
    ) -> "LayoutBuilder":
        """
        添加两列图表布局

        Args:
            left_chart_html: 左侧图表HTML
            right_chart_html: 右侧图表HTML
            left_title: 左侧标题
            right_title: 右侧标题
        """
        left_title_html = f"<h3>{left_title}</h3>" if left_title else ""
        right_title_html = f"<h3>{right_title}</h3>" if right_title else ""

        self.sections.append(
            {
                "type": "two_column_charts",
                "content": f"""
            <div class="two-column-layout">
                <div class="column">
                    {left_title_html}
                    <div class="chart-wrapper">
                        {left_chart_html}
                    </div>
                </div>
                <div class="column">
                    {right_title_html}
                    <div class="chart-wrapper">
                        {right_chart_html}
                    </div>
                </div>
            </div>
            """,
            }
        )
        return self

    def add_panel_grid(
        self,
        panel_html_map: Dict[str, str],
        *,
        panel_specs: Optional[list[PanelSpec]] = None,
        title: str = "Panels",
    ) -> "LayoutBuilder":
        """
        添加可扩展面板网格（按 PanelSpec 顺序）

        Args:
            panel_html_map: {"panel_id": "<html>"} 映射
            panel_specs: 面板规范列表，默认使用 default_timeline_panel_specs()
            title: 区域标题
        """
        specs = panel_specs or default_timeline_panel_specs()
        cards: list[str] = []
        for spec in specs:
            panel_html = panel_html_map.get(spec.panel_id)
            if not panel_html:
                continue
            subtitle = (
                f'<div class="panel-subtitle">{spec.description}</div>'
                if spec.description
                else ""
            )
            cards.append(
                f"""
                <div class="panel-card" style="min-height: {spec.min_height};">
                    <h3>{spec.title}</h3>
                    {subtitle}
                    <div class="panel-body">
                        {panel_html}
                    </div>
                </div>
                """
            )

        if not cards:
            return self

        self.sections.append(
            {
                "type": "panel_grid",
                "content": f"""
            <div class="panel-grid-section">
                <h2>{title}</h2>
                <div class="panel-grid">
                    {"".join(cards)}
                </div>
            </div>
            """,
            }
        )
        return self

    def add_findings(
        self, findings: list[Finding], title: str = "Findings"
    ) -> "LayoutBuilder":
        """
        添加发现列表

        Args:
            findings: 发现列表
            title: 标题
        """
        items = []
        for finding in findings:
            items.append(self._render_finding(finding))

        findings_html = "\n".join(items)

        self.sections.append(
            {
                "type": "findings",
                "content": f"""
            <div class="findings-section">
                <h2>{title}</h2>
                <div class="findings-list">
                    {findings_html}
                </div>
            </div>
            """,
            }
        )
        return self

    def add_recommendations(
        self,
        recommendations: list[Recommendation],
        title: str = "Recommendations",
    ) -> "LayoutBuilder":
        """
        添加建议列表

        Args:
            recommendations: 建议列表
            title: 标题
        """
        items = []
        for rec in recommendations:
            items.append(self._render_recommendation(rec))

        rec_html = "\n".join(items)

        self.sections.append(
            {
                "type": "recommendations",
                "content": f"""
            <div class="recommendations-section">
                <h2>{title}</h2>
                <div class="recommendations-list">
                    {rec_html}
                </div>
            </div>
            """,
            }
        )
        return self

    def add_table(
        self,
        data: list[dict[str, Any]],
        title: str = "",
        sortable: bool = True,
    ) -> "LayoutBuilder":
        """
        添加表格

        Args:
            data: 数据行列表
            title: 表格标题
            sortable: 是否可排序
        """
        if not data:
            return self

        headers = list(data[0].keys())
        rows_html = ""

        for row in data:
            cells = "".join(f"<td>{str(row[h])}</td>" for h in headers)
            rows_html += f"<tr>{cells}</tr>"

        headers_html = "".join(f"<th>{h}</th>" for h in headers)

        sortable_attr = "sortable" if sortable else ""

        title_html = f"<h3>{title}</h3>" if title else ""

        self.sections.append(
            {
                "type": "table",
                "content": f"""
            <div class="table-section">
                {title_html}
                <table class="data-table" {sortable_attr}>
                    <thead>
                        <tr>{headers_html}</tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
            """,
            }
        )
        return self

    def add_code_block(self, code: str, language: str = "python") -> "LayoutBuilder":
        """
        添加代码块

        Args:
            code: 代码内容
            language: 语言类型
        """
        import html

        escaped_code = html.escape(code)

        self.sections.append(
            {
                "type": "code_block",
                "content": f"""
            <div class="code-block">
                <pre><code class="language-{language}">{escaped_code}</code></pre>
            </div>
            """,
            }
        )
        return self

    def add_section(self, html: str) -> "LayoutBuilder":
        """
        添加自定义HTML部分

        Args:
            html: HTML内容
        """
        self.sections.append({"type": "custom", "content": html})
        return self

    def add_divider(self) -> "LayoutBuilder":
        """添加分隔线"""
        self.sections.append(
            {"type": "divider", "content": '<hr class="section-divider">'}
        )
        return self

    def add_script(self, script: str) -> "LayoutBuilder":
        """
        添加JavaScript代码

        Args:
            script: JavaScript代码
        """
        self.scripts.append(script)
        return self

    def add_style(self, style: str) -> "LayoutBuilder":
        """
        添加CSS样式

        Args:
            style: CSS代码
        """
        self.styles.append(style)
        return self

    def build(self, include_styles: bool = True) -> str:
        """
        构建完整HTML

        Args:
            include_styles: 是否包含内置样式

        Returns:
            完整HTML字符串
        """
        sections_html = "".join(s["content"] for s in self.sections)

        if include_styles:
            css = self._get_builtin_css()
            # 添加自定义样式
            if self.styles:
                css += "\n" + "\n".join(self.styles)
        else:
            css = "\n".join(self.styles)

        scripts = "\n".join(self.scripts)

        template = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{self.title}</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
            <style>
            {css}
            </style>
        </head>
        <body>
            <div class="container">
                {sections_html}
            </div>
            <script>
            {scripts if scripts else "// No additional scripts"}
            </script>
        </body>
        </html>
        """

        return template

    def _get_score_class(self, score: float) -> str:
        """获取得分对应的CSS类"""
        if score >= 80:
            return "good"
        elif score >= 60:
            return "warning"
        else:
            return "critical"

    def _get_score_color(self, score: float) -> str:
        """获取得分对应的颜色"""
        if score >= 80:
            return "#10b981"
        elif score >= 60:
            return "#f59e0b"
        else:
            return "#ef4444"

    def _render_finding(self, finding: Finding) -> str:
        """渲染单个发现"""
        severity_badge = f'<span class="badge badge-{finding.severity.value}">{finding.severity.value.upper()}</span>'

        evidence_items = ""
        for key, value in finding.evidence.items():
            if key not in ["components", "metrics"]:
                if isinstance(value, float):
                    evidence_items += f'<div class="evidence-item"><span class="evidence-key">{key}:</span> <span class="evidence-value">{value:.4f}</span></div>'
                else:
                    evidence_items += f'<div class="evidence-item"><span class="evidence-key">{key}:</span> <span class="evidence-value">{value}</span></div>'

        components = (
            ", ".join(finding.affected_components)
            if finding.affected_components
            else finding.category
        )

        return f"""
        <div class="finding-item severity-{finding.severity.value}">
            <div class="finding-header">
                <span class="finding-title">{finding.title}</span>
                {severity_badge}
            </div>
            <div class="finding-description">{finding.description}</div>
            <div class="finding-meta">
                <span class="finding-component">Component: {components}</span>
            </div>
            {f'<div class="finding-evidence">{evidence_items}</div>' if evidence_items else ""}
        </div>
        """

    def _render_recommendation(self, rec: Recommendation) -> str:
        """渲染单个建议"""
        priority_class = (
            "high" if rec.priority >= 8 else "medium" if rec.priority >= 5 else "low"
        )

        actions_html = "".join(f"<li>{action}</li>" for action in rec.actions)

        return f"""
        <div class="recommendation-item priority-{priority_class}">
            <div class="rec-header">
                <span class="rec-title">{rec.title}</span>
                <span class="rec-priority">P{rec.priority}</span>
            </div>
            <div class="rec-description">{rec.description}</div>
            <div class="rec-meta">
                <span class="rec-impact"><strong>预期影响:</strong> {rec.estimated_impact}</span>
                <span class="rec-effort"><strong>工作量:</strong> {rec.effort}</span>
            </div>
            <div class="rec-actions">
                <strong>建议操作:</strong>
                <ul>{actions_html}</ul>
            </div>
            {f'<div class="rec-references"><strong>参考:</strong> {", ".join(f"<a href={ref} target=_blank>{ref}</a>" for ref in rec.references)}</div>' if rec.references else ""}
        </div>
        """

    def _get_builtin_css(self) -> str:
        """获取内置CSS样式"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #2c3e50;
            line-height: 1.6;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        /* Header */
        .report-header {
            background: white;
            border-radius: 16px;
            padding: 40px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            text-align: center;
        }

        .report-header h1 {
            font-size: 2.5rem;
            color: #2c3e50;
            margin-bottom: 12px;
        }

        .subtitle {
            color: #7f8c8d;
            font-size: 1.1rem;
        }

        /* Summary Card */
        .summary-card {
            background: white;
            border-radius: 16px;
            padding: 32px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            display: flex;
            align-items: center;
            gap: 32px;
        }

        .score-good { border-left: 6px solid #10b981; }
        .score-warning { border-left: 6px solid #f59e0b; }
        .score-critical { border-left: 6px solid #ef4444; }

        .score-circle {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 48px;
            font-weight: bold;
            color: white;
            flex-shrink: 0;
        }

        .summary-content {
            flex: 1;
        }

        .summary-content h3 {
            font-size: 1.5rem;
            margin-bottom: 12px;
            color: #2c3e50;
        }

        .summary-text {
            font-size: 1.1rem;
            color: #7f8c8d;
            margin-bottom: 16px;
        }

        .summary-details {
            display: flex;
            gap: 24px;
            flex-wrap: wrap;
        }

        .summary-detail {
            display: flex;
            gap: 8px;
        }

        .detail-label {
            font-weight: 600;
            color: #2c3e50;
        }

        .detail-value {
            color: #7f8c8d;
        }

        /* Metrics Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }

        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
        }

        .metric-label {
            color: #7f8c8d;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #2c3e50;
        }

        /* Chart */
        .chart-section {
            background: white;
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .chart-section h3 {
            margin-bottom: 16px;
            color: #2c3e50;
        }

        .chart-wrapper {
            position: relative;
            height: 400px;
        }

        .panel-grid-section {
            background: white;
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .panel-grid-section h2 {
            margin-bottom: 16px;
            color: #2c3e50;
        }

        .panel-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 16px;
        }

        .panel-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px;
            display: flex;
            flex-direction: column;
        }

        .panel-card h3 {
            margin-bottom: 6px;
            color: #2c3e50;
            font-size: 1rem;
        }

        .panel-subtitle {
            color: #64748b;
            font-size: 0.85rem;
            margin-bottom: 10px;
        }

        .panel-body {
            flex: 1;
        }

        /* Two Column Layout */
        .two-column-layout {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 24px;
            margin-bottom: 24px;
        }

        .two-column-layout .column {
            background: white;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .two-column-layout h3 {
            margin-bottom: 16px;
            color: #2c3e50;
        }

        /* Findings */
        .findings-section, .recommendations-section {
            background: white;
            border-radius: 16px;
            padding: 32px;
            margin-bottom: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .findings-section h2, .recommendations-section h2 {
            margin-bottom: 24px;
            color: #2c3e50;
        }

        .finding-item, .recommendation-item {
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
            border-left: 4px solid #cbd5e1;
            background: #f8fafc;
        }

        .severity-critical { border-left-color: #ef4444; background: #fef2f2; }
        .severity-high { border-left-color: #f97316; background: #fff7ed; }
        .severity-medium { border-left-color: #f59e0b; background: #fffbeb; }
        .severity-low { border-left-color: #3b82f6; background: #eff6ff; }

        .finding-header, .rec-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }

        .finding-title, .rec-title {
            font-weight: 600;
            font-size: 1.1rem;
            color: #2c3e50;
        }

        .badge {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .badge-critical { background: #ef4444; color: white; }
        .badge-high { background: #f97316; color: white; }
        .badge-medium { background: #f59e0b; color: white; }
        .badge-low { background: #3b82f6; color: white; }

        .finding-description, .rec-description {
            color: #7f8c8d;
            margin-bottom: 12px;
        }

        .finding-meta, .rec-meta {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin-bottom: 12px;
        }

        .finding-meta span, .rec-meta span {
            font-size: 0.9rem;
            color: #7f8c8d;
        }

        .finding-evidence {
            margin-top: 12px;
            padding: 12px;
            background: white;
            border-radius: 8px;
        }

        .evidence-item {
            display: flex;
            gap: 8px;
            margin-bottom: 4px;
        }

        .evidence-key {
            font-weight: 600;
            color: #2c3e50;
        }

        .evidence-value {
            color: #7f8c8d;
        }

        /* Recommendations */
        .priority-high { border-left-color: #ef4444; }
        .priority-medium { border-left-color: #f59e0b; }
        .priority-low { border-left-color: #3b82f6; }

        .rec-priority {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.9rem;
            font-weight: 600;
            background: #e2e8f0;
        }

        .rec-actions {
            margin-top: 12px;
        }

        .rec-actions ul {
            margin-left: 20px;
            color: #7f8c8d;
        }

        .rec-actions li {
            margin-bottom: 4px;
        }

        /* Table */
        .table-section {
            background: white;
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow-x: auto;
        }

        .table-section h3 {
            margin-bottom: 16px;
            color: #2c3e50;
        }

        .data-table {
            width: 100%;
            border-collapse: collapse;
        }

        .data-table th, .data-table td {
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }

        .data-table th {
            font-weight: 600;
            background: #f8fafc;
            color: #2c3e50;
        }

        .data-table tr:hover {
            background: #f8fafc;
        }

        /* Code Block */
        .code-block {
            background: #1e293b;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
            overflow-x: auto;
        }

        .code-block pre {
            margin: 0;
        }

        .code-block code {
            color: #e2e8f0;
            font-family: "Monaco", "Consolas", monospace;
            font-size: 0.9rem;
        }

        /* Divider */
        .section-divider {
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
            margin: 32px 0;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .two-column-layout {
                grid-template-columns: 1fr;
            }

            .summary-card {
                flex-direction: column;
                text-align: center;
            }

            .metrics-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        """
