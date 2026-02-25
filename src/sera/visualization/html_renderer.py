"""HTML renderer for the search tree visualization.

Generates a self-contained HTML file with inline CSS and JavaScript.
Uses D3.js (CDN with inline fallback note) for interactive tree visualization.
No Jinja2 dependency -- pure f-string / string.Template.
"""

from __future__ import annotations

import json
from pathlib import Path
from string import Template
from typing import Any


def render_html(
    tree_data: dict,
    stats_data: dict,
    node_artifacts: dict[str, dict],
    step: int,
    output_path: Path,
) -> Path:
    """Generate a self-contained HTML visualization file.

    Parameters
    ----------
    tree_data : dict
        D3.js hierarchy data structure.
    stats_data : dict
        Statistics summary from ``compute_stats()``.
    node_artifacts : dict[str, dict]
        Per-node artifacts (experiment_code, stdout, stderr, metrics).
    step : int
        Current search step number.
    output_path : Path
        Where to write the HTML file.

    Returns
    -------
    Path
        The output path.
    """
    tree_json = json.dumps(tree_data, default=str, ensure_ascii=False)
    stats_json = json.dumps(stats_data, default=str, ensure_ascii=False)
    artifacts_json = json.dumps(node_artifacts, default=str, ensure_ascii=False)

    total_nodes = stats_data.get("total_nodes", 0)

    html = _HTML_TEMPLATE.safe_substitute(
        step=step,
        total_nodes=total_nodes,
        tree_data_json=tree_json,
        stats_data_json=stats_json,
        artifacts_json=artifacts_json,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = Template(r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SERA Search Tree Visualization — Step $step</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #e0e0e0; }
header { background: #16213e; padding: 12px 24px; display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #0f3460; }
header h1 { font-size: 18px; color: #e94560; }
header .meta { font-size: 14px; color: #a0a0a0; }
main { display: flex; height: calc(100vh - 200px); min-height: 400px; }
#tree-container { flex: 1; overflow: hidden; background: #0f0f23; position: relative; }
#tree-container svg { width: 100%; height: 100%; }
#detail-panel { width: 380px; overflow-y: auto; background: #16213e; border-left: 2px solid #0f3460; padding: 16px; font-size: 13px; }
#detail-panel h2 { color: #e94560; font-size: 15px; margin-bottom: 12px; }
#detail-panel .section-title { color: #53a8b6; font-weight: bold; margin-top: 12px; margin-bottom: 4px; font-size: 12px; text-transform: uppercase; }
#detail-panel .field { margin-bottom: 6px; }
#detail-panel .field .label { color: #a0a0a0; display: inline-block; width: 100px; }
#detail-panel .field .value { color: #e0e0e0; }
.config-table { width: 100%; border-collapse: collapse; margin: 4px 0; font-size: 12px; }
.config-table th, .config-table td { border: 1px solid #333; padding: 3px 6px; text-align: left; }
.config-table th { background: #0f3460; }
.hypothesis-text { background: #1a1a2e; padding: 8px; border-radius: 4px; margin: 4px 0; white-space: pre-wrap; word-wrap: break-word; font-size: 12px; max-height: 120px; overflow-y: auto; }
.btn-group { display: flex; gap: 6px; margin-top: 8px; }
.btn { padding: 4px 10px; background: #0f3460; border: 1px solid #53a8b6; border-radius: 3px; color: #e0e0e0; cursor: pointer; font-size: 11px; }
.btn:hover { background: #1a4a8a; }
.btn:disabled { opacity: 0.4; cursor: not-allowed; }
.status-badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: bold; }
#stats-panel { background: #16213e; border-top: 2px solid #0f3460; padding: 16px 24px; display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start; }
.stat-card { background: #1a1a2e; border-radius: 6px; padding: 12px; min-width: 180px; }
.stat-card h3 { font-size: 12px; color: #53a8b6; margin-bottom: 8px; text-transform: uppercase; }
.stat-card .big-number { font-size: 28px; font-weight: bold; color: #e94560; }
.stat-card .sub { font-size: 11px; color: #a0a0a0; }
.chart-container { min-width: 200px; max-width: 350px; }
.chart-container canvas { max-height: 120px; }
.bar-chart { display: flex; align-items: flex-end; gap: 4px; height: 80px; }
.bar { display: flex; flex-direction: column; align-items: center; }
.bar-fill { width: 24px; border-radius: 2px 2px 0 0; transition: height 0.3s; }
.bar-label { font-size: 9px; color: #a0a0a0; margin-top: 2px; writing-mode: vertical-lr; transform: rotate(180deg); max-height: 50px; overflow: hidden; }
.bar-value { font-size: 9px; color: #e0e0e0; }

/* Modal */
.modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 1000; justify-content: center; align-items: center; }
.modal.active { display: flex; }
.modal-content { background: #1a1a2e; border: 1px solid #0f3460; border-radius: 8px; width: 80%; max-width: 900px; max-height: 80vh; display: flex; flex-direction: column; }
.modal-header { display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; border-bottom: 1px solid #333; }
.modal-header h3 { color: #e94560; font-size: 14px; }
.modal-close { background: none; border: none; color: #e0e0e0; font-size: 20px; cursor: pointer; }
.modal-body { flex: 1; overflow: auto; padding: 16px; }
.modal-body pre { background: #0f0f23; padding: 12px; border-radius: 4px; font-size: 12px; white-space: pre-wrap; word-wrap: break-word; font-family: 'Fira Code', 'Consolas', monospace; }

/* Tooltip */
.tooltip { position: absolute; background: #16213e; border: 1px solid #53a8b6; border-radius: 4px; padding: 8px 12px; font-size: 12px; pointer-events: none; z-index: 100; max-width: 300px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }

/* Node shapes */
.node circle { stroke-width: 2px; cursor: pointer; }
.node rect { stroke-width: 2px; cursor: pointer; }
.node polygon { stroke-width: 2px; cursor: pointer; }
.node text { font-size: 10px; fill: #e0e0e0; }
.link { fill: none; stroke-width: 1.5px; }

/* Best node highlight */
.node.best circle, .node.best rect, .node.best polygon { stroke: #FFD700 !important; stroke-width: 3px; }
.node.infeasible circle, .node.infeasible rect, .node.infeasible polygon { stroke-dasharray: 4,3; stroke: #E57373; }

/* Children count */
.children-count { font-size: 10px; color: #a0a0a0; margin: 2px 0; }
.lcb-history { width: 100%; max-width: 350px; }
.lcb-history svg { width: 100%; }
</style>
</head>
<body>

<header>
    <h1>SERA Search Tree Visualization</h1>
    <div class="meta">Step: <strong>$step</strong> | Nodes: <strong>$total_nodes</strong></div>
</header>

<main>
    <div id="tree-container"></div>
    <div id="detail-panel">
        <h2>Node Detail</h2>
        <p style="color:#a0a0a0;">Click a node in the tree to view details.</p>
    </div>
</main>

<section id="stats-panel"></section>

<div id="code-modal" class="modal">
    <div class="modal-content">
        <div class="modal-header">
            <h3 id="modal-title">Code</h3>
            <button class="modal-close" onclick="closeModal()">&times;</button>
        </div>
        <div class="modal-body">
            <pre id="modal-code"></pre>
        </div>
    </div>
</div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
// =========================================================================
// Data (injected by Python)
// =========================================================================
const TREE_DATA = $tree_data_json;
const STATS_DATA = $stats_data_json;
const NODE_ARTIFACTS = $artifacts_json;

const STATUS_COLORS = {
    "pending":   "#E0E0E0",
    "running":   "#FFF176",
    "evaluated": "#81C784",
    "failed":    "#E57373",
    "timeout":   "#FFB74D",
    "oom":       "#CE93D8",
    "pruned":    "#BDBDBD",
    "expanded":  "#64B5F6"
};

const OP_SHAPES = {
    "draft":   "circle",
    "debug":   "triangle",
    "improve": "rect"
};

// =========================================================================
// Tree Visualization
// =========================================================================
(function() {
    const container = document.getElementById('tree-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    const svg = d3.select('#tree-container')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    const g = svg.append('g');

    // Zoom and pan
    const zoom = d3.zoom()
        .scaleExtent([0.1, 4])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });
    svg.call(zoom);

    // Tooltip
    const tooltip = d3.select('body').append('div').attr('class', 'tooltip').style('display', 'none');

    // Build hierarchy
    const root = d3.hierarchy(TREE_DATA, d => d.children);

    // Tree layout
    const treeLayout = d3.tree()
        .size([height - 80, width - 200])
        .separation((a, b) => (a.parent === b.parent ? 1.5 : 2));

    treeLayout(root);

    // Compute node count for auto-scaling
    const nodeCount = root.descendants().length;
    const autoScale = nodeCount > 50 ? Math.max(0.3, 50 / nodeCount) : 1;

    // Links
    g.selectAll('.link')
        .data(root.links())
        .enter()
        .append('path')
        .attr('class', 'link')
        .attr('d', d3.linkHorizontal()
            .x(d => d.y + 100)
            .y(d => d.x + 40))
        .attr('stroke', d => {
            const op = d.target.data.data ? d.target.data.data.branching_op : 'draft';
            if (op === 'debug') return '#E57373';
            if (op === 'improve') return '#64B5F6';
            return '#666';
        })
        .attr('stroke-dasharray', d => {
            const op = d.target.data.data ? d.target.data.data.branching_op : 'draft';
            return op === 'debug' ? '6,3' : 'none';
        });

    // Nodes
    const nodes = g.selectAll('.node')
        .data(root.descendants())
        .enter()
        .append('g')
        .attr('class', d => {
            let cls = 'node';
            const data = d.data.data || {};
            if (data.node_id === STATS_DATA.best_node?.node_id) cls += ' best';
            if (data.feasible === false) cls += ' infeasible';
            return cls;
        })
        .attr('transform', d => `translate($${d.y + 100},$${d.x + 40})`)
        .on('click', (event, d) => showNodeDetail(d.data))
        .on('mouseover', (event, d) => {
            const data = d.data.data || {};
            const hyp = (data.hypothesis || '').substring(0, 50);
            const mu = data.mu !== null && data.mu !== undefined ? data.mu.toFixed(4) : 'N/A';
            const se = data.se !== null && data.se !== undefined ? data.se.toFixed(4) : 'N/A';
            tooltip.style('display', 'block')
                .html(`<strong>${data.status || 'pending'}</strong><br>${hyp}...<br>μ=${mu} ± ${se}`)
                .style('left', (event.pageX + 12) + 'px')
                .style('top', (event.pageY - 12) + 'px');
        })
        .on('mouseout', () => tooltip.style('display', 'none'));

    // Draw shapes based on branching_op
    nodes.each(function(d) {
        const el = d3.select(this);
        const data = d.data.data || {};
        const op = data.branching_op || 'draft';
        const status = data.status || 'pending';
        const color = STATUS_COLORS[status] || '#E0E0E0';
        const lcb = data.lcb;
        const baseSize = 12;
        const size = lcb != null ? Math.max(8, Math.min(24, baseSize + lcb * 2)) : baseSize;

        if (op === 'debug') {
            // Triangle
            const h = size;
            const points = `0,$${-h} $${h*0.866},$${h*0.5} $${-h*0.866},$${h*0.5}`;
            el.append('polygon').attr('points', points).attr('fill', color).attr('stroke', color);
        } else if (op === 'improve') {
            // Rectangle
            el.append('rect')
                .attr('x', -size).attr('y', -size)
                .attr('width', size * 2).attr('height', size * 2)
                .attr('rx', 3)
                .attr('fill', color).attr('stroke', color);
        } else {
            // Circle (draft and default)
            el.append('circle').attr('r', size).attr('fill', color).attr('stroke', color);
        }
    });

    // Auto-fit
    svg.call(zoom.transform, d3.zoomIdentity.translate(40, 20).scale(autoScale));
})();

// =========================================================================
// Detail Panel
// =========================================================================
function showNodeDetail(nodeData) {
    const data = nodeData.data || nodeData;
    const panel = document.getElementById('detail-panel');
    if (!data || !data.node_id) {
        panel.innerHTML = '<h2>Node Detail</h2><p style="color:#a0a0a0;">No data available.</p>';
        return;
    }

    const statusColor = STATUS_COLORS[data.status] || '#E0E0E0';
    const nid = data.node_id;
    const artifacts = NODE_ARTIFACTS[nid] || {};

    let configRows = '';
    if (data.experiment_config) {
        for (const [k, v] of Object.entries(data.experiment_config)) {
            configRows += `<tr><td>${esc(k)}</td><td>${esc(JSON.stringify(v))}</td></tr>`;
        }
    }

    let childrenHtml = '';
    if (data.children_ids && data.children_ids.length > 0) {
        childrenHtml = data.children_ids.map(cid =>
            `<div style="margin:2px 0;font-size:12px;">&#x2022; ${esc(cid.substring(0,12))}...</div>`
        ).join('');
    } else {
        childrenHtml = '<em>None</em>';
    }

    let failureHtml = '';
    if (data.failure_context && data.failure_context.length > 0) {
        failureHtml = data.failure_context.map(fc =>
            `<div>[${esc(fc.error_category||'')}] ${esc(fc.lesson||'')}</div>`
        ).join('');
    }

    const hasCode = !!artifacts.experiment_code;
    const hasStdout = !!artifacts.stdout;
    const hasStderr = !!artifacts.stderr;

    panel.innerHTML = `
        <h2>Node Detail</h2>
        <div class="field"><span class="label">Node ID:</span> <span class="value">${esc(nid.substring(0,20))}...</span></div>
        <div class="field"><span class="label">Status:</span> <span class="status-badge" style="background:${statusColor};color:#000;">${esc(data.status)}</span></div>
        <div class="field"><span class="label">Operator:</span> <span class="value">${esc(data.branching_op||'')}</span></div>
        <div class="field"><span class="label">Depth:</span> <span class="value">${data.depth}</span></div>
        <div class="field"><span class="label">Created:</span> <span class="value">${esc(data.created_at||'')}</span></div>

        <div class="section-title">Hypothesis</div>
        <div class="hypothesis-text">${esc(data.hypothesis||'')}</div>

        ${data.rationale ? `<div class="section-title">Rationale</div><div class="hypothesis-text">${esc(data.rationale)}</div>` : ''}

        <div class="section-title">Experiment Config</div>
        ${configRows ? `<table class="config-table"><tr><th>Key</th><th>Value</th></tr>${configRows}</table>` : '<em>No configuration</em>'}

        <div class="section-title">Metrics</div>
        <div class="field"><span class="label">μ =</span> <span class="value">${data.mu !== null && data.mu !== undefined ? data.mu : 'N/A'}</span></div>
        <div class="field"><span class="label">SE =</span> <span class="value">${data.se !== null && data.se !== undefined ? data.se : 'N/A'}</span></div>
        <div class="field"><span class="label">LCB =</span> <span class="value">${data.lcb !== null && data.lcb !== undefined ? data.lcb : 'N/A'}</span></div>
        <div class="field"><span class="label">Eval Runs:</span> <span class="value">${data.eval_runs || 0}</span></div>
        <div class="field"><span class="label">Feasible:</span> <span class="value">${data.feasible ? '&#x2713;' : '&#x2717;'}</span></div>
        <div class="field"><span class="label">Priority:</span> <span class="value">${data.priority !== null ? data.priority : 'N/A'}</span></div>

        <div class="section-title">Resources</div>
        <div class="field"><span class="label">Total Cost:</span> <span class="value">${data.total_cost || 0}</span></div>
        <div class="field"><span class="label">Wall Time:</span> <span class="value">${(data.wall_time_sec || 0).toFixed(1)}s</span></div>

        ${failureHtml ? `<div class="section-title">Failure Context (ECHO)</div>${failureHtml}` : ''}

        <div class="section-title">Children</div>
        ${childrenHtml}

        ${data.error_message ? `<div class="section-title">Error Message</div><div class="hypothesis-text" style="color:#E57373;">${esc(data.error_message)}</div>` : ''}

        <div class="btn-group">
            <button class="btn" ${hasCode ? `onclick="showModal('${esc(nid)}','code')"` : 'disabled'}>Code</button>
            <button class="btn" ${hasStdout ? `onclick="showModal('${esc(nid)}','stdout')"` : 'disabled'}>stdout</button>
            <button class="btn" ${hasStderr ? `onclick="showModal('${esc(nid)}','stderr')"` : 'disabled'}>stderr</button>
        </div>
    `;
}

function esc(s) {
    if (s == null) return '';
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// =========================================================================
// Modal
// =========================================================================
function showModal(nodeId, type) {
    const artifacts = NODE_ARTIFACTS[nodeId] || {};
    let content = '';
    let title = '';
    if (type === 'code') { content = artifacts.experiment_code || ''; title = 'Experiment Code'; }
    else if (type === 'stdout') { content = artifacts.stdout || ''; title = 'stdout'; }
    else if (type === 'stderr') { content = artifacts.stderr || ''; title = 'stderr'; }

    document.getElementById('modal-title').textContent = title + ' — ' + nodeId.substring(0, 12);
    document.getElementById('modal-code').textContent = content;
    document.getElementById('code-modal').classList.add('active');
}
function closeModal() { document.getElementById('code-modal').classList.remove('active'); }
document.getElementById('code-modal').addEventListener('click', function(e) {
    if (e.target === this) closeModal();
});

// =========================================================================
// Stats Panel
// =========================================================================
(function() {
    const panel = document.getElementById('stats-panel');
    const s = STATS_DATA;

    // Card: Step
    let html = `<div class="stat-card"><h3>Search Step</h3><div class="big-number">${s.step}</div></div>`;

    // Card: Total Nodes
    html += `<div class="stat-card"><h3>Total Nodes</h3><div class="big-number">${s.total_nodes}</div>
        <div class="sub">Success rate: ${(s.success_rate * 100).toFixed(1)}%</div></div>`;

    // Card: Best Node
    if (s.best_node && s.best_node.node_id) {
        html += `<div class="stat-card"><h3>Best Node</h3>
            <div style="font-size:12px;">${esc(s.best_node.node_id.substring(0,12))}...</div>
            <div class="big-number">${s.best_node.lcb !== null ? s.best_node.lcb.toFixed(4) : 'N/A'}</div>
            <div class="sub">μ=${s.best_node.mu !== null ? s.best_node.mu.toFixed(4) : 'N/A'} ± ${s.best_node.se !== null ? s.best_node.se.toFixed(4) : 'N/A'}</div>
        </div>`;
    }

    // Status bar chart
    html += '<div class="stat-card chart-container"><h3>Status Distribution</h3><div class="bar-chart">';
    const maxCount = Math.max(...Object.values(s.status_counts || {}), 1);
    for (const [status, count] of Object.entries(s.status_counts || {})) {
        const h = Math.max(4, (count / maxCount) * 70);
        const color = STATUS_COLORS[status] || '#666';
        html += `<div class="bar">
            <div class="bar-value">${count}</div>
            <div class="bar-fill" style="height:${h}px;background:${color};"></div>
            <div class="bar-label">${status}</div>
        </div>`;
    }
    html += '</div></div>';

    // Operator distribution
    html += '<div class="stat-card chart-container"><h3>Operator Distribution</h3><div class="bar-chart">';
    const maxOp = Math.max(...Object.values(s.operator_counts || {}), 1);
    const opColors = { draft: '#666', debug: '#E57373', improve: '#64B5F6' };
    for (const [op, count] of Object.entries(s.operator_counts || {})) {
        const h = Math.max(4, (count / maxOp) * 70);
        html += `<div class="bar">
            <div class="bar-value">${count}</div>
            <div class="bar-fill" style="height:${h}px;background:${opColors[op]||'#888'};"></div>
            <div class="bar-label">${op}</div>
        </div>`;
    }
    html += '</div></div>';

    // Depth distribution
    html += '<div class="stat-card chart-container"><h3>Depth Distribution</h3><div class="bar-chart">';
    const maxD = Math.max(...Object.values(s.depth_distribution || {}), 1);
    for (const [depth, count] of Object.entries(s.depth_distribution || {})) {
        const h = Math.max(4, (count / maxD) * 70);
        html += `<div class="bar">
            <div class="bar-value">${count}</div>
            <div class="bar-fill" style="height:${h}px;background:#53a8b6;"></div>
            <div class="bar-label">d${depth}</div>
        </div>`;
    }
    html += '</div></div>';

    // LCB History (simple SVG line chart)
    if (s.best_lcb_history && s.best_lcb_history.length > 1) {
        const hist = s.best_lcb_history;
        const chartW = 300, chartH = 80, pad = 20;
        const minLcb = Math.min(...hist.map(h => h.lcb));
        const maxLcb = Math.max(...hist.map(h => h.lcb));
        const range = maxLcb - minLcb || 1;

        let points = hist.map((h, i) => {
            const x = pad + (i / (hist.length - 1)) * (chartW - 2 * pad);
            const y = pad + (1 - (h.lcb - minLcb) / range) * (chartH - 2 * pad);
            return `${x},${y}`;
        }).join(' ');

        html += `<div class="stat-card lcb-history"><h3>Best LCB History</h3>
            <svg viewBox="0 0 ${chartW} ${chartH}">
                <polyline points="${points}" fill="none" stroke="#e94560" stroke-width="2"/>
                <text x="${pad}" y="${chartH-2}" fill="#a0a0a0" font-size="9">${minLcb.toFixed(2)}</text>
                <text x="${pad}" y="${pad-2}" fill="#a0a0a0" font-size="9">${maxLcb.toFixed(2)}</text>
            </svg>
        </div>`;
    }

    panel.innerHTML = html;
})();
</script>
</body>
</html>""")
