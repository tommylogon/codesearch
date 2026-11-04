"""Set-of-Mark (SoM) graph renderer for vision-capable LLM agents and CLI previews.

Renders a code neighborhood subgraph (`graph.merged_neighborhood`) to a high-contrast PNG
where every node card displays its numeric ID (`#234605`), symbol name, kind, and file:line,
and every directed edge displays an arrowhead and relation badge. Vision agents interact
turn-by-turn by passing any visible `#id` back into `render_code_graph(symbol=..., expand=...)`
or `code_neighbors` / `get_code`.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import config, graph

KIND_COLOR = {
    "file": "#4da3ff",
    "class": "#f2c94c",
    "function": "#6fcf97",
    "method": "#27ae60",
    "variable": "#bb86fc",
    "comment": "#9aa2b1",
    "section": "#f2994a",
    "block": "#56ccf2",
    "element": "#eb5757",
    "record": "#9b8afb",
    "tag": "#7f8c9b",
}

EDGE_COLOR = {
    "DEFINES": "#6b7280",
    "CALLS": "#6fcf97",
    "REFERENCES": "#bb86fc",
    "IMPORTS": "#4da3ff",
    "INHERITS": "#f2c94c",
    "ANNOTATES": "#8b93a1",
    "LINKS_TO": "#f2994a",
    "CONNECTS_TO": "#56ccf2",
}


def _load_font(size: int, bold: bool = False):
    candidates = (
        ["consolab.ttf", "segoeuib.ttf", "arialbd.ttf"]
        if bold else ["consola.ttf", "segoeui.ttf", "arial.ttf"]
    )
    for name in candidates:
        for base in (Path("C:/Windows/Fonts"), Path("/usr/share/fonts")):
            p = base / name
            if p.exists():
                try:
                    return ImageFont.truetype(str(p), size)
                except OSError:
                    pass
    return ImageFont.load_default()


def _node_labels(n: dict) -> tuple[str, str, str]:
    nid = f"#{n['id']}"
    qname = n.get("qualified_name") or ""
    local = qname.split("::", 1)[1] if "::" in qname else (n.get("name") or "")
    base_file = (n.get("file") or "").split("/")[-1]
    if n.get("kind") == "file":
        title = base_file or qname
    else:
        title = local or base_file or f"{n.get('kind')}@{n.get('line_start', 1)}"
    if len(title) > 36:
        title = title[:35] + "..."
    sub = f"{n.get('kind', '')}  {base_file}:{n.get('line_start', 1)}"
    if len(sub) > 40:
        sub = sub[:39] + "..."
    return nid, title, sub


def _layout(nodes: list[dict], edges: list[dict], root_id: int,
            expanded_ids: list[int], width: int, height: int,
            cards: dict[int, tuple[float, float, str, str, str]]) -> dict[int, tuple[float, float]]:
    """Deterministic hub-and-ring seed followed by spring relaxation and exact box separation."""
    if not nodes:
        return {}
    header_h = 74
    pad_x, pad_y = 180.0, 72.0
    cx = width / 2.0
    cy = header_h + (height - header_h) / 2.0
    rx = max(220.0, (width / 2.0) - pad_x)
    ry = max(160.0, ((height - header_h) / 2.0) - pad_y)

    pos: dict[int, list[float]] = {}
    hubs = [root_id] + [eid for eid in expanded_ids if eid != root_id]
    hub_set = set(hubs)
    extra_hubs = [h for h in hubs[1:] if any(n["id"] == h for n in nodes)]

    if extra_hubs:
        pos[root_id] = [cx - rx * 0.42, cy]
        for i, hid in enumerate(extra_hubs):
            ang = (2.0 * math.pi * i) / max(1, len(extra_hubs))
            pos[hid] = [cx + rx * 0.22 + math.cos(ang) * rx * 0.20, cy + math.sin(ang) * ry * 0.30]
    else:
        pos[root_id] = [cx, cy]

    # Group remaining nodes by which hub they connect to first
    adj: dict[int, set[int]] = {n["id"]: set() for n in nodes}
    for e in edges:
        s, t = e["from"], e["to"]
        if s in adj and t in adj:
            adj[s].add(t)
            adj[t].add(s)

    others = [n["id"] for n in nodes if n["id"] not in pos]
    by_hub: dict[int, list[int]] = {h: [] for h in [root_id, *extra_hubs]}
    unassigned: list[int] = []
    for nid in others:
        placed = False
        for h in [root_id, *extra_hubs]:
            if h in adj.get(nid, set()):
                by_hub[h].append(nid)
                placed = True
                break
        if not placed:
            unassigned.append(nid)
    by_hub[root_id].extend(unassigned)

    for h, members in by_hub.items():
        if not members:
            continue
        hx, hy = pos.get(h, [cx, cy])
        ring_rx = rx * (0.76 if not extra_hubs else 0.48)
        ring_ry = ry * (0.76 if not extra_hubs else 0.72)
        for i, nid in enumerate(members):
            ang = (2.0 * math.pi * i) / len(members) - math.pi / 2.0
            pos[nid] = [hx + math.cos(ang) * ring_rx, hy + math.sin(ang) * ring_ry]

    ids = [n["id"] for n in nodes]
    for step in range(90):
        disp = {nid: [0.0, 0.0] for nid in ids}
        #Gentle spring attraction only on early steps
        if step < 25:
            for e in edges:
                s, t = e["from"], e["to"]
                if s in pos and t in pos and s != t:
                    dx = pos[t][0] - pos[s][0]
                    dy = pos[t][1] - pos[s][1]
                    dist = math.hypot(dx, dy) or 1.0
                    target = 250.0
                    if dist > target * 1.4:
                        pull = (dist - target) * 0.03
                        px, py = (dx / dist) * pull, (dy / dist) * pull
                        if s not in hub_set:
                            disp[s][0] += px
                            disp[s][1] += py
                        if t not in hub_set:
                            disp[t][0] -= px
                            disp[t][1] -= py
        # Exact per-card bounding-box repulsion
        for i in range(len(ids)):
            u = ids[i]
            ux, uy = pos[u]
            uw, uh = cards[u][0], cards[u][1]
            for j in range(i + 1, len(ids)):
                v = ids[j]
                vx, vy = pos[v]
                vw, vh = cards[v][0], cards[v][1]
                dx = ux - vx
                dy = uy - vy
                ox = ((uw + vw) / 2.0 + 18.0) - abs(dx)
                oy = ((uh + vh) / 2.0 + 14.0) - abs(dy)
                if ox > 0 and oy > 0:
                    if ox < oy:
                        sx = (1.0 if dx >= 0 else -1.0) * (ox * 0.6 + 1.0)
                        disp[u][0] += sx
                        disp[v][0] -= sx
                    else:
                        sy = (1.0 if dy >= 0 else -1.0) * (oy * 0.6 + 1.0)
                        disp[u][1] += sy
                        disp[v][1] -= sy
        for nid in ids:
            hw = cards[nid][0] / 2.0 + 16.0
            hh = cards[nid][1] / 2.0 + 14.0
            pos[nid][0] = min(width - hw, max(hw, pos[nid][0] + max(-36.0, min(36.0, disp[nid][0]))))
            pos[nid][1] = min(height - hh, max(header_h + hh, pos[nid][1] + max(-36.0, min(36.0, disp[nid][1]))))

    return {nid: (xy[0], xy[1]) for nid, xy in pos.items()}


def _clip_to_box(x1: float, y1: float, x2: float, y2: float, hw: float, hh: float) -> tuple[float, float]:
    """Return the point where the segment (x1, y1) -> (x2, y2) enters the target box at (x2, y2)."""
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return x2, y2
    tx = (hw / abs(dx)) if abs(dx) > 1e-6 else float("inf")
    ty = (hh / abs(dy)) if abs(dy) > 1e-6 else float("inf")
    t = min(tx, ty)
    if t >= 1.0:
        return x2, y2
    return x2 - dx * t, y2 - dy * t


def render_subgraph(conn, root_id: int, depth: int = 1, expand_ids: list[int] | None = None,
                    kinds: tuple[str, ...] | None = None, max_nodes: int = 60,
                    out_path: Path | None = None) -> dict:
    """Render a Set-of-Mark subgraph PNG and return both metadata and the saved file path."""
    expand_clean = [int(x) for x in (expand_ids or []) if int(x) != root_id]
    data = graph.merged_neighborhood(conn, root_id, depth=depth, expand_ids=expand_clean,
                                     kinds=kinds, max_nodes=max_nodes)
    nodes = data["nodes"]
    edges = data["edges"]

    n_count = len(nodes)
    width = 1600 if n_count <= 24 else (1920 if n_count <= 42 else 2200)
    height = 1000 if n_count <= 24 else (1220 if n_count <= 42 else 1440)

    img = Image.new("RGB", (width, height), "#14161a")
    draw = ImageDraw.Draw(img)
    f_title = _load_font(15, bold=True)
    f_bold = _load_font(12, bold=True)
    f_small = _load_font(11, bold=False)
    f_edge = _load_font(10, bold=True)

    # Header bar
    draw.rectangle([0, 0, width, 64], fill="#1b1e24", outline="#2c313a")
    root_node = next((n for n in nodes if n["id"] == root_id), None)
    root_name = root_node["qualified_name"] if root_node else f"#{root_id}"
    exp_str = f"  |  expanded: {', '.join('#' + str(x) for x in data['expanded'])}" if data.get("expanded") else ""
    draw.text((16, 10), f"codesearch graph  |  root: #{root_id} {root_name}  |  {len(nodes)} nodes, {len(edges)} edges{exp_str}",
              fill="#d7dae0", font=f_title)
    draw.text((16, 36),
              "Interact: pass any [#id] to render_code_graph(symbol=root_id, expand='id1,id2') to expand neighbors, or get_code(file, line_start, line_end) to read source.",
              fill="#8b93a1", font=f_small)

    # Legend on top right
    lx = width - 580
    for ek, ecol in EDGE_COLOR.items():
        draw.rectangle([lx, 14, lx + 14, 18], fill=ecol)
        draw.text((lx + 18, 8), ek, fill="#aab1bd", font=f_small)
        lx += 78

    # Precompute card sizes from exact font metrics
    cards: dict[int, tuple[float, float, str, str, str]] = {}
    for n in nodes:
        nid_str, title, sub = _node_labels(n)
        id_w = draw.textlength(nid_str, font=f_bold) + 12.0
        top_w = id_w + draw.textlength(title, font=f_bold) + 22.0
        bot_w = draw.textlength(sub, font=f_small) + 20.0
        w = max(176.0, top_w, bot_w)
        h = 46.0
        cards[n["id"]] = (w, h, nid_str, title, sub)

    pos = _layout(nodes, edges, root_id, data.get("expanded", []), width, height, cards)

    # Draw directed edges first
    for e in edges:
        s, t = e["from"], e["to"]
        if s not in pos or t not in pos or s == t:
            continue
        x1, y1 = pos[s]
        x2, y2 = pos[t]
        tw, th, _, _, _ = cards[t]
        ex, ey = _clip_to_box(x1, y1, x2, y2, tw / 2.0 + 4.0, th / 2.0 + 4.0)
        col = EDGE_COLOR.get(e["kind"], "#6b7280")
        draw.line([(x1, y1), (ex, ey)], fill=col, width=2)

        # Arrowhead at (ex, ey)
        ang = math.atan2(ey - y1, ex - x1)
        alen = 10.0
        a1 = (ex - alen * math.cos(ang - 0.42), ey - alen * math.sin(ang - 0.42))
        a2 = (ex - alen * math.cos(ang + 0.42), ey - alen * math.sin(ang + 0.42))
        draw.polygon([(ex, ey), a1, a2], fill=col)

        # Midpoint edge badge for non-DEFINES edges
        if e["kind"] != "DEFINES":
            mx = (x1 * 0.45 + ex * 0.55)
            my = (y1 * 0.45 + ey * 0.55)
            lbl = e["kind"] if not e.get("meta") else f"{e['kind']}:{e['meta']}"
            bw = len(lbl) * 6.2 + 8.0
            draw.rectangle([mx - bw / 2, my - 7, mx + bw / 2, my + 7], fill="#14161a", outline=col)
            draw.text((mx - bw / 2 + 4, my - 6), lbl, fill=col, font=f_edge)

    # Draw node cards with Set-of-Mark [#id] badges
    expanded_set = set(data.get("expanded", []))
    for n in nodes:
        nid = n["id"]
        if nid not in pos:
            continue
        cx, cy = pos[nid]
        w, h, nid_str, title, sub = cards[nid]
        x0, y0 = cx - w / 2.0, cy - h / 2.0
        x1, y1 = cx + w / 2.0, cy + h / 2.0
        bg = KIND_COLOR.get(n["kind"], "#8b93a1")

        if nid == root_id:
            draw.rounded_rectangle([x0 - 4, y0 - 4, x1 + 4, y1 + 4], radius=9, outline="#4da3ff", width=3)
        elif nid in expanded_set:
            draw.rounded_rectangle([x0 - 3, y0 - 3, x1 + 3, y1 + 3], radius=8, outline="#f2c94c", width=2)

        draw.rounded_rectangle([x0, y0, x1, y1], radius=6, fill=bg, outline="#0b0d10", width=2)

        # Dark Set-of-Mark ID badge on top-left of card
        id_w = draw.textlength(nid_str, font=f_bold) + 12.0
        draw.rounded_rectangle([x0 + 5, y0 + 5, x0 + 5 + id_w, y0 + 23], radius=4, fill="#14161a")
        draw.text((x0 + 11, y0 + 6), nid_str, fill="#ffffff", font=f_bold)

        # Symbol name and file:line subtitle
        draw.text((x0 + 11 + id_w, y0 + 6), title, fill="#111318", font=f_bold)
        draw.text((x0 + 8, y0 + 26), sub, fill="#1b1e24", font=f_small)

    if out_path is None:
        renders_dir = config.DATA_DIR / "renders"
        renders_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", root_name)[:48].strip("_") or str(root_id)
        exp_tag = ("_exp_" + "_".join(map(str, expand_clean[:4]))) if expand_clean else ""
        out_path = renders_dir / f"graph_{root_id}_{slug}{exp_tag}.png"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    img.save(out_path, format="PNG")
    url = f"http://127.0.0.1:{config.WEB_PORT}/?node={root_id}&depth={depth}"
    if expand_clean:
        url += "&expand=" + ",".join(map(str, expand_clean))

    return {
        "image_path": str(out_path.resolve()),
        "url": url,
        "root": root_id,
        "expanded": data.get("expanded", []),
        "truncated": data["truncated"],
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": [
            {"id": n["id"], "mark": f"#{n['id']}", "kind": n["kind"], "name": n["name"],
             "qualified_name": n["qualified_name"], "file": n["file"],
             "lines": [n["line_start"], n["line_end"]]}
            for n in nodes
        ],
        "edges": edges,
        "next_actions": {
            "expand_node_in_place": f"render_code_graph(symbol={root_id}, depth={depth}, expand='<comma_separated_node_ids>')",
            "inspect_node_code": "get_code(rel_path='<node.file>', line_start=<line_start>, line_end=<line_end>)",
            "open_interactive_web_ui": url,
        },
    }
