"""HTML extractor (stdlib html.parser).

Nodes: `section` for h1-h3 headings, `element` for anything carrying an id, and
`block` for inline <script>/<style>. Inline scripts are returned in `scripts` so the
indexer can send them through the JS extractor with the right line offset.
<script src>, <link href> and <img src> become IMPORTS refs.
"""
from __future__ import annotations

from html.parser import HTMLParser

_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
         "param", "source", "track", "wbr"}


class _Collector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.nodes: list[dict] = []
        self.refs: list[dict] = []
        self.scripts: list[dict] = []
        self.title = ""
        self._tag_stack: list[str] = []
        self._capture: dict | None = None
        self._script: dict | None = None

    def _line(self) -> int:
        return self.getpos()[0]

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        line = self._line()
        if tag == "script":
            if a.get("src"):
                self.refs.append({"source": "", "kind": "IMPORTS", "target": a["src"],
                                  "qualifier": "", "meta": ""})
            else:
                self._script = {"start": line, "buf": [], "tag": "script",
                                "type": a.get("type", "")}
        elif tag == "style":
            self._script = {"start": line, "buf": [], "tag": "style", "type": ""}
        elif tag == "link" and a.get("href"):
            self.refs.append({"source": "", "kind": "IMPORTS", "target": a["href"],
                              "qualifier": "", "meta": ""})
        if tag in ("h1", "h2", "h3"):
            self._capture = {"tag": tag, "line": line, "text": []}
        if a.get("id"):
            classes = a.get("class", "")
            sig = f"<{tag} id=\"{a['id']}\"" + (f" class=\"{classes}\"" if classes else "") + ">"
            self.nodes.append({"qname": f"#{a['id']}", "kind": "element", "name": a["id"],
                               "parent": None, "signature": sig, "docstring": "",
                               "source": "", "line_start": line, "line_end": line})
        if tag not in _VOID:
            self._tag_stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        a = dict(attrs)
        if tag == "link" and a.get("href"):
            self.refs.append({"source": "", "kind": "IMPORTS", "target": a["href"],
                              "qualifier": "", "meta": ""})
        elif tag == "img" and a.get("src"):
            self.refs.append({"source": "", "kind": "IMPORTS", "target": a["src"],
                              "qualifier": "", "meta": ""})

    def handle_endtag(self, tag):
        line = self._line()
        if self._script is not None and tag in ("script", "style"):
            s = self._script
            self._script = None
            code = "".join(s["buf"])
            if code.strip():
                lang = "css" if s["tag"] == "style" else "js"
                if s["tag"] == "script" and s["type"] not in ("", "module", "text/javascript"):
                    lang = "json" if "json" in s["type"] else "text"
                end = s["start"] + code.count("\n")
                self.nodes.append({"qname": f"<{s['tag']}>@{s['start']}", "kind": "block",
                                   "name": lang, "parent": None, "signature": lang,
                                   "docstring": "", "source": code.strip("\n"),
                                   "line_start": s["start"], "line_end": max(end, line)})
                if lang == "js":
                    self.scripts.append({"offset": s["start"] - 1, "text": code,
                                         "module": s["type"] == "module"})
        if self._capture is not None and tag == self._capture["tag"]:
            cap = self._capture
            self._capture = None
            title = " ".join("".join(cap["text"]).split())
            if title:
                self.nodes.append({"qname": f"{cap['tag']}:{title}", "kind": "section", "name": title,
                                   "parent": None, "signature": f"<{cap['tag']}> {title}",
                                   "docstring": title, "source": "",
                                   "line_start": cap["line"], "line_end": line})
        if tag in self._tag_stack:
            while self._tag_stack and self._tag_stack.pop() != tag:
                pass

    def handle_data(self, data):
        if self._script is not None:
            self._script["buf"].append(data)
            return
        if self._capture is not None:
            self._capture["text"].append(data)
        if self._tag_stack and self._tag_stack[-1] == "title":
            self.title += data


def parse(src: str, rel_path: str = "") -> dict:
    c = _Collector()
    try:
        c.feed(src)
        c.close()
    except Exception:
        pass
    seen: set[str] = set()
    nodes = []
    for n in c.nodes:
        if n["qname"] in seen:
            n["qname"] = f"{n['qname']}@{n['line_start']}"
        seen.add(n["qname"])
        nodes.append(n)
    return {"file_doc": " ".join(c.title.split()), "nodes": nodes, "refs": c.refs,
            "scripts": c.scripts}
