"""Python extractor: stdlib `ast` for structure, `tokenize` for comments.

Output contract (shared by every parser):
    {"file_doc": str,
     "nodes": [{"qname", "kind", "name", "parent", "signature", "docstring",
                "source", "line_start", "line_end"}],
     "refs":  [{"source", "kind", "target", "qualifier", "meta"}]}

`source`/`parent` use local qualified names ("Class::method"). The indexer prefixes
the file path. A ref with source "" belongs to the file node.
"""
from __future__ import annotations

import ast
import io
import textwrap
import tokenize

# Methods that mutate the container they are called on: `obj.items.append(x)` is a
# write to `obj.items`, not a read.
MUTATORS = {"append", "extend", "insert", "pop", "popitem", "remove", "discard", "add", "clear",
            "update", "setdefault", "sort", "reverse", "appendleft", "popleft"}


def _doc(node) -> str:
    try:
        return textwrap.dedent(ast.get_docstring(node, clean=False) or "").strip()
    except Exception:
        return ""


def _signature(node) -> str:
    a = node.args
    parts: list[str] = []
    positional = list(getattr(a, "posonlyargs", [])) + list(a.args)
    defaults = [None] * (len(positional) - len(a.defaults)) + list(a.defaults)
    for arg, default in zip(positional, defaults):
        s = arg.arg
        if arg.annotation is not None:
            s += ": " + ast.unparse(arg.annotation)
        if default is not None:
            s += " = " + ast.unparse(default)
        parts.append(s)
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        parts.append("*")
    for arg, default in zip(a.kwonlyargs, a.kw_defaults):
        s = arg.arg
        if arg.annotation is not None:
            s += ": " + ast.unparse(arg.annotation)
        if default is not None:
            s += " = " + ast.unparse(default)
        parts.append(s)
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    sig = f"{prefix}{node.name}({', '.join(parts)})"
    if node.returns is not None:
        sig += " -> " + ast.unparse(node.returns)
    return sig


class _BodyScan(ast.NodeVisitor):
    """Collect calls, name references and self.attr accesses inside one function.
    Nested function/class definitions are scanned too: their behavior is part of
    the enclosing function's footprint."""

    def __init__(self, self_name: str | None):
        self.self_name = self_name
        self.calls: dict[tuple[str, str], None] = {}      # (name, qualifier)
        self.names: dict[tuple[str, str], None] = {}      # (name, "r"/"w")
        self.attrs: dict[tuple[str, str], None] = {}      # (attr, "r"/"w") on self
        self.obj_attrs: dict[tuple[str, str], None] = {}  # (attr, "r"/"w") on any other object
        self.stores: set[str] = set()
        self.globals: set[str] = set()
        self._mutated: set[int] = set()

    def visit_Global(self, node):
        self.globals.update(node.names)

    def visit_arg(self, node):
        self.stores.add(node.arg)

    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in MUTATORS:
            self._mark_mutated(f.value)             # x.items.append(...) / ledger.update(...)
        if isinstance(f, ast.Name):
            self.calls[(f.id, "")] = None
        elif isinstance(f, ast.Attribute):
            if isinstance(f.value, ast.Name) and f.value.id == self.self_name:
                self.calls[(f.attr, "self")] = None
            elif isinstance(f.value, ast.Name):
                self.calls[(f.attr, f"recv:{f.value.id}")] = None
            else:
                self.calls[(f.attr, "attr")] = None
        self.generic_visit(node)

    def visit_Name(self, node):
        if id(node) in self._mutated:
            return
        mode = "w" if isinstance(node.ctx, (ast.Store, ast.Del)) else "r"
        if mode == "w":
            self.stores.add(node.id)
        self.names[(node.id, mode)] = None

    def _note_attr(self, node: ast.Attribute, mode: str) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == self.self_name:
            self.attrs[(node.attr, mode)] = None
        else:
            self.obj_attrs[(node.attr, mode)] = None

    def _mark_mutated(self, target) -> None:
        """Record a write to `target` (Attribute or Name) and suppress the incidental
        read that visiting the same node would otherwise add."""
        if isinstance(target, ast.Attribute):
            self._note_attr(target, "w")
        elif isinstance(target, ast.Name):
            self.names[(target.id, "w")] = None
        else:
            return
        self._mutated.add(id(target))

    def visit_Attribute(self, node):
        if id(node) not in self._mutated:
            mode = "w" if isinstance(node.ctx, (ast.Store, ast.Del)) else "r"
            self._note_attr(node, mode)
        self.generic_visit(node)

    def visit_Subscript(self, node):
        # `x.state[key] = v` mutates x.state even though the Attribute itself is a Load
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._mark_mutated(node.value)
        self.generic_visit(node)


def _comments(src: str) -> list[dict]:
    """Group consecutive full-line comments into blocks; trailing comments stand alone."""
    toks: list[tuple[int, str, bool]] = []
    try:
        lines = src.splitlines()
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                row = tok.start[0]
                full_line = lines[row - 1][: tok.start[1]].strip() == "" if row - 1 < len(lines) else True
                text = tok.string.lstrip("#").strip()
                if row == 1 and (tok.string.startswith("#!") or "coding" in tok.string):
                    continue
                toks.append((row, text, full_line))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    blocks: list[dict] = []
    cur: dict | None = None
    for row, text, full_line in toks:
        if full_line and cur is not None and cur["full"] and row == cur["end"] + 1:
            cur["text"].append(text)
            cur["end"] = row
            continue
        cur = {"start": row, "end": row, "text": [text], "full": full_line}
        blocks.append(cur)
    out = []
    for b in blocks:
        body = "\n".join(b["text"]).strip()
        if not body:
            continue
        out.append({"qname": f"<comment>@{b['start']}", "kind": "comment", "name": "",
                    "parent": None, "signature": "", "docstring": body, "source": "",
                    "line_start": b["start"], "line_end": b["end"]})
    return out


def parse(src: str, rel_path: str = "") -> dict:
    result = {"file_doc": "", "nodes": [], "refs": []}
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return result
    lines = src.splitlines()
    result["file_doc"] = _doc(tree)
    nodes, refs = result["nodes"], result["refs"]
    seen_attrs: set[str] = set()

    def snippet(node) -> str:
        start = node.lineno - 1
        if getattr(node, "decorator_list", None):
            start = min(start, node.decorator_list[0].lineno - 1)
        return "\n".join(lines[start: node.end_lineno])

    def add_variable(qname, name, parent, line, annotation="", doc="", src_line=""):
        nodes.append({"qname": qname, "kind": "variable", "name": name, "parent": parent,
                      "signature": annotation, "docstring": doc, "source": src_line,
                      "line_start": line, "line_end": line})

    def handle_function(node, parent_q, class_q, self_name):
        qname = f"{parent_q}::{node.name}" if parent_q else node.name
        kind = "method" if class_q else "function"
        decos = [ast.unparse(d) for d in node.decorator_list]
        sig = _signature(node)
        if decos:
            sig = " ".join("@" + d for d in decos) + " " + sig
        nodes.append({"qname": qname, "kind": kind, "name": node.name, "parent": parent_q or None,
                      "signature": sig, "docstring": _doc(node), "source": snippet(node),
                      "line_start": node.lineno, "line_end": node.end_lineno or node.lineno})
        scan = _BodyScan(self_name)
        scan.visit(node.args)
        for stmt in node.body:
            scan.visit(stmt)
        # A name assigned (or a parameter) in the function is a local unless declared
        # global: it is not a reference to the module-level variable of the same name.
        local_names = scan.stores - scan.globals
        for name, qual in scan.calls:
            refs.append({"source": qname, "kind": "CALLS", "target": name,
                         "qualifier": qual, "meta": ""})
        for name, mode in scan.names:
            if name in local_names:
                continue
            refs.append({"source": qname, "kind": "REFERENCES", "target": name,
                         "qualifier": "module", "meta": mode})
        for attr, mode in scan.obj_attrs:
            refs.append({"source": qname, "kind": "REFERENCES", "target": attr,
                         "qualifier": "attr", "meta": mode})
        if class_q:
            for attr, mode in scan.attrs:
                refs.append({"source": qname, "kind": "REFERENCES", "target": attr,
                             "qualifier": "class:" + class_q, "meta": mode})
                if mode == "w" and f"{class_q}::{attr}" not in seen_attrs:
                    seen_attrs.add(f"{class_q}::{attr}")
                    add_variable(f"{class_q}::{attr}", attr, class_q, node.lineno,
                                 doc="instance attribute assigned in " + node.name)
        # directly nested defs are surfaced as their own nodes
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                handle_function(stmt, qname, None, None)

    def handle_class(node, parent_q):
        qname = f"{parent_q}::{node.name}" if parent_q else node.name
        bases = []
        for b in node.bases:
            try:
                bases.append(ast.unparse(b))
            except Exception:
                pass
        decos = [ast.unparse(d) for d in node.decorator_list]
        sig = ("@" + " @".join(decos) + " " if decos else "") + f"class {node.name}" + \
              (f"({', '.join(bases)})" if bases else "")
        nodes.append({"qname": qname, "kind": "class", "name": node.name, "parent": parent_q or None,
                      "signature": sig, "docstring": _doc(node), "source": "",
                      "line_start": node.lineno, "line_end": node.end_lineno or node.lineno})
        for base in bases:
            refs.append({"source": qname, "kind": "INHERITS", "target": base.split(".")[-1],
                         "qualifier": "", "meta": ""})
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                first = stmt.args.args[0].arg if stmt.args.args else None
                is_static = any(isinstance(d, ast.Name) and d.id == "staticmethod"
                                for d in stmt.decorator_list)
                handle_function(stmt, qname, qname, None if is_static else first)
            elif isinstance(stmt, ast.ClassDef):
                handle_class(stmt, qname)
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                handle_assign(stmt, qname)

    def handle_assign(stmt, parent_q):
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        annotation = ast.unparse(stmt.annotation) if isinstance(stmt, ast.AnnAssign) else ""
        for t in targets:
            if isinstance(t, ast.Name):
                qname = f"{parent_q}::{t.id}" if parent_q else t.id
                add_variable(qname, t.id, parent_q or None, stmt.lineno, annotation,
                             src_line=lines[stmt.lineno - 1].strip() if stmt.lineno - 1 < len(lines) else "")
                nodes[-1]["line_end"] = stmt.end_lineno or stmt.lineno

    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            handle_function(stmt, "", None, None)
        elif isinstance(stmt, ast.ClassDef):
            handle_class(stmt, "")
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            handle_assign(stmt, "")
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                refs.append({"source": "", "kind": "IMPORTS", "target": alias.name,
                             "qualifier": "0", "meta": ""})
        elif isinstance(stmt, ast.ImportFrom):
            refs.append({"source": "", "kind": "IMPORTS", "target": stmt.module or "",
                         "qualifier": str(stmt.level or 0),
                         "meta": ",".join(a.name for a in stmt.names)})

    nodes.extend(_comments(src))
    return result
