#!/usr/bin/env node
/**
 * Batch JS/TS extractor for codesearch.
 *
 * stdin : JSON array of jobs  [{id, text, ts, offset}]
 *         ts      true when `text` is TypeScript (types are blanked, line numbers survive)
 *         offset  line offset added to every line (inline <script> blocks in html)
 * stdout: JSON array          [{id, result | error}]
 *
 * `result` follows the same contract as py_parser.py: {file_doc, nodes, refs}.
 * acorn is resolved from the repo's node_modules.
 */
import { parse } from 'acorn';
import { stripTypeScriptTypes } from 'node:module';
import { readFileSync } from 'node:fs';

process.removeAllListeners('warning');

const FN_TYPES = new Set(['FunctionDeclaration', 'FunctionExpression', 'ArrowFunctionExpression']);
// Methods that mutate the container they are called on: x.items.push(v) writes x.items.
const MUTATORS = new Set(['push', 'pop', 'shift', 'unshift', 'splice', 'sort', 'reverse', 'set',
                          'delete', 'add', 'clear', 'assign', 'fill']);

// strip mode blanks types in place, so every offset and line number survives. `transform` mode
// reformats the code and would shift line numbers, so it is deliberately not used as a fallback:
// the few files with enums or parameter properties are reported as parse problems instead.
function tsToJs(text) {
  return stripTypeScriptTypes(text, { mode: 'strip' });
}

function parseSource(js) {
  const comments = [];
  const opts = { ecmaVersion: 'latest', locations: true, allowHashBang: true,
                 allowReturnOutsideFunction: true, onComment: comments };
  try {
    return { ast: parse(js, { ...opts, sourceType: 'module' }), comments };
  } catch (_) {
    comments.length = 0;
    return { ast: parse(js, { ...opts, sourceType: 'script' }), comments };
  }
}

function extract(job) {
  const original = job.text;
  const js = job.ts ? tsToJs(original) : original;
  const { ast, comments } = parseSource(js);
  const off = job.offset || 0;
  const lines = original.split(/\r?\n/);
  const nodes = [];
  const refs = [];
  const attrSeen = new Set();
  const consumedDoc = new Set();

  const L = (n) => n.loc.start.line + off;
  const E = (n) => n.loc.end.line + off;
  const slice = (n) => lines.slice(n.loc.start.line - 1, n.loc.end.line).join('\n');
  const sigOf = (from, body) => original.slice(from, body ? body.start : from).replace(/\s+/g, ' ').trim();

  // JSDoc directly above a node becomes its docstring and is not emitted as a comment node.
  const jsdocFor = (startLine) => {
    for (const c of comments) {
      if (c.type === 'Block' && c.value.startsWith('*') && c.loc.end.line === startLine - 1) {
        consumedDoc.add(c);
        return cleanBlock(c.value);
      }
    }
    return '';
  };

  function cleanBlock(value) {
    return value.split(/\r?\n/).map((l) => l.replace(/^\s*\*+ ?/, '')).join('\n').trim();
  }

  function addNode(n) { nodes.push(n); }

  function scanBody(root, qname, classQ) {
    const calls = new Map();
    const names = new Map();
    const attrs = new Map();
    const objAttrs = new Map();
    const mutated = new Set();   // nodes written through (x[k] = v, x.push(v)): not also a read
    const selfIdent = classQ ? classQ.split('::').pop() : null;
    const isSelfObj = (o) => o && (o.type === 'ThisExpression' || (selfIdent && o.type === 'Identifier' && o.name === selfIdent));
    const noteAttr = (m, mode) => {
      if (!m || m.type !== 'MemberExpression' || m.computed || m.property.type !== 'Identifier') return;
      (isSelfObj(m.object) ? attrs : objAttrs).set(m.property.name + '|' + mode, 1);
    };
    const markMutated = (t) => {
      if (!t) return;
      if (t.type === 'MemberExpression') noteAttr(t, 'w');
      else if (t.type === 'Identifier') names.set(t.name + '|w', 1);
      else return;
      mutated.add(t);
    };
    const visit = (n, ctx) => {
      if (!n || typeof n.type !== 'string') return;
      switch (n.type) {
        case 'CallExpression':
        case 'NewExpression': {
          const c = n.callee;
          if (c.type === 'Identifier') calls.set(c.name + '|', 1);
          else if (c.type === 'MemberExpression' && !c.computed && c.property.type === 'Identifier') {
            const qual = isSelfObj(c.object) ? 'self'
              : (c.object.type === 'Identifier' ? 'recv:' + c.object.name : 'attr');
            calls.set(c.property.name + '|' + qual, 1);
            // x.items.push(...) / registry.set(...) mutate what they are called on
            if (MUTATORS.has(c.property.name)) markMutated(c.object);
          }
          break;
        }
        case 'AssignmentExpression': {
          if (n.left.type === 'Identifier') names.set(n.left.name + '|w', 1);
          else if (n.left.type === 'MemberExpression') {
            noteAttr(n.left, 'w');
            if (n.operator !== '=') noteAttr(n.left, 'r');
            if (n.left.computed) markMutated(n.left.object);
            visit(n.left.object, ctx);
            if (n.left.computed) visit(n.left.property, ctx);
            visit(n.right, ctx);
            return;
          }
          break;
        }
        case 'UpdateExpression': {
          const a = n.argument;
          if (a.type === 'Identifier') names.set(a.name + '|w', 1);
          else if (a.type === 'MemberExpression') {
            noteAttr(a, 'w');
            visit(a.object, ctx);
            return;
          }
          break;
        }
        case 'MemberExpression': {
          if (!mutated.has(n)) noteAttr(n, 'r');
          visit(n.object, ctx);
          if (n.computed) visit(n.property, ctx);
          return;
        }
        case 'Identifier':
          if (!mutated.has(n)) names.set(n.name + '|r', 1);
          return;
        case 'Property':
          if (!n.computed && n.key.type === 'Identifier' && !n.shorthand) { visit(n.value, ctx); return; }
          break;
        case 'MethodDefinition':
        case 'PropertyDefinition':
          if (!n.computed) { visit(n.value, ctx); return; }
          break;
        default:
      }
      for (const k of Object.keys(n)) {
        if (k === 'loc' || k === 'type') continue;
        const v = n[k];
        if (Array.isArray(v)) v.forEach((x) => x && typeof x.type === 'string' && visit(x, ctx));
        else if (v && typeof v.type === 'string') visit(v, ctx);
      }
    };
    visit(root, null);
    for (const key of calls.keys()) {
      const [t, q] = key.split('|');
      refs.push({ source: qname, kind: 'CALLS', target: t, qualifier: q, meta: '' });
    }
    for (const key of names.keys()) {
      const [t, m] = key.split('|');
      refs.push({ source: qname, kind: 'REFERENCES', target: t, qualifier: 'module', meta: m });
    }
    for (const key of objAttrs.keys()) {
      const [t, m] = key.split('|');
      refs.push({ source: qname, kind: 'REFERENCES', target: t, qualifier: 'attr', meta: m });
    }
    if (classQ) {
      for (const key of attrs.keys()) {
        const [t, m] = key.split('|');
        refs.push({ source: qname, kind: 'REFERENCES', target: t, qualifier: 'class:' + classQ, meta: m });
        if (m === 'w' && !attrSeen.has(classQ + '::' + t)) {
          attrSeen.add(classQ + '::' + t);
          addNode({ qname: classQ + '::' + t, kind: 'variable', name: t, parent: classQ, signature: '',
                    docstring: 'instance attribute assigned in ' + qname.split('::').pop(),
                    source: '', line_start: L(root), line_end: L(root) });
        }
      }
    }
  }

  function addFunction(nameStr, fnNode, parentQ, classQ, outer, sigFrom) {
    const qname = parentQ ? parentQ + '::' + nameStr : nameStr;
    const stmt = outer || fnNode;
    addNode({
      qname, kind: classQ ? 'method' : 'function', name: nameStr, parent: parentQ || null,
      signature: nameStr === '' ? '' : sigOf(sigFrom ?? fnNode.start, fnNode.body),
      docstring: jsdocFor(stmt.loc.start.line), source: slice(stmt),
      line_start: L(stmt), line_end: E(stmt),
    });
    scanBody(fnNode.body, qname, classQ);
    // directly nested function declarations become their own nodes
    if (fnNode.body && fnNode.body.type === 'BlockStatement') {
      for (const s of fnNode.body.body) {
        if (s.type === 'FunctionDeclaration' && s.id) addFunction(s.id.name, s, qname, null, null, s.start);
      }
    }
  }

  function addClass(node, parentQ, outer) {
    const name = node.id ? node.id.name : '<anonymous>';
    const qname = parentQ ? parentQ + '::' + name : name;
    const stmt = outer || node;
    const bases = node.superClass && node.superClass.type === 'Identifier' ? [node.superClass.name] : [];
    addNode({
      qname, kind: 'class', name, parent: parentQ || null,
      signature: 'class ' + name + (bases.length ? ' extends ' + bases[0] : ''),
      docstring: jsdocFor(stmt.loc.start.line), source: '', line_start: L(stmt), line_end: E(stmt),
    });
    bases.forEach((b) => refs.push({ source: qname, kind: 'INHERITS', target: b, qualifier: '', meta: '' }));
    for (const m of node.body.body) {
      if (m.type === 'MethodDefinition' && !m.computed) {
        const mname = m.key.name || String(m.key.value);
        addFunction(mname, m.value, qname, qname, m, m.start);
      } else if (m.type === 'PropertyDefinition' && !m.computed && m.value && FN_TYPES.has(m.value.type)) {
        addFunction(m.key.name || String(m.key.value), m.value, qname, qname, m, m.start);
      } else if (m.type === 'PropertyDefinition' && !m.computed && m.key.name) {
        addNode({ qname: qname + '::' + m.key.name, kind: 'variable', name: m.key.name, parent: qname,
                  signature: '', docstring: jsdocFor(m.loc.start.line), source: slice(m),
                  line_start: L(m), line_end: E(m) });
        attrSeen.add(qname + '::' + m.key.name);
      }
    }
  }

  function isMethodObject(expr) {
    return expr && expr.type === 'ObjectExpression' && expr.properties.some(
      (p) => p.type === 'Property' && !p.computed && p.value && FN_TYPES.has(p.value.type)
    );
  }

  function isIIFE(expr) {
    return expr && expr.type === 'CallExpression' && expr.callee && FN_TYPES.has(expr.callee.type)
      && expr.callee.body && expr.callee.body.type === 'BlockStatement';
  }

  function addObjectModule(name, objExpr, parentQ, outer) {
    const qname = parentQ ? parentQ + '::' + name : name;
    const stmt = outer || objExpr;
    addNode({
      qname, kind: 'class', name, parent: parentQ || null,
      signature: 'namespace ' + name,
      docstring: jsdocFor(stmt.loc.start.line), source: '', line_start: L(stmt), line_end: E(stmt),
    });
    for (const p of objExpr.properties) {
      if (p.type !== 'Property' || p.computed) continue;
      const pname = p.key.name || String(p.key.value);
      if (p.value && FN_TYPES.has(p.value.type)) {
        addFunction(pname, p.value, qname, qname, p, p.start);
      } else if (pname) {
        addNode({ qname: qname + '::' + pname, kind: 'variable', name: pname, parent: qname,
                  signature: '', docstring: jsdocFor(p.loc.start.line), source: slice(p),
                  line_start: L(p), line_end: E(p) });
        attrSeen.add(qname + '::' + pname);
      }
    }
  }

  function addIIFEModule(name, callExpr, parentQ, outer) {
    const qname = name ? (parentQ ? parentQ + '::' + name : name) : parentQ;
    const stmt = outer || callExpr;
    if (name) {
      addNode({
        qname, kind: 'class', name, parent: parentQ || null,
        signature: 'module ' + name,
        docstring: jsdocFor(stmt.loc.start.line), source: '', line_start: L(stmt), line_end: E(stmt),
      });
    }
    const defined = new Set();
    for (const s of callExpr.callee.body.body) {
      scopeStmt(s, null, qname, qname || null);
      if (s.type === 'FunctionDeclaration' && s.id) defined.add(s.id.name);
      if (s.type === 'ReturnStatement' && s.argument && s.argument.type === 'ObjectExpression') {
        for (const p of s.argument.properties) {
          if (p.type === 'Property' && !p.computed && !p.shorthand && p.value && FN_TYPES.has(p.value.type)) {
            const pname = p.key.name || String(p.key.value);
            if (!defined.has(pname)) addFunction(pname, p.value, qname, qname || null, p, p.start);
          }
        }
      }
    }
  }

  function scopeStmt(stmt, outer, parentQ = '', classQ = null) {
    switch (stmt.type) {
      case 'ImportDeclaration':
        refs.push({ source: '', kind: 'IMPORTS', target: stmt.source.value, qualifier: '',
                    meta: stmt.specifiers.map((s) => (s.imported && s.imported.name) || s.local.name).join(',') });
        break;
      case 'ExportAllDeclaration':
        refs.push({ source: '', kind: 'IMPORTS', target: stmt.source.value, qualifier: '', meta: '*' });
        break;
      case 'ExportNamedDeclaration':
        if (stmt.source) refs.push({ source: '', kind: 'IMPORTS', target: stmt.source.value, qualifier: '', meta: '' });
        if (stmt.declaration) scopeStmt(stmt.declaration, stmt, parentQ, classQ);
        break;
      case 'ExportDefaultDeclaration':
        if (stmt.declaration && ['FunctionDeclaration', 'ClassDeclaration'].includes(stmt.declaration.type)
            && stmt.declaration.id) scopeStmt(stmt.declaration, stmt, parentQ, classQ);
        break;
      case 'FunctionDeclaration':
        if (stmt.id) addFunction(stmt.id.name, stmt, parentQ, classQ, outer, stmt.start);
        break;
      case 'ClassDeclaration':
        addClass(stmt, parentQ, outer);
        break;
      case 'VariableDeclaration': {
        const holder = outer || stmt;
        for (const d of stmt.declarations) {
          if (d.id.type !== 'Identifier') continue;
          if (d.init && (d.init.type === 'ArrowFunctionExpression' || d.init.type === 'FunctionExpression')) {
            addFunction(d.id.name, d.init, parentQ, classQ, holder, holder.start);
          } else if (d.init && d.init.type === 'ClassExpression') {
            addClass(Object.assign({}, d.init, { id: d.id }), parentQ, holder);
          } else if (isMethodObject(d.init)) {
            addObjectModule(d.id.name, d.init, parentQ, holder);
          } else if (isIIFE(d.init)) {
            addIIFEModule(d.id.name, d.init, parentQ, holder);
          } else {
            const vq = parentQ ? parentQ + '::' + d.id.name : d.id.name;
            addNode({ qname: vq, kind: 'variable', name: d.id.name, parent: parentQ || null,
                      signature: stmt.kind, docstring: jsdocFor(holder.loc.start.line),
                      source: slice(holder), line_start: L(holder), line_end: E(holder) });
          }
        }
        break;
      }
      case 'ExpressionStatement': {
        const e = stmt.expression;
        if (isIIFE(e)) {
          if (e.callee.id && e.callee.id.name) {
            addFunction(e.callee.id.name, e.callee, parentQ, classQ, stmt, e.callee.start);
          } else {
            addIIFEModule('', e, parentQ, stmt);
            scanBody(e.callee.body, parentQ || '', classQ);
          }
        } else if (e.type === 'AssignmentExpression' && e.left.type === 'MemberExpression' && !e.left.computed
            && e.left.property.type === 'Identifier') {
          const propName = e.left.property.name;
          const obj = e.left.object.type === 'Identifier' && e.left.object.name !== 'window' ? e.left.object.name : parentQ;
          if (e.right && FN_TYPES.has(e.right.type)) {
            addFunction(propName, e.right, obj, obj || null, stmt, stmt.start);
          } else if (isMethodObject(e.right)) {
            addObjectModule(propName, e.right, obj, stmt);
          } else if (isIIFE(e.right)) {
            addIIFEModule(propName, e.right, obj, stmt);
          }
        }
        break;
      }
      default:
    }
  }

  for (const stmt of ast.body) scopeStmt(stmt, null, '', null);

  // comments: group adjacent full-line // comments, keep block comments whole
  const out = [];
  let cur = null;
  for (const c of comments) {
    if (consumedDoc.has(c)) { cur = null; continue; }
    const startLine = c.loc.start.line;
    const before = (lines[startLine - 1] || '').slice(0, c.loc.start.column);
    const fullLine = before.trim() === '';
    if (c.type === 'Line') {
      const text = c.value.trim();
      if (fullLine && cur && cur.full && cur.type === 'Line' && startLine === cur.end + 1) {
        cur.text.push(text); cur.end = startLine; continue;
      }
      cur = { type: 'Line', start: startLine, end: startLine, text: [text], full: fullLine };
      out.push(cur);
    } else {
      cur = null;
      out.push({ type: 'Block', start: startLine, end: c.loc.end.line, text: [cleanBlock(c.value)], full: fullLine });
    }
  }
  for (const c of out) {
    const body = c.text.join('\n').trim();
    if (!body) continue;
    addNode({ qname: '<comment>@' + (c.start + off), kind: 'comment', name: '', parent: null,
              signature: '', docstring: body, source: '', line_start: c.start + off, line_end: c.end + off });
  }
  return { file_doc: '', nodes, refs };
}

const jobs = JSON.parse(readFileSync(0, 'utf8'));
const results = jobs.map((job) => {
  try {
    return { id: job.id, result: extract(job) };
  } catch (e) {
    return { id: job.id, error: String(e && e.message ? e.message : e) };
  }
});
process.stdout.write(JSON.stringify(results));
