"""Single-source renderer: one block list -> IEEE LaTeX (A4, pdflatex) and IEEE-style DOCX."""
import re, io, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- inline markup
# source markup:  **bold**  *italic*  $math$  [@k1;@k2] citations  unicode dashes/symbols
GREEK = {r'\mu':'μ',r'\sigma':'σ',r'\tau':'τ',r'\theta':'θ',r'\pm':'±',r'\ge':'≥',r'\le':'≤',r'\approx':'≈',
         r'\times':'×',r'\langle':'⟨',r'\rangle':'⟩',r'\rho':'ρ',r'\ell':'ℓ',r'\cdot':'·',r'\pi':'π',r'\lambda':'λ',
         r'\Delta':'Δ',r'\in':'∈',r'\|':'‖',r'\ln':'ln',r'\max':'max',r'\sum':'Σ',r'\dots':'…',r'\ldots':'…',r'\to':'→'}

def tex_text(s, cites):
    """convert source string to LaTeX"""
    out = []
    i = 0
    s = re.sub(r'\\figref\{([\w:]+)\}', lambda m: '\x03F' + m.group(1) + '\x04', s)
    s = re.sub(r'\\tabref\{([\w:]+)\}', lambda m: '\x03T' + m.group(1) + '\x04', s)
    s = re.sub(r'\\eqref\{([\w:]+)\}', lambda m: '\x03E' + m.group(1) + '\x04', s)
    # protect math
    parts = re.split(r'(\$[^$]+\$)', s)
    for p in parts:
        if p.startswith('$') and p.endswith('$') and len(p) > 1:
            out.append(p); continue
        p = p.replace('\\', r'\textbackslash{}')
        for a, b in (('&', r'\&'), ('%', r'\%'), ('#', r'\#'), ('_', r'\_')):
            p = p.replace(a, b)
        p = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', p)
        p = re.sub(r'\*(.+?)\*', r'\\textit{\1}', p)
        def cite(m):
            ks = [k.strip().lstrip('@') for k in m.group(1).split(';')]
            return r'\cite{' + ','.join(ks) + '}'
        p = re.sub(r'\[((?:@[\w]+;?\s*)+)\]', cite, p)
        for a, b in (('‖', r'$\|$'), ('½', r'$\frac12$'), ('⟨', r'$\langle$'), ('⟩', r'$\rangle$'), ('⁻³', r'$^{-3}$'), ('³', r'$^3$'), ('²', r'$^2$'), ('⁴', r'$^4$'), ('ⁿ', r'$^n$'), ('ᵀ', r'$^{\top}$'), ('ỹ', r'$\tilde y$'), ('ᵢ', r'$_i$'), ('₁', r'$_1$'), ('₂', r'$_2$'), ('∈', r'$\in$'), ('→', r'$\to$'), ('Σ', r'$\Sigma$'), ('π', r'$\pi$'), ('Φ', r'$\Phi$'), ('σ', r'$\sigma$'), ('τ', r'$\tau$'), ('θ', r'$\theta$'), ('φ', r'$\varphi$'), ('Δ', r'$\Delta$'), ('…', r'\ldots{}'), ('ε', r'$\epsilon$'), ('–', '--'), ('—', '---'), ('’', "'"), ('“', '``'), ('”', "''"), ('±', r'$\pm$'), ('×', r'$\times$'),
                     ('≈', r'$\approx$'), ('≥', r'$\geq$'), ('≤', r'$\leq$'), ('μ', r'$\mu$'), ('é', r"\'e"), ('í', r"\'{\i}"), ('č', r'\v{c}'),
                     ('ü', r'\"u'), ('ó', r"\'o"), ('Ö', r'\"O'), ('−', '$-$'), ('·', r'$\cdot$')):
            p = p.replace(a, b)
        out.append(p)
    res = ''.join(out)
    res = re.sub('\x03F([\\w:]+)\x04', lambda m: 'Fig.~\\ref{' + m.group(1) + '}', res)
    res = re.sub('\x03T([\\w:]+)\x04', lambda m: 'Table~\\ref{' + m.group(1) + '}', res)
    res = re.sub('\x03E([\\w:]+)\x04', lambda m: '(\\ref{' + m.group(1) + '})', res)
    return res

def runs_docx(s):
    """convert source string to list of (text, style-dict) runs for docx; cites resolved by caller beforehand"""
    runs = []
    # split into math / non-math
    for part in re.split(r'(\$[^$]+\$)', s):
        if part.startswith('$') and part.endswith('$') and len(part) > 1:
            runs += math_runs(part[1:-1])
        else:
            runs += md_runs(part)
    return runs

def md_runs(s):
    runs = []
    pos = 0
    for m in re.finditer(r'\*\*(.+?)\*\*|\*(.+?)\*', s):
        if m.start() > pos: runs.append((s[pos:m.start()], {}))
        if m.group(1) is not None: runs.append((m.group(1), {'b': True}))
        else: runs.append((m.group(2), {'i': True}))
        pos = m.end()
    if pos < len(s): runs.append((s[pos:], {}))
    return runs

def math_runs(m):
    for a, b in GREEK.items(): m = m.replace(a, b)
    m = re.sub(r'\\mathbf\{(\w+)\}', lambda x: '\x01' + x.group(1) + '\x02', m)
    m = re.sub(r'\\mathrm\{(\w+)\}', r'\1', m)
    m = re.sub(r'\\text\{([^}]*)\}', r'\1', m)
    m = m.replace('{', '').replace('}', '') if '_' not in m and '^' not in m else m
    runs = []; i = 0; buf = ''; bold = False
    def flush():
        nonlocal buf
        if buf: runs.append((buf, {'i': True, 'b': bold})); buf = ''
    while i < len(m):
        c = m[i]
        if c == '\x01': flush(); bold = True; i += 1; continue
        if c == '\x02': flush(); bold = False; i += 1; continue
        if c in '_^':
            flush(); i += 1
            if i < len(m) and m[i] == '{':
                j = m.index('}', i); arg = m[i+1:j]; i = j + 1
            else: arg = m[i]; i += 1
            for a, b in GREEK.items(): arg = arg.replace(a, b)
            runs.append((arg.replace('\\', ''), {'sub' if c == '_' else 'sup': True})); continue
        if c in '{}': i += 1; continue
        buf += c; i += 1
    flush(); return runs

# ---------------------------------------------------------------- references
def order_cites(blocks):
    order = []
    def scan(s):
        for m in re.finditer(r'\[((?:@[\w]+;?\s*)+)\]', s):
            for k in m.group(1).split(';'):
                k = k.strip().lstrip('@')
                if k and k not in order: order.append(k)
    for b in blocks:
        for v in b.values():
            if isinstance(v, str): scan(v)
            elif isinstance(v, list):
                for r in v:
                    if isinstance(r, str): scan(r)
                    elif isinstance(r, (list, tuple)):
                        for c in r:
                            if isinstance(c, str): scan(c)
    return order

def cite_numbers(s, order):
    def f(m):
        ks = [k.strip().lstrip('@') for k in m.group(1).split(';')]
        nums = sorted(order.index(k) + 1 for k in ks)
        return '[' + ', '.join(str(n) for n in nums) + ']'
    return re.sub(r'\[((?:@[\w]+;?\s*)+)\]', f, s)

# ---------------------------------------------------------------- LaTeX renderer
def render_tex(meta, blocks, refs, order, path):
    L = []
    L.append(r'\documentclass[conference,a4paper]{IEEEtran}')
    L.append(r'\usepackage[T1]{fontenc}\usepackage{cite,amsmath,amssymb,graphicx,booktabs,url,textcomp,microtype}')
    L.append(r'\graphicspath{{../figures/}}\hyphenation{Para-meter-ised}')
    L.append(r'\setlength{\textfloatsep}{6pt plus 2pt minus 2pt}\setlength{\floatsep}{6pt plus 2pt minus 2pt}\setlength{\intextsep}{6pt plus 2pt minus 2pt}')
    L.append(r'\begin{document}')
    L.append(r'\title{' + tex_text(meta['title'], order) + '}')
    L.append(r'\author{\IEEEauthorblockN{' + tex_text(meta['authors'], order) + r'}\IEEEauthorblockA{' + r'\\'.join(tex_text(x.strip(), order) for x in meta['affil'].split('|')) + '}}')
    L.append(r'\maketitle')
    L.append(r'\begin{abstract}' + tex_text(meta['abstract'], order) + r'\end{abstract}')
    L.append(r'\begin{IEEEkeywords}' + tex_text(meta['keywords'], order) + r'\end{IEEEkeywords}')
    for b in blocks:
        t = b['t']
        if t == 'h1': L.append(r'\section{' + tex_text(b['text'], order) + '}')
        elif t == 'h2': L.append(r'\subsection{' + tex_text(b['text'], order) + '}')
        elif t == 'p': L.append(tex_text(b['text'], order) + '\n')
        elif t == 'eq': L.append(r'\begin{equation}' + b['tex'] + (r'\label{' + b['label'] + '}' if b.get('label') else '') + r'\end{equation}')
        elif t == 'fig':
            env = 'figure*' if b['w'] == 'wide' else 'figure'
            parts = b['files']; n = len(parts)
            body = []
            for f, frac in parts:
                body.append(r'\includegraphics[width=%s]{%s}' % ((r'%.3f\textwidth' % frac) if b['w'] == 'wide' else (r'%.3f\columnwidth' % frac), f))
            L.append(r'\begin{%s}[!t]\centering %s\caption{%s}\label{%s}\end{%s}' % (env, (r'\\[2pt]' if b.get('stack') else r'\hfill ').join(body) if n > 1 else body[0], tex_text(b['cap'], order), b['label'], env))
        elif t == 'table':
            env = 'table*' if b['w'] == 'wide' else 'table'
            cols = b['cols']; rows = b['rows']
            L.append(r'\begin{%s}[!t]\centering\caption{%s}\label{%s}\scriptsize\setlength{\tabcolsep}{%s}' % (env, tex_text(b['cap'], order), b['label'], b.get('tabcolsep', '3pt')))
            L.append(r'\begin{tabular}{%s}\toprule' % cols)
            L.append(' & '.join(tex_text(c, order) for c in b['head']) + r'\\\midrule')
            for r in rows:
                if r == '--': L.append(r'\midrule'); continue
                L.append(' & '.join(tex_text(str(c), order) for c in r) + r'\\')
            L.append(r'\bottomrule\end{tabular}\end{%s}' % env)
    L.append(r'\begin{thebibliography}{99}\footnotesize')
    for k in order:
        L.append(r'\bibitem{%s} %s' % (k, tex_text(refs[k], order).replace('doi: ', 'doi: ')))
    L.append(r'\end{thebibliography}\end{document}')
    open(path, 'w').write('\n'.join(L))

# ---------------------------------------------------------------- DOCX renderer
def eq_png(tex, path):
    fig = plt.figure(figsize=(4, 0.5)); fig.text(0, 0.5, '$' + tex + '$', fontsize=12, va='center')
    fig.savefig(path, dpi=300, bbox_inches='tight', transparent=False, facecolor='white'); plt.close(fig)

def render_docx(meta, blocks, refs, order, path, figdir, tmpdir):
    from docx import Document
    from docx.shared import Pt, Inches, Mm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.section import WD_SECTION
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    doc = Document()
    st = doc.styles['Normal']; st.font.name = 'Times New Roman'; st.font.size = Pt(10)
    st.element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')
    st.paragraph_format.space_after = Pt(0); st.paragraph_format.space_before = Pt(0)
    sec = doc.sections[0]
    sec.page_width = Mm(210); sec.page_height = Mm(297)
    sec.left_margin = sec.right_margin = Mm(15.9); sec.top_margin = Mm(17); sec.bottom_margin = Mm(17)
    def set_cols(section, n):
        pr = section._sectPr
        for c in pr.findall(qn('w:cols')): pr.remove(c)
        c = OxmlElement('w:cols'); c.set(qn('w:num'), str(n)); c.set(qn('w:space'), '283' if n == 2 else '0'); pr.append(c)
    def para(runs, size=10, align=WD_ALIGN_PARAGRAPH.JUSTIFY, bold=False, italic=False, before=0, after=2.5, indent=True, keep=False):
        p = doc.add_paragraph(); p.alignment = align
        pf = p.paragraph_format; pf.space_before = Pt(before); pf.space_after = Pt(after)
        if indent: pf.first_line_indent = Mm(4.2)
        if keep: pf.keep_with_next = True
        for text, sty in runs:
            r = p.add_run(text); r.font.size = Pt(size); r.font.name = 'Times New Roman'
            r.bold = bold or sty.get('b', False); r.italic = italic or sty.get('i', False)
            if sty.get('sub'): r.font.subscript = True
            if sty.get('sup'): r.font.superscript = True
        return p
    def R(s): return runs_docx(cite_numbers(s, order))
    # title block (single column)
    set_cols(sec, 1)
    para([(meta['title'], {})], size=20, align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, indent=False, after=6)
    para([(meta['authors'], {})], size=11, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, after=0)
    para([(meta['affil'].replace(' | ', ', '), {'i': True})], size=9, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, after=8)
    # two-column body
    s2 = doc.add_section(WD_SECTION.CONTINUOUS); set_cols(s2, 2)
    p = para([('Abstract—', {'b': True, 'i': True})] + [(t, dict(st, b=True)) for t, st in R(meta['abstract'])], size=9, after=3, indent=False)
    para([('Index Terms—', {'b': True, 'i': True}), (meta['keywords'], {'i': True})], size=9, after=6, indent=False)
    counters = {'sec': 0, 'sub': 0, 'fig': 0, 'tab': 0, 'eq': 0}
    ROM = ['I','II','III','IV','V','VI','VII','VIII','IX','X','XI','XII','XIII']
    refmap = {}
    # pre-number figs/tables/eqs for cross refs
    n_f = n_t = n_e = 0
    for b in blocks:
        if b['t'] == 'fig': n_f += 1; refmap[b['label']] = 'Fig. %d' % n_f
        if b['t'] == 'table': n_t += 1; refmap[b['label']] = 'Table %s' % ROM[n_t - 1]
        if b['t'] == 'eq' and b.get('label'): n_e += 1; refmap[b['label']] = '(%d)' % n_e
    def xref(s):
        s = re.sub(r'\\ref\{([\w:]+)\}', lambda m: refmap[m.group(1)].replace('Fig. ', '').replace('Table ', '') if False else refmap[m.group(1)], s)
        return s
    def srcx(s):  # resolve \figref{lab} / \tabref{lab} / \eqref{lab}
        s = re.sub(r'\\figref\{([\w:]+)\}', lambda m: refmap[m.group(1)], s)
        s = re.sub(r'\\tabref\{([\w:]+)\}', lambda m: refmap[m.group(1)], s)
        s = re.sub(r'\\eqref\{([\w:]+)\}', lambda m: refmap[m.group(1)], s)
        return s
    in_wide = False
    def go_wide():
        nonlocal in_wide
        if not in_wide:
            sx = doc.add_section(WD_SECTION.CONTINUOUS); set_cols(sx, 1); in_wide = True
    def go_cols():
        nonlocal in_wide
        if in_wide:
            sx = doc.add_section(WD_SECTION.CONTINUOUS); set_cols(sx, 2); in_wide = False
    def shade_borders(tbl, header=True):
        tblPr = tbl._tbl.tblPr
        b = OxmlElement('w:tblBorders')
        for side, val in (('top', 8), ('bottom', 8), ('insideH', 0)):
            e = OxmlElement('w:' + side); e.set(qn('w:val'), 'single' if val else 'nil'); e.set(qn('w:sz'), str(val)); e.set(qn('w:color'), '000000'); b.append(e)
        tblPr.append(b)
    sec_i = 0; sub_i = 0; fi = 0; ti = 0
    for b in blocks:
        t = b['t']
        if t == 'h1':
            go_cols(); sec_i += 1; sub_i = 0
            para([(ROM[sec_i - 1] + '. ' + b['text'].upper(), {})], size=10, align=WD_ALIGN_PARAGRAPH.CENTER, before=4, after=2, indent=False, keep=True)
        elif t == 'h2':
            go_cols(); sub_i += 1
            para([(chr(64 + sub_i) + '. ' + b['text'], {'i': True})], size=10, align=WD_ALIGN_PARAGRAPH.LEFT, before=3, after=2, indent=False, keep=True)
        elif t == 'p':
            go_cols(); para(R(srcx(b['text'])), size=10)
        elif t == 'eq':
            go_cols(); fn = os.path.join(tmpdir, 'eq%d.png' % id(b)); eq_png(b['mpl'], fn)
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(3)
            p.add_run().add_picture(fn, width=Inches(min(2.9, b.get('docx_w', 2.2) * 0.85)))
            if b.get('label'):
                r = p.add_run('   ' + refmap[b['label']]); r.font.size = Pt(10)
        elif t == 'fig':
            fi += 1
            if b['w'] == 'wide': go_wide()
            else: go_cols()
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before = Pt(3); p.paragraph_format.keep_with_next = True
            total_w = (7.1 if b['w'] == 'wide' else 3.45) * 0.85
            for f, frac in b['files']:
                p.add_run().add_picture(os.path.join(figdir, f + '.png'), width=Inches(total_w * frac))
                if len(b['files']) > 1: p.add_run('  ')
            para(R(srcx('Fig. %d. ' % fi + b['cap'])), size=8, align=WD_ALIGN_PARAGRAPH.CENTER if len(b['cap']) < 95 else WD_ALIGN_PARAGRAPH.JUSTIFY, after=5, indent=False)
        elif t == 'table':
            ti += 1
            if b['w'] == 'wide': go_wide()
            else: go_cols()
            para(R(srcx('TABLE %s. ' % ROM[ti - 1] + b['cap'])), size=8, align=WD_ALIGN_PARAGRAPH.CENTER, before=3, after=2, indent=False, keep=True)
            ncol = len(b['head']); tbl = doc.add_table(rows=0, cols=ncol); tbl.alignment = 1
            tbl.autofit = False
            lay = OxmlElement('w:tblLayout'); lay.set(qn('w:type'), 'fixed'); tbl._tbl.tblPr.append(lay)
            tot = 7.0 if b['w'] == 'wide' else 3.4
            ws = b.get('colw') or ([1.5] + [1.0] * (ncol - 1))
            ws = [tot * x / sum(ws) for x in ws]
            for j_, col_ in enumerate(tbl.columns): col_.width = Inches(ws[j_])
            tbl.alignment = 0 if b['w'] != 'wide' else 1
            mar = OxmlElement('w:tblCellMar')
            for side in ('left', 'right'):
                e = OxmlElement('w:' + side); e.set(qn('w:w'), '35'); e.set(qn('w:type'), 'dxa'); mar.append(e)
            tbl._tbl.tblPr.append(mar)
            shade_borders(tbl)
            def addrow(cells, bold=False, rule=False):
                row = tbl.add_row()
                for j, c in enumerate(cells):
                    cell = row.cells[j]; cell.width = Inches(ws[j]); cp = cell.paragraphs[0]
                    cp.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
                    for text, sty in runs_docx(cite_numbers(srcx(str(c)), order)):
                        r = cp.add_run(text); r.font.size = Pt(7); r.bold = bold or sty.get('b', False); r.italic = sty.get('i', False)
                        if sty.get('sub'): r.font.subscript = True
                        if sty.get('sup'): r.font.superscript = True
                    cp.paragraph_format.space_after = Pt(0)
                    if rule:
                        tcPr = cell._tc.get_or_add_tcPr(); bd = OxmlElement('w:tcBorders'); e = OxmlElement('w:bottom'); e.set(qn('w:val'), 'single'); e.set(qn('w:sz'), '4'); e.set(qn('w:color'), '000000'); bd.append(e); tcPr.append(bd)
            addrow(b['head'], bold=True, rule=True)
            for r in b['rows']:
                if r == '--': continue
                addrow(r)
            sp = doc.add_paragraph(); sp.paragraph_format.space_after = Pt(4)
    go_cols()
    para([('REFERENCES', {})], size=10, align=WD_ALIGN_PARAGRAPH.CENTER, before=6, after=3, indent=False, keep=True)
    for i, k in enumerate(order, 1):
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Mm(6); p.paragraph_format.first_line_indent = Mm(-6); p.paragraph_format.space_after = Pt(1.5)
        r = p.add_run('[%d]  ' % i); r.font.size = Pt(7.5)
        for text, sty in runs_docx(refs[k]):
            r = p.add_run(text); r.font.size = Pt(7.5); r.italic = sty.get('i', False)
    doc.save(path)
