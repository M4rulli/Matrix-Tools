/**
 * Notebook Export Utilities
 * Handles LaTeX (.tex), PDF, and JSON export formats for mathematical exercises
 */

window.addEventListener('load', function() {
});

// Export notebook as LaTeX (.tex) file
function exportNotebookAsLatex() {
    // Fallback se getNotebook non è disponibile
    let notebook;
    if (typeof getNotebook === 'function') {
        notebook = getNotebook();
    } else if (typeof notebookManager !== 'undefined' && notebookManager) {
        notebook = notebookManager.getNotebook();
    } else {
        showNotification('Sistema quaderno non inizializzato!', 'error');
        return;
    }
    
    if (!notebook || notebook.length === 0) {
        showNotification('Il quaderno è vuoto!', 'warning');
        return;
    }

    let latexContent = `\\documentclass[12pt,a4paper]{article}
\\usepackage[utf8]{inputenc}
\\usepackage[italian]{babel}
\\usepackage{amsmath}
\\usepackage{amsfonts}
\\usepackage{amssymb}
\\usepackage{graphicx}
\\usepackage{geometry}
\\usepackage{fancyhdr}
\\usepackage{xcolor}
\\usepackage{tikz}
\\usepackage{float}

\\geometry{margin=2.5cm}
\\pagestyle{fancy}
\\fancyhf{}
\\rhead{\\thepage}
\\lhead{Quaderno Matematico}

\\title{Quaderno degli Esercizi Matematici}
\\author{Matrix Tools}
\\date{\\today}

\\begin{document}

\\maketitle
\\tableofcontents
\\newpage

`;

    // Sort exercises by timestamp
    const sortedNotebook = notebook.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

    sortedNotebook.forEach((exercise, index) => {
        // Usa la stringa TeX salvata come outputLatex se disponibile
        const latexOut = exercise.outputLatex;

        // Clean and wrap TikZ in figure floats
        let cleaned = cleanLatexOutput(latexOut);
        cleaned = cleaned.replace(
            /\\begin\{tikzpicture\}([\s\S]*?)\\end\{tikzpicture\}/g,
            '\\begin{figure}[H]\n\\centering\n\\begin{tikzpicture}$1\\end{tikzpicture}\n\\end{figure}\\noindent\n'
        );
        latexContent += `\\section{${escapeLatex(exercise.title)}}

${cleaned}

\\newpage

`;
    });

    latexContent += `\\end{document}`;

    downloadFile(latexContent, 'quaderno-matematico.tex', 'text/plain');
    showNotification('File LaTeX scaricato!', 'success');
}

// Export notebook as PDF (using browser print)
function exportNotebookAsPdf() {
    // Fallback se getNotebook non è disponibile
    let notebook;
    if (typeof getNotebook === 'function') {
        notebook = getNotebook();
    } else if (typeof notebookManager !== 'undefined' && notebookManager) {
        notebook = notebookManager.getNotebook();
    } else {
        showNotification('Sistema quaderno non inizializzato!', 'error');
        return;
    }
    
    if (!notebook || notebook.length === 0) {
        showNotification('Il quaderno è vuoto!', 'warning');
        return;
    }

    // Open a dedicated preview/pop‑up window for PDF inspection
    const printWindow = window.open('', '_blank');
    // If the popup is blocked, fallback to current window
    if (!printWindow) {
        console.warn('Popup blocked; falling back to current window');
    }
    const sortedNotebook = notebook.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

    let htmlContent = `
<!DOCTYPE html>
<html>
<head>
    <title>Quaderno Matematico</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script>
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']],
                processEscapes: true
            }
        };
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        /* ------------------------------------------------ */
        body { 
            font-family: 'Times New Roman', serif; 
            margin: 0.5cm 1cm; /* top/bottom:0.5cm, left/right:1cm */
            line-height: 1.4; 
        }
        h1 { margin: 0.2em 0; }
        h2 { margin: 0.1em 0 0.5em; }
        hr { margin: 0.5em 0; }
        /* Center any SVG diagrams inside exercises */
        .exercise-content svg {
            display: block;
            margin: 0.5em auto; 
        }
        /* Ensure SVG text uses a supported font */
        .exercise-content svg text,
        .exercise-content svg * {
            font-family: 'DejaVu Sans', sans-serif !important;
        }
        .exercise {
          margin: 0.5cm 0;
          page-break-inside: avoid;
        }
        .exercise-header { 
            border-bottom: 2px solid #333; 
            padding-bottom: 0.3cm; 
            margin-bottom: 0.3cm; 
        }
        .exercise-title { 
            font-size: 1.5em; 
            font-weight: bold; 
            margin-bottom: 0.5cm; 
        }
        .exercise-meta { 
            color: #666; 
            font-size: 0.9em; 
        }
        .exercise-content { 
            font-family: 'Georgia', serif; 
            background: #f5f5f5; 
            padding: 0.3cm;      /* ridotto per eliminare spazio bianco */
            border-radius: 0.5cm; 
        }
        /* Rimuove spazio bianco all'inizio del contenuto esercizio */
        .exercise-content > *:first-child {
            margin-top: 0 !important;
        }
        @media print {
            .exercise { page-break-after: always; }
            .exercise:last-child { page-break-after: auto; }
        }
    </style>
</head>
<body>
    <h1>Quaderno degli Esercizi Matematici</h1>
    <h2>Matrix Tools</h2>
    <hr>
    <h2>Indice degli esercizi</h2>
    <ol>
      ${sortedNotebook.map((ex, i) => `<li><a href="#exercise-${i+1}">${escapeHtml(ex.title)}</a></li>`).join('')}
    </ol>
    <hr>
`;

    sortedNotebook.forEach((exercise, index) => {
        htmlContent += `
    <div id="exercise-${index+1}" class="exercise">
        <div class="exercise-header">
            <div class="exercise-title">${escapeHtml(exercise.title)}</div>
            <div class="exercise-meta">
                <strong>Algoritmo:</strong> ${escapeHtml(exercise.algorithm)} | 
                <strong>Data:</strong> ${escapeHtml(exercise.date)}
            </div>
        </div>
        <div class="exercise-content">
            ${exercise.htmlOutput || exercise.outputLatex || exercise.output || ''}
        </div>
    </div>
`;
    });

    htmlContent += `
</body>
</html>`;

    // printWindow.document.write(htmlContent);
    printWindow.document.write('<!DOCTYPE html><html><head><title>Caricamento...</title></head><body>Preparazione PDF...</body></html>');
    printWindow.document.close();

    const poll = setInterval(() => {
        try {
            if (printWindow.document && printWindow.document.readyState === 'complete') {
                clearInterval(poll);
                printWindow.document.open();
                printWindow.document.write(htmlContent);
                printWindow.document.close();
            }
        } catch (e) {
            console.warn('[PDF] Polling failed:', e);
        }
    }, 100);
    
    showNotification('Preparazione PDF in corso...', 'info');
}

// Export notebook as JSON file
function exportNotebookAsJson() {
    // Fallback se getNotebook non è disponibile
    let notebook;
    if (typeof getNotebook === 'function') {
        notebook = getNotebook();
    } else if (typeof notebookManager !== 'undefined' && notebookManager) {
        notebook = notebookManager.getNotebook();
    } else {
        showNotification('Sistema quaderno non inizializzato!', 'error');
        return;
    }
    
    if (!notebook || notebook.length === 0) {
        showNotification('Il quaderno è vuoto!', 'warning');
        return;
    }

    const exportData = {
        metadata: {
            exportDate: new Date().toISOString(),
            totalExercises: notebook.length,
            version: "1.0",
            source: "Matrix Tools - Mathematical Calculator"
        },
        exercises: notebook.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))
    };

    const jsonContent = JSON.stringify(exportData, null, 2);
    downloadFile(jsonContent, 'quaderno-matematico.json', 'application/json');
    showNotification('File JSON scaricato!', 'success');
}

// Expose functions to global scope
window.exportNotebookAsLatex = exportNotebookAsLatex;
window.exportNotebookAsPdf = exportNotebookAsPdf;
window.exportNotebookAsJson = exportNotebookAsJson;


// Helper functions

function escapeLatex(text) {
    if (!text || typeof text !== 'string') {
        return String(text || '');
    }
    return text
        .replace(/\\/g, '\\textbackslash{}')
        .replace(/\{/g, '\\{')
        .replace(/\}/g, '\\}')
        .replace(/\$/g, '\\$')
        .replace(/&/g, '\\&')
        .replace(/%/g, '\\%')
        .replace(/#/g, '\\#')
        .replace(/\^/g, '\\textasciicircum{}')
        .replace(/_/g, '\\_')
        .replace(/~/g, '\\textasciitilde{}');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function cleanLatexOutput(output) {
    // Sanitize HTML-rich output and keep only pure LaTeX (and plain text)
    if (!output || typeof output !== 'string') {
        return String(output || '');
    }

    let tex = output;

    // 1. Convert <p class="centered-math"> ... </p> to just the inner TeX block with line breaks
    tex = tex.replace(/<p[^>]*?>\s*/gi, '\n');
    tex = tex.replace(/<\/p>/gi, '\n');

    // 2. Remove all remaining HTML tags (e.g., <b>, <div>, etc.)
    tex = tex.replace(/<[^>]+>/g, '');

    // 3. Decode basic HTML entities
    tex = tex
        .replace(/&amp;/g, '&')
        .replace(/&lt;/g, '<')
        .replace(/&gt;/g, '>')
        .replace(/&nbsp;/g, ' ');




    // 4. Remove duplicate blank lines
    tex = tex.replace(/\s*\n\s*/g, '');

    // 5. Trim leading/trailing whitespace
    tex = tex.trim();

    // 6. Remove only nested align environments, but preserve top-level align
    const alignRegex = /\\begin\{align\}([\s\S]*?)\\end\{align\}/g;
    tex = tex.replace(alignRegex, (match, inner) => {
        // Remove internal begin/end align to flatten nested ones
        const cleanedInner = inner.replace(/\\begin\{align\}/g, '').replace(/\\end\{align\}/g, '');
        return '\\begin{align}' + cleanedInner + '\\end{align}';
    });

    // 7. Ensure every \hline is followed by a trailing space for safety
    tex = tex.replace(/\\hline(?!\s)/g, '\\hline ');

    return tex;
}

function downloadFile(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

// Export functionality moved to header dropdown
document.addEventListener('DOMContentLoaded', function() {
});
