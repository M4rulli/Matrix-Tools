
// Utility per rilevare dispositivo mobile
function isMobileDevice() {
    return window.innerWidth <= 768;
}

// Notebook System - Gestione completa del quaderno matematico
// File separato per modularità e manutenibilità

class NotebookManager {
    constructor() {
        this.storageKey = 'mathNotebook';
        this.currentExerciseIndex = -1;
        this.notebookData = [];
        this.isSaving = false;
        this.init();
    }

    init() {
        // 1. Carica il quaderno
        this.loadNotebook();
        // 2. Aggiorna subito il contatore
        this.updateNotebookCount();
        // 3. Setup event listeners
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Event listener per il bottone di apertura notebook nella sidebar
        const notebookButton = document.querySelector('.notebook-button');
        if (notebookButton) {
            notebookButton.addEventListener('click', () => this.openNotebook());
        }

        // Event listener per il bottone salva nella output zone
        document.addEventListener('click', (e) => {
            if (e.target.closest('.save-to-notebook-btn')) {
                this.saveToNotebook();
            }
        });
    }

    // Salva esercizio nel quaderno
    saveToNotebook() {
        // Previeni doppio salvataggio
        if (this.isSaving) {
            return;
        }
        this.isSaving = true;
        
        // Cerca il div di output corretto (sia 'output', 'outputModular', 'outputDNF', o fallback generico)
        let outputDiv =
            document.getElementById('output')
        // --- AGGIUNTA: Estrai il rawLatex ---
        const rawLatex =
            outputDiv && (
                outputDiv.dataset.latex
            );
        
        if (!outputDiv || outputDiv.innerHTML.trim() === '') {
            this.showNotification('Nessun risultato da salvare!', 'warning');
            this.isSaving = false;
            return;
        }
        
        // Rimuovi il bottone salva dal contenuto da salvare
        const outputContent = outputDiv.cloneNode(true);
        const saveButton = outputContent.querySelector('.save-to-notebook-btn');
        if (saveButton) {
            saveButton.remove();
        }

        // Rimuovi il diagramma SVG statico se presente
        const oldGraph = outputContent.querySelector('#hasse-graph');
        if (oldGraph) oldGraph.remove();

        // Inject Hasse graph placeholder into the cloned output HTML
        const placeholderId = `hasse-graph-notebook-${Date.now()}`;
        // Find the exact "1) Diagramma di Hasse:" element by text and insert wrapper after it
        const titleEl = [...outputContent.querySelectorAll('p, div')]
            .find(el => el.textContent.trim().startsWith('1) Diagramma di Hasse:'));
        if (titleEl && titleEl.parentNode) {
            const wrapper = document.createElement('div');
            wrapper.id = placeholderId;
            titleEl.parentNode.insertBefore(wrapper, titleEl.nextSibling);
        }

        // === Calcolo titolo incrementale per ogni tipo ===
        let exerciseTitle = this.generateExerciseTitle();
        const path = window.location.pathname;

        if (path.includes('potenze-modulari')) {
            const count = this.notebookData.filter(ex =>
                ex.algorithm && ex.algorithm.includes('Potenza Modulare')
            ).length + 1;
            exerciseTitle = `Potenza Modulare ${count}`;
        } else if (path.includes('diagramma-hasse')) {
            const count = this.notebookData.filter(ex =>
                ex.algorithm && ex.algorithm.includes('Diagramma di Hasse')
            ).length + 1;
            exerciseTitle = `Diagramma di Hasse ${count}`;
        } else if (path.includes('polinomi-booleani')) {
            const count = this.notebookData.filter(ex =>
                ex.algorithm && ex.algorithm.includes('Polinomi Booleani')
            ).length + 1;
            exerciseTitle = `Polinomio Booleano ${count}`;
        } else if (path.includes('linearizzazione')) {
            const count = this.notebookData.filter(ex =>
                ex.algorithm && ex.algorithm.includes('Linearizzazione Sistema')
            ).length + 1;
            exerciseTitle = `Linearizzazione Sistema ${count}`;
        } else if (path.includes('sistemi-lineari')) {
            const count = this.notebookData.filter(ex =>
                ex.title && ex.title.startsWith('Sistema Lineare')
            ).length + 1;
            exerciseTitle = `Sistema Lineare ${count}`;
        } else if (path.includes('equazioni-differenziali')) {
            const count = this.notebookData.filter(ex =>
                ex.algorithm && ex.algorithm.includes('Equazione Differenziale')
            ).length + 1;
            exerciseTitle = `Equazione Differenziale ${count}`;
        } else if (path.includes('equazioni-differenze')) {
            const count = this.notebookData.filter(ex =>
                ex.algorithm && ex.algorithm.includes('Equazione alle Differenze')
            ).length + 1;
            exerciseTitle = `Equazione alle Differenze ${count}`;
        }
        else if (path.includes('studio-funzione')) {
            const count = this.notebookData.filter(ex =>
                ex.algorithm && ex.algorithm.includes('Studio di Funzione')
            ).length + 1;
            exerciseTitle = `Studio di Funzione ${count}`;
        }
        else if (path.includes('autovalori-autovettori')) {
            const count = this.notebookData.filter(ex =>
                ex.title && ex.title.startsWith('Autovalori e Autovettori')
            ).length + 1;
            exerciseTitle = `Autovalori e Autovettori ${count}`;
        }
        else if (path.includes('determinante-laplace')) {
            const count = this.notebookData.filter(ex =>
                ex.title && ex.title.startsWith('Determinante con Laplace')
            ).length + 1;
            exerciseTitle = `Determinante con Laplace ${count}`;
        }
        else if (path.includes('decomposizione-spettrale')) {
            const count = this.notebookData.filter(ex =>
                ex.title && ex.title.startsWith('Decomposizione spettrale')
            ).length + 1;
            exerciseTitle = `Decomposizione Spettrale ${count}`;
        }
        else if (path.includes('teorema-cinese')) {
            const count = this.notebookData.filter(ex =>
                ex.algorithm && ex.algorithm.includes('Teorema Cinese del Resto')
            ).length + 1;
            exerciseTitle = `Teorema Cinese del Resto ${count}`;
        }
        else if (path.includes('identita-bezout')) {
            const countBezout = this.notebookData.filter(ex =>
                ex.algorithm && ex.algorithm.includes('Identità di Bézout')
            ).length + 1;
            exerciseTitle = `Identità di Bézout ${countBezout}`;
        }
        else if (path.includes('simplesso')) {
            const count = this.notebookData.filter(ex =>
                ex.title && ex.title.startsWith('Metodo del Simplesso')
            ).length + 1;
            exerciseTitle = `Metodo del Simplesso ${count}`;
        }
        else if (path.includes('condizioni-complementari')) {
            const count = this.notebookData.filter(ex =>
                ex.title && ex.title.startsWith('Condizioni di Complementarità')
            ).length + 1;
            exerciseTitle = `Condizioni di Complementarità ${count}`;
        }

        const exercise = {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            title: exerciseTitle,
            date: new Date().toLocaleDateString('it-IT', {
                year: 'numeric',
                month: 'long',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            }),
            algorithm: this.generateExerciseTitle(),
            input: this.getInputString ? this.getInputString() : undefined,
            htmlOutput: outputContent.innerHTML,
            outputLatex: rawLatex,
            rawData: window.currentExercise?.rawData || undefined,
            graphPayload: window.currentExercise?.graphPayload || undefined,
            placeholderId // add this line
        };
        console.log('Saved exercise object:', exercise);

        this.notebookData.push(exercise);
        this.saveNotebook();
        this.updateNotebookCount();
        
        this.showNotification('Esercizio aggiunto al quaderno!', 'success');
        
        // Reset flag dopo un breve delay
        setTimeout(() => {
            this.isSaving = false;
        }, 1000);
    }

    // Genera titolo automatico per l'esercizio
    generateExerciseTitle() {
        // Se è disponibile window.currentExercise, usa quel titolo
        if (window.currentExercise && window.currentExercise.title) {
            return window.currentExercise.title;
        }
        
        // Altrimenti determina dal URL/pathname
        const path = window.location.pathname;
        let exerciseType = "Esercizio Matematico";
        
        if (path.includes('potenze-modulari')) {
            exerciseType = "Calcolo Potenza Modulare";
        } else if (path.includes('diagramma-hasse')) {
            exerciseType = "Diagramma di Hasse";
        } else if (path.includes('polinomi-booleani')) {
            exerciseType = "Polinomi Booleani";
        } else if (path.includes('linearizzazione')) {
            exerciseType = "Linearizzazione Sistema";
        } else if (path.includes('sistemi-lineari')) {
            exerciseType = "Sistema Lineare";
        } else if (path.includes('equazioni-differenziali')) {
            exerciseType = "Equazione Differenziale";
        } else if (path.includes('equazioni-differenze')) {
            exerciseType = "Equazione alle Differenze";
        }
        else if (path.includes('studio-funzione')) {
            exerciseType = "Studio di Funzione";
        }
        else if (path.includes('autovalori-autovettori')) {
            exerciseType = "Autovalori e Autovettori";
        }
        else if (path.includes('determinante-laplace')) {
            exerciseType = "Determinante con Laplace";
        }
        else if (path.includes('teorema-cinese')) {
            exerciseType = "Teorema Cinese del Resto";
        }
        else if (path.includes('identita-bezout')) {
            exerciseType = "Identità di Bézout";
        }
        else if (path.includes('simplesso')) {
            exerciseType = "Metodo del Simplesso";
        }
        else if (path.includes('condizioni-complementari')) {
            exerciseType = "Condizioni di Complementarità";
        }
        
        return exerciseType;
    }

    // Apri modal del quaderno
    openNotebook() {
        this.loadNotebook();
        this.renderNotebook();
        const modalElement = document.getElementById('notebookModal');
        const modal = new bootstrap.Modal(modalElement);
        
        // Aggiungi event listener per pulire alla chiusura
        modalElement.addEventListener('hidden.bs.modal', () => {
            // Rimuovi classi modal Bootstrap
            document.body.classList.remove('modal-open');
            document.body.style.overflow = '';
            document.body.style.paddingRight = '';
            
            // Rimuovi backdrop
            const backdrop = document.querySelector('.modal-backdrop');
            if (backdrop) {
                backdrop.remove();
            }
            
            // Forza ripristino scroll
            setTimeout(() => {
                document.documentElement.style.overflow = '';
                document.body.style.overflow = '';
            }, 100);
        });
        
        modal.show();
    }

    // Renderizza il contenuto del quaderno
    renderNotebook() {
        const notebookList = document.getElementById('notebookList');
        const notebookViewer = document.getElementById('notebookViewer');
        const emptyNotebook = document.getElementById('emptyNotebook');

        if (this.notebookData.length === 0) {
            if (notebookList) notebookList.style.display = 'none';
            if (notebookViewer) notebookViewer.style.display = 'none';
            if (emptyNotebook) emptyNotebook.style.display = 'block';
            return;
        }

        if (emptyNotebook) emptyNotebook.style.display = 'none';
        if (notebookList) notebookList.style.display = 'block';
        if (notebookViewer) notebookViewer.style.display = 'block';

        // Renderizza indice
        this.renderIndex();
        
        // Aggiorna i badge nell'header
        this.updateHeaderBadges();
        
        // Mostra primo esercizio se nessuno è selezionato
        if (this.currentExerciseIndex === -1 && this.notebookData.length > 0) {
            this.currentExerciseIndex = 0;
        }
        this.renderCurrentExercise();
    }

    // Renderizza l'indice degli esercizi
    renderIndex() {
        const notebookIndexContent = document.getElementById('notebookIndexContent');
        if (!notebookIndexContent) return;
        
        notebookIndexContent.innerHTML = this.notebookData.map((exercise, index) => `
            <div class="notebook-index-item ${index === this.currentExerciseIndex ? 'active' : ''}" 
                 onclick="notebookManager.selectExercise(${index})">
                <div class="index-item-number">${index + 1}</div>
                <div class="index-item-content">
                    <div class="index-item-title">${exercise.title}</div>
                    <div class="index-item-date">${exercise.date}</div>
                </div>
                <button class="btn btn-sm btn-outline-danger index-item-delete" 
                        onclick="event.stopPropagation(); notebookManager.deleteExercise(${index})" 
                        title="Elimina esercizio">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `).join('');
        
        // Aggiorna i badge nell'header
        this.updateHeaderBadges();
    }

    // Aggiorna i badge informativi nell'header
    updateHeaderBadges() {
        const savedCountBadge = document.getElementById('savedCountBadge');
        const lastUpdateBadge = document.getElementById('lastUpdateBadge');
        
        if (savedCountBadge) {
            savedCountBadge.textContent = this.notebookData.length;
        }
        
        if (lastUpdateBadge) {
            if (this.notebookData.length > 0) {
                const lastExercise = this.notebookData[this.notebookData.length - 1];
                let date;
                if (lastExercise.timestamp) {
                    date = new Date(lastExercise.timestamp);
                } else if (lastExercise.date) {
                    // Fallback per vecchi dati
                    date = new Date();
                }
                if (date && !isNaN(date.getTime())) {
                    lastUpdateBadge.textContent = date.toLocaleDateString('it-IT', {
                        day: '2-digit',
                        month: '2-digit',
                        year: 'numeric'
                    });
                } else {
                    lastUpdateBadge.textContent = 'Oggi';
                }
            } else {
                lastUpdateBadge.textContent = '-';
            }
        }
    }

    // Renderizza l'esercizio corrente
    renderCurrentExercise() {
        const notebookExerciseContent = document.getElementById('notebookExerciseContent');
        const viewerCounter = document.getElementById('viewerCounter');
        
        if (!notebookExerciseContent) return;
        
        // Aggiorna i bottoni di navigazione
        this.updateNavigationButtons();
        
        if (this.currentExerciseIndex >= 0 && this.currentExerciseIndex < this.notebookData.length) {
            const exercise = this.notebookData[this.currentExerciseIndex];
            console.log('Rendering exercise in notebook:', exercise);
            
            if (viewerCounter) {
                viewerCounter.textContent = `Esercizio ${this.currentExerciseIndex + 1} di ${this.notebookData.length}`;
            }
            
            let mainContent = exercise.htmlOutput || exercise.outputLatex || '';

            notebookExerciseContent.innerHTML = `
                <div class="exercise-header">
                    <h6>${exercise.title}</h6>
                    <small class="text-muted">${exercise.date}</small>
                </div>
                <div class="exercise-content">
                    ${mainContent}
                </div>
            `;

            // Render Hasse diagram into pre-injected placeholder
            if (
                exercise.placeholderId &&
                exercise.graphPayload &&
                typeof renderHasseDiagramFromPayload === 'function'
            ) {
                renderHasseDiagramFromPayload(
                    exercise.placeholderId,
                    exercise.graphPayload,
                    document.documentElement.getAttribute('data-theme') || 'light'
                );
                // Nuova gestione ResizeObserver e fallback
                const wrapper = document.getElementById(exercise.placeholderId);
                if (wrapper) {
                    const observer = new ResizeObserver(entries => {
                        const svg = wrapper.querySelector('svg');
                        if (svg) {
                            const height = svg.getBoundingClientRect().height;
                            if (!isNaN(height) && height > 0) {
                                wrapper.style.height = `${height}px`;
                            }
                        }
                    });
                    observer.observe(wrapper);

                    // Fallback ritardato per sicurezza
                    setTimeout(() => {
                        const svg = wrapper.querySelector('svg');
                        if (svg) {
                            const height = svg.getBoundingClientRect().height;
                            if (!isNaN(height) && height > 0) {
                                wrapper.style.height = `${height}px`;
                            }
                        }
                    }, 500);
                }
            }
            // Re-renderizza MathJax se presente
            if (window.MathJax && window.MathJax.typesetPromise) {
                window.MathJax.typesetPromise([notebookExerciseContent]).catch((err) => {
                    console.warn('MathJax render error:', err);
                });
            }
        } else {
            if (viewerCounter) {
                viewerCounter.textContent = 'Seleziona un esercizio';
            }
            
            notebookExerciseContent.innerHTML = `
                <div class="exercise-placeholder">
                    <i class="fas fa-mouse-pointer" style="font-size: 3rem; opacity: 0.3; margin-bottom: 1rem;"></i>
                    <h5>Seleziona un esercizio</h5>
                    <p>Clicca su un esercizio nell'indice per visualizzarlo qui</p>
                </div>
            `;
        }
    }

    // Aggiorna lo stato dei bottoni di navigazione
    updateNavigationButtons() {
        const prevBtn = document.querySelector('[onclick="notebookManager.previousExercise()"]');
        const nextBtn = document.querySelector('[onclick="notebookManager.nextExercise()"]');
        
        if (prevBtn) {
            prevBtn.disabled = this.currentExerciseIndex <= 0;
        }
        
        if (nextBtn) {
            nextBtn.disabled = this.currentExerciseIndex >= this.notebookData.length - 1;
        }
    }

    // Navigazione esercizi
    selectExercise(index) {
        this.currentExerciseIndex = index;
        this.renderIndex();
        this.renderCurrentExercise();
    }

    previousExercise() {
        if (this.currentExerciseIndex > 0) {
            this.currentExerciseIndex--;
            this.renderIndex();
            this.renderCurrentExercise();
        }
    }

    nextExercise() {
        if (this.currentExerciseIndex < this.notebookData.length - 1) {
            this.currentExerciseIndex++;
            this.renderIndex();
            this.renderCurrentExercise();
        }
    }

    toggleIndex() {
        const indexContent = document.getElementById('notebookIndexContent');
        const toggleIcon = document.getElementById('indexToggleIcon');
        const toggleBtn = document.querySelector('.index-toggle-btn');
        
        if (indexContent && toggleIcon && toggleBtn) {
            const isCollapsed = indexContent.classList.contains('collapsed');
            
            if (isCollapsed) {
                // Expanding
                indexContent.classList.remove('collapsed');
                toggleBtn.classList.remove('collapsed');
                toggleIcon.className = 'fas fa-chevron-up';
            } else {
                // Collapsing
                indexContent.classList.add('collapsed');
                toggleBtn.classList.add('collapsed');
                toggleIcon.className = 'fas fa-chevron-down';
            }
        }
    }

    deleteExercise(index) {
        if (confirm('Sei sicuro di voler eliminare questo esercizio?')) {
            this.notebookData.splice(index, 1);
            
            // Aggiusta l'indice corrente se necessario
            if (this.currentExerciseIndex >= index) {
                this.currentExerciseIndex--;
            }
            
            if (this.currentExerciseIndex < 0 && this.notebookData.length > 0) {
                this.currentExerciseIndex = 0;
            }
            
            this.saveNotebook();
            this.updateNotebookCount();
            this.renderNotebook();
            
            this.showNotification('Esercizio eliminato', 'info');
        }
    }

    clearNotebook() {
        if (confirm('Sei sicuro di voler svuotare completamente il quaderno?')) {
            this.notebookData = [];
            this.currentExerciseIndex = -1;
            this.saveNotebook();
            this.updateNotebookCount();
            this.renderNotebook();
            
            this.showNotification('Quaderno svuotato', 'info');
        }
    }

    showExportMenu() {
        const exportSubmenu = document.getElementById('exportSubmenu');
        if (exportSubmenu) {
            const isVisible = exportSubmenu.style.display === 'block';
            exportSubmenu.style.display = isVisible ? 'none' : 'block';
        }
    }

    hideExportMenu() {
        const exportSubmenu = document.getElementById('exportSubmenu');
        if (exportSubmenu) {
            exportSubmenu.style.display = 'none';
        }
    }

    exportAsLatex() {
        console.log('exportAsLatex called');
        console.log('Current notebook data:', this.notebookData);
        console.log('Notebook length:', this.notebookData ? this.notebookData.length : 'undefined');
        if (typeof window.exportNotebookAsLatex === 'function') {
            console.log('Calling window.exportNotebookAsLatex');
            window.exportNotebookAsLatex();
        } else {
            console.error('window.exportNotebookAsLatex is not a function');
            this.showNotification('Funzione LaTeX non disponibile', 'error');
        }
        this.hideExportMenu();
    }

    exportAsPdf() {
        console.log('exportAsPdf called');
        if (typeof window.exportNotebookAsPdf === 'function') {
            window.exportNotebookAsPdf();
        } else {
            this.showNotification('Funzione PDF non disponibile', 'error');
        }
        this.hideExportMenu();
    }

    exportAsJson() {
        console.log('exportAsJson called');
        if (typeof window.exportNotebookAsJson === 'function') {
            window.exportNotebookAsJson();
        } else {
            this.showNotification('Funzione JSON non disponibile', 'error');
        }
        this.hideExportMenu();
    }

    saveNotebook() {
        localStorage.setItem(this.storageKey, JSON.stringify(this.notebookData));
        localStorage.setItem('mathNotebookCount', this.notebookData.length);
    }

    loadNotebook() {
        try {
            const saved = localStorage.getItem(this.storageKey);
            if (saved && saved !== 'undefined' && saved !== 'null') {
                const parsed = JSON.parse(saved);
                // Imposta direttamente notebookData se valido, altrimenti array vuoto
                this.notebookData = Array.isArray(parsed) ? parsed : [];
            } else {
                this.notebookData = [];
            }
        } catch (error) {
            console.warn('Errore caricamento quaderno:', error);
            this.notebookData = [];
            // Pulisci localStorage corrotto
            localStorage.removeItem(this.storageKey);
        }
    }

    getNotebook() {
        return this.notebookData;
    }

    updateNotebookCount() {
        // Aggiorna contatore nella sidebar
        const countElement = document.getElementById('notebookCount');
        if (countElement) {
            countElement.textContent = this.notebookData.length;
        }

        // Aggiorna badge nel modal del quaderno
        const savedCountBadge = document.getElementById('savedCountBadge');
        if (savedCountBadge) {
            savedCountBadge.textContent = this.notebookData.length;
        }

        // Notifica il contatore per entrambe le sidebar
        const count = this.notebookData.length;
        document.dispatchEvent(new CustomEvent('notebookUpdated', {
            detail: { suffix: 'desktop', count: count }
        }));
        document.dispatchEvent(new CustomEvent('notebookUpdated', {
            detail: { suffix: 'mobile', count: count }
        }));
    }

    showNotification(message, type = 'info') {
        // Rimuovi notifiche esistenti
        const existingToasts = document.querySelectorAll('.toast-notification');
        existingToasts.forEach(toast => toast.remove());

        const toast = document.createElement('div');
        toast.className = `toast-notification toast-${type}`;
        toast.innerHTML = `
            <div class="toast-content">
                <i class="${this.getToastIcon(type)} me-2"></i>
                ${message}
            </div>
        `;

        document.body.appendChild(toast);

        // Mostra il toast
        setTimeout(() => toast.classList.add('show'), 100);

        // Rimuovi il toast dopo 3 secondi
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    getToastIcon(type) {
        const icons = {
            'success': 'fas fa-check-circle',
            'warning': 'fas fa-exclamation-triangle',
            'info': 'fas fa-info-circle',
            'error': 'fas fa-times-circle'
        };
        return icons[type] || icons.info;
    }
}

/**
 * Inietta il bottone “Aggiungi al quaderno” nell’elemento di output
 * e registra i dati in window.currentExercise per NotebookManager
 *
 * @param {HTMLElement} outputDiv – il div che contiene l’HTML di risultato
 * @param {Object} payload – { title, algorithm, input, htmlOutput, outputLatex }
 */
window.injectSaveButton = function (outputDiv, payload) {
    if (!outputDiv) return;

    // 1) Salva i dati per il notebook
    window.currentExercise = payload;

    // 2) Evita duplicati
    if (outputDiv.querySelector('.save-to-notebook-btn')) return;

    // 3) Crea il bottone
    const btn = document.createElement('button');
    btn.className = 'save-to-notebook-btn';
    btn.title = 'Aggiungi al quaderno';
    btn.innerHTML = '<i class="fas fa-plus"></i> Aggiungi al quaderno';
    btn.onclick = () => {
        if (window.notebookManager) {
            window.notebookManager.saveToNotebook();
        } else if (typeof saveToNotebook === 'function') {
            // fallback legacy support
            saveToNotebook();
        }
    };

    // 4) Inserisci nel DOM e mostra
    outputDiv.appendChild(btn);
    btn.style.display = 'flex';
};

// Inizializza il manager del quaderno quando il DOM è pronto
let notebookManager;
document.addEventListener('DOMContentLoaded', () => {
    // Leggi valore salvato per evitare transizione 0→x
    const storedCount = parseInt(localStorage.getItem('mathNotebookCount') || '0');
    const desktopTarget = document.getElementById('exercise-count-desktop');
    const mobileTarget = document.getElementById('exercise-count-mobile');
    if (desktopTarget) desktopTarget.textContent = storedCount;
    if (mobileTarget) mobileTarget.textContent = storedCount;

    // Inizializza tutto normalmente
    notebookManager = new NotebookManager();
    window.notebookManager = notebookManager;
    console.log('NotebookManager initialized and exposed globally');
});

// Funzioni globali per compatibilità con il codice esistente
function saveToNotebook() {
    if (notebookManager) {
        notebookManager.saveToNotebook();
    }
}

function openNotebook() {
    if (notebookManager) {
        notebookManager.openNotebook();
    }
}

function clearNotebook() {
    if (notebookManager) {
        notebookManager.clearNotebook();
    }
}

function getNotebook() {
    return notebookManager ? notebookManager.getNotebook() : [];
}

function updateNotebookCount() {
    if (notebookManager) {
        notebookManager.updateNotebookCount();
    }
}


function showNotification(message, type) {
    if (notebookManager) {
        notebookManager.showNotification(message, type);
    }
}
