function renderHasseDiagramFromPayload(containerId, graphPayload) {
  const maxRetries = 10;
  let attempts = 0;

  function tryRender() {
    const container = document.getElementById(containerId);
    if (!container) {
      if (attempts++ < maxRetries) {
        setTimeout(tryRender, 100);
      } else {
        console.warn('Contenitore non trovato:', containerId);
      }
      return;
    }

    const tempDiv = document.createElement('div');
    tempDiv.style.position = 'absolute';
    tempDiv.style.width = '0';
    tempDiv.style.height = '0';
    tempDiv.style.overflow = 'hidden';
    document.body.appendChild(tempDiv);

    const currentColor = getComputedStyle(document.body).getPropertyValue('--bs-body-color').trim();

    const styleArr = [
      {
        selector: 'node',
        style: {
          'label': 'data(label)',
          'background-opacity': 0,
          'background-color': 'transparent',
          'text-opacity': 1,
          'text-valign': 'center',
          'text-halign': 'center',
          'font-size': '13px',
          'color': currentColor,
          'text-outline-width': 0,
          'border-width': 0
        }
      },
      {
        selector: 'edge',
        style: {
          'width': 1.5,
          'line-color': currentColor,
          'curve-style': 'bezier',
          'target-arrow-shape': 'none',
          'line-cap': 'round'
        }
      }
    ];

    const cy = cytoscape({
      container: tempDiv,
      elements: graphPayload.elements,
      style: styleArr,
      layout: graphPayload.layout || {
        name: 'dagre',
        rankDir: 'BT',
        padding: 20
      },
      userZoomingEnabled: false,
      userPanningEnabled: false,
      boxSelectionEnabled: false
    });

    cy.ready(() => {
      if (typeof cy.svg === 'function') {
        const svg = cy.svg({ full: true, scale: 1.2 });

        // Crea wrapper SVG centrato e scalato
        const wrapper = document.createElement('div');
        wrapper.style.cssText = `
          display: flex;
          justify-content: center;
          --bs-body-color: ${document.documentElement.getAttribute('data-theme') === 'dark' ? 'white' : 'black'};
        `;
        const inner = document.createElement('div');
        // Rileva se siamo su mobile
        const isMobile = window.matchMedia('(max-width: 768px)').matches;
        const scale = isMobile ? 0.3 : 0.5;

        inner.style.cssText = `
          transform: scale(${scale});
          transform-origin: top center;
        `;
        inner.innerHTML = svg;

        const applySvgTheme = () => {
          const theme = document.documentElement.getAttribute('data-theme');
          const svgEl = inner.querySelector('svg');
          if (!svgEl) return;
          const color = theme === 'dark' ? 'white' : 'black';

          // 1) Rimuove i path dei nodi (fill diverso da 'none')
          svgEl.querySelectorAll('path').forEach(pathEl => {
            if (pathEl.getAttribute('fill') !== 'none') {
              pathEl.remove();
            }
          });

          // 2) Colora tutti i path rimanenti (gli archi) e i testi
          svgEl.querySelectorAll('path').forEach(pathEl => {
            pathEl.setAttribute('stroke', color);
          });
          svgEl.querySelectorAll('text').forEach(textEl => {
            textEl.setAttribute('fill', color);
            textEl.setAttribute('font-family', `'Computer Modern', 'CMU Serif', 'Latin Modern Roman', serif`);
          });
        };

        applySvgTheme();

        const observer = new MutationObserver(() => applySvgTheme());
        observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

        wrapper.appendChild(inner);

        // Inserisci nel container
        container.innerHTML = '';
        container.appendChild(wrapper);

        // Adatta dinamicamente l'altezza del container all'altezza effettiva dello SVG
        const svgEl = inner.querySelector('svg');
        if (svgEl) {
          requestAnimationFrame(() => {
            const svgHeight = svgEl.getBoundingClientRect().height; 
            container.style.height = `${svgHeight}px`;
          });
        }
      } else {
        container.innerHTML = `
          &lt;div class="alert alert-warning"&gt;
            <i class="fas fa-exclamation-triangle me-2"></i>
            Diagramma non disponibile (plugin mancante)
          &lt;/div&gt;
        `;
        console.error('cytoscape-svg plugin non disponibile');
      }
      cy.destroy();
      document.body.removeChild(tempDiv);
    });
  }

  tryRender();
}