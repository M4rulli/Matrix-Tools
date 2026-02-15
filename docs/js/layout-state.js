if (document.readyState === "loading") {
    document.documentElement.classList.add("disable-transitions");
} else {
    // In rari casi il DOM è già pronto, forza immediatamente
    requestAnimationFrame(() => {
        document.documentElement.classList.add("disable-transitions");
    });
}

document.addEventListener("DOMContentLoaded", () => {
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            document.documentElement.classList.remove("disable-transitions");
            const html = document.documentElement;
            html.style.removeProperty("background-color");
            html.style.removeProperty("color");
            html.style.removeProperty("transition");
        });
    });
});

// Theme: initial handling

(function immediateThemeApplication() {
    const savedTheme = localStorage.getItem("theme");
    const prefersDark =
        window.matchMedia &&
        window.matchMedia("(prefers-color-scheme: dark)").matches;
    const theme = savedTheme || (prefersDark ? "dark" : "light");

    const html = document.documentElement;

    html.setAttribute("data-theme", theme);
    html.className = theme === "dark" ? "dark" : "";
    html.style.setProperty(
        "background-color",
        theme === "dark" ? "#0f172a" : "#ffffff",
        "important",
    );
    html.style.setProperty(
        "color",
        theme === "dark" ? "#f1f5f9" : "#1e293b",
        "important",
    );
    html.style.setProperty("transition", "none", "important");

    window.__initialTheme = theme;

    if (theme === "dark") {
        // Aspetta che il toggle sia nel DOM e imposta `checked = true`
        const observer = new MutationObserver(() => {
            const toggle = document.getElementById("themeToggle");
            if (toggle && !toggle.checked) {
                toggle.checked = true;
                observer.disconnect();
            }
        });
        observer.observe(document.documentElement, {
            childList: true,
            subtree: true,
        });
    }
})();

document.addEventListener("DOMContentLoaded", () => {
    const themeToggle = document.getElementById("themeToggle");
    if (themeToggle) {
        const theme =
            document.documentElement.getAttribute("data-theme") || "light";
        themeToggle.checked = theme === "dark";
    }
});

// Toggle function globale (può essere richiamata inline)
window.toggleTheme = function () {
    const toggle = document.getElementById("themeToggle");
    const html = document.documentElement;
    const isDark = toggle.checked;

    html.setAttribute("data-theme", isDark ? "dark" : "light");
    html.className = isDark ? "dark" : "";
    localStorage.setItem("theme", isDark ? "dark" : "light");
};

// Sidebar: layout iniziale

(function () {
    // Static frontend pages use a different sidebar system (component-loader).
    // Skip legacy sidebar bootstrapping to prevent open->close flicker on refresh.
    const hasLegacyWrapper = !!document.querySelector(".wrapper");
    if (!hasLegacyWrapper) {
        document.documentElement.classList.remove("sidebar-initial-expanded");
        return;
    }

    const sidebarExpanded = localStorage.getItem("sidebarExpanded") === "true";

    if (sidebarExpanded) {
        // Evita il flicker animato
        document.documentElement.classList.add("sidebar-initial-expanded");

        const applySidebarState = () => {
            const wrapper = document.querySelector(".wrapper");
            const sidebar = document.getElementById("sidebar");
            if (wrapper) {
                wrapper.classList.add("sidebar-expanded");
            }
            if (sidebar) {
                sidebar.classList.add("expanded");
            }
        };

        if (
            document.readyState === "complete" ||
            document.readyState === "interactive"
        ) {
            applySidebarState();
        } else {
            document.addEventListener("DOMContentLoaded", applySidebarState);
        }
    }

    // Remove only the temporary class to disable animation lock
    document.addEventListener("DOMContentLoaded", () => {
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                document.documentElement.classList.remove(
                    "sidebar-initial-expanded",
                );
            });
        });
    });
})();

// Notebook functions

function getNotebookExercises() {
    const exercises = localStorage.getItem("notebookExercises");
    return exercises ? JSON.parse(exercises) : [];
}

function saveNotebookExercises(exercises) {
    localStorage.setItem("notebookExercises", JSON.stringify(exercises));
}

function addExerciseToNotebook(exercise) {
    const exercises = getNotebookExercises();
    exercises.push(exercise);
    saveNotebookExercises(exercises);
    alert("Esercizio aggiunto al Quaderno!");
}

document.addEventListener("DOMContentLoaded", () => {
    const addToQuadernoButton = document.getElementById(
        "add-to-quaderno-button",
    );

    if (addToQuadernoButton) {
        addToQuadernoButton.addEventListener("click", () => {
            const inputCard = document.getElementById("input-card"); // Assumi che l'input sia in un card con id 'input-card'
            const outputCard = document.getElementById("output-card"); // Assumi che l'output sia in un card con id 'output-card'

            if (inputCard && outputCard) {
                const inputContent = inputCard.innerHTML; // O un altro modo per catturare l'input
                const outputContent = outputCard.innerHTML; // O un altro modo per catturare l'output

                const exercise = { input: inputContent, output: outputContent };
                addExerciseToNotebook(exercise);
            }
        });
    }
});
