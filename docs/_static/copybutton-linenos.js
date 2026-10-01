// sphinx-copybutton drops the line-number spans (.linenos) from copied code, but
// Sphinx renders a separator space after each of them, outside the span. Copied
// Python would then start every line with one space and fail with an
// IndentationError. Moving that space into the span keeps the rendering identical
// and lets copybutton remove it together with the number.
document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("pre .linenos").forEach((node) => {
        const next = node.nextSibling;
        if (next && next.nodeType === Node.TEXT_NODE && next.textContent.startsWith(" ")) {
            next.textContent = next.textContent.slice(1);
            node.textContent += " ";
        }
    });
});
