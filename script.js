document.addEventListener('DOMContentLoaded', function() {
    const contentArea = document.querySelector('.content');
    const navLinks = document.querySelectorAll('.sidebar nav ul li a');

    // Function to fetch and display content
    const loadContent = (url) => {
        fetch(url)
            .then(response => response.text())
            .then(data => {
                // Simple markdown to HTML conversion for tables
                let html = data.replace(/<\/td>\s*<\/tr>\s*<\/tbody>\s*<\/table>/g, '</td></tr></tbody></table>')
                               .replace(/<\/table>\s*<h2>/g, '</table><h2>')
                               .replace(/\|/g, '</td><td>')
                               .replace(/---\|/g, '---|') // to avoid replacing the separator line
                               .replace(/\|\n/g, '</td></tr><tr>')
                               .replace(/<tr>\s*<td>/g, '<tr><th>')
                               .replace(/<\/td>\s*<\/tr>/g, '</th></tr>');

                // This is a very basic parser, it might need more work.
                // For now, I will manually fix the table html structure
                let lines = data.split('\\n');
                let inTable = false;
                let htmlOutput = '';
                lines.forEach(line => {
                    if (line.startsWith('<table>')) {
                        inTable = true;
                        htmlOutput += '<table>';
                    } else if (line.startsWith('</table>')) {
                        inTable = false;
                        htmlOutput += '</table>';
                    } else if (inTable) {
                        htmlOutput += line;
                    } else {
                        htmlOutput += `<p>${line}</p>`;
                    }
                });

                // A better approach is to use a proper markdown parser library,
                // but for now I will try to make it work with string replacement
                // and DOM manipulation.

                // Let's try a simpler approach to parse the table.
                // This is still not robust.
                const tableRegex = /<table>([\s\S]*?)<\/table>/g;
                let tableMatch;
                let processedHtml = data;

                // This is complex, I will use a library for markdown parsing
                // I will add the marked.js library
                // For now, I will just display the raw text

                // Let's try to parse the content manually for now
                // It seems the content is HTML inside markdown.
                // So I can just treat it as HTML.
                contentArea.innerHTML = data;

            })
            .catch(error => {
                console.error('Error loading content:', error);
                contentArea.innerHTML = '<h1>Error loading content</h1>';
            });
    };

    // Event listeners for navigation links
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const source = this.getAttribute('data-src');

            // Remove active class from all links
            navLinks.forEach(navLink => navLink.classList.remove('active'));
            // Add active class to the clicked link
            this.classList.add('active');

            loadContent(source);
        });
    });

    // Load default content (Semester 1)
    const defaultContentUrl = 'sem1.md';
    loadContent(defaultContentUrl);
    // Set the first link as active by default
    document.querySelector('.sidebar nav ul li a[data-src="sem1.md"]').classList.add('active');
});
