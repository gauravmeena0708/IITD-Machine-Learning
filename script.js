document.addEventListener('DOMContentLoaded', function() {
    const contentArea = document.querySelector('.content');
    const navLinks = document.querySelectorAll('.sidebar nav ul li a');

    // Function to fetch and display content
    const loadContent = (url) => {
        fetch(url)
            .then(response => response.text())
            .then(data => {
                contentArea.innerHTML = marked.parse(data);
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
