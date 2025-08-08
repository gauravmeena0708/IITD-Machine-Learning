document.addEventListener('DOMContentLoaded', function() {
    const navigation = document.getElementById('navigation');
    const contentArea = document.getElementById('content-area');
    let courseData = null;

    // Fetch course data
    fetch('courses.json')
        .then(response => response.json())
        .then(data => {
            courseData = data;
            generateNavigation(data);
            // Load default content (first semester)
            if (data.semesters.length > 0) {
                renderSemester(data.semesters[0]);
                setActiveLink(navigation.querySelector('a'));
            }
        })
        .catch(error => {
            console.error('Error loading course data:', error);
            contentArea.innerHTML = '<h2>Error loading course data</h2>';
        });

    // Generate navigation
    function generateNavigation(data) {
        const navList = document.createElement('ul');

        // Semesters
        data.semesters.forEach(semester => {
            const listItem = document.createElement('li');
            const link = document.createElement('a');
            link.href = '#';
            link.textContent = semester.name;
            link.addEventListener('click', (e) => {
                e.preventDefault();
                renderSemester(semester);
                setActiveLink(link);
            });
            listItem.appendChild(link);
            navList.appendChild(listItem);
        });

        // Other Resources
        data.other_resources.forEach(category => {
            const listItem = document.createElement('li');
            const link = document.createElement('a');
            link.href = '#';
            link.textContent = category.category;
            link.addEventListener('click', (e) => {
                e.preventDefault();
                renderOtherResource(category);
                setActiveLink(link);
            });
            listItem.appendChild(link);
            navList.appendChild(listItem);
        });

        // Projects
        if(data.projects) {
            const listItem = document.createElement('li');
            const link = document.createElement('a');
            link.href = '#';
            link.textContent = 'Projects';
            link.addEventListener('click', (e) => {
                e.preventDefault();
                renderProjects(data.projects);
                setActiveLink(link);
            });
            listItem.appendChild(link);
            navList.appendChild(listItem);
        }

        navigation.appendChild(navList);
    }

    // Set active link
    function setActiveLink(activeLink) {
        const links = navigation.querySelectorAll('a');
        links.forEach(link => link.classList.remove('active'));
        if (activeLink) {
            activeLink.classList.add('active');
        }
    }

    // Render semester content
    function renderSemester(semester) {
        contentArea.innerHTML = `<h2>${semester.name}</h2>`;
        semester.courses.forEach(course => {
            const courseDiv = document.createElement('div');
            courseDiv.className = 'course';

            let html = `<h3>${course.title}</h3>`;
            if (course.professor) {
                html += `<p><strong>Professor:</strong> ${course.professor}</p>`;
            }
            if (course.website) {
                html += `<p><a href="${course.website}" target="_blank">Course Website</a></p>`;
            }

            html += createResourceList('Related Courses', course.related_courses);
            html += createResourceList('Books', course.books);
            html += createResourceList('Resources', course.resources);

            courseDiv.innerHTML = html;
            contentArea.appendChild(courseDiv);
        });
    }

    // Render other resource
    function renderOtherResource(category) {
        contentArea.innerHTML = `<h2>${category.category}</h2>`;
        if (category.content) {
            // For now, just display the raw content.
            // This can be improved to render markdown or html.
            const pre = document.createElement('pre');
            const code = document.createElement('code');
            code.textContent = category.content;
            pre.appendChild(code);
            contentArea.appendChild(pre);
        } else if (category.topics) {
            category.topics.forEach(topic => {
                const topicDiv = document.createElement('div');
                topicDiv.className = 'course'; // reuse course styling
                let html = `<h3>${topic.name}</h3>`;
                html += createResourceList('', topic.links);
                topicDiv.innerHTML = html;
                contentArea.appendChild(topicDiv);
            });
        }
    }

    // Render projects
    function renderProjects(projects) {
        contentArea.innerHTML = '<h2>Projects</h2>';
        const projectDiv = document.createElement('div');
        projectDiv.className = 'course';
        projectDiv.innerHTML = `<p>${projects.description}</p>`;
        contentArea.appendChild(projectDiv);
    }

    // Helper to create a list of links
    function createResourceList(title, items) {
        if (!items || items.length === 0) {
            return '';
        }
        let listHtml = `<h4>${title}</h4><ul class="resource-list">`;
        items.forEach(item => {
            if (item.url) {
                listHtml += `<li><a href="${item.url}" target="_blank">${item.title}</a></li>`;
            } else {
                listHtml += `<li>${item.title}</li>`;
            }
        });
        listHtml += '</ul>';
        return listHtml;
    }
});
