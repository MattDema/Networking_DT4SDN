// Load saved theme preference
const savedTheme = localStorage.getItem('theme') || 'light';
document.body.setAttribute('data-theme', savedTheme);
updateThemeButton(savedTheme);

function toggleTheme() {
    const body = document.body;
    const currentTheme = body.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    
    body.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeButton(newTheme);
}

function updateThemeButton(theme) {
    const icon = document.getElementById('theme-icon');
    const text = document.getElementById('theme-text');
    
    if (theme === 'dark') {
        icon.textContent = '☀️';
        text.textContent = 'Light';
    } else {
        icon.textContent = '🌙';
        text.textContent = 'Dark';
    }
}

// Auto-refresh every 3 seconds
setTimeout(function() {
    window.location.reload();
}, 3000);