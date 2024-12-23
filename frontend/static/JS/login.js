document.addEventListener('DOMContentLoaded', function() {
    const loginBtn = document.getElementById('login');
    const signupBtn = document.getElementById('signup');
    const loginForm = document.querySelector('.login');
    const signupForm = document.querySelector('.signup');

    loginBtn.addEventListener('click', function() {
        loginForm.classList.remove('slide-up');
        signupForm.classList.add('slide-up');
    });

    signupBtn.addEventListener('click', function() {
        signupForm.classList.remove('slide-up');
        loginForm.classList.add('slide-up');
    });
});
