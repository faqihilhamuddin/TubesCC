const signInButton = document.getElementById('signInButton');  
const signUpButton = document.getElementById('signUpButton');  
const submitSignIn = document.getElementById('submitSignIn'); 
const submitSignUp = document.getElementById('submitSignUp');  
const signInForm = document.getElementById('signInForm');  
const signUpForm = document.getElementById('signUpForm');  
const signInMessage = document.getElementById('signInMessage'); 
const signUpMessage = document.getElementById('signUpMessage'); 

// Username dan password admin
const adminUsername = 'admin';
const adminPassword = 'admin';

// Tampilan awal: Tampilkan form masuk dan sembunyikan form daftar
signInForm.style.display = "block";
signUpForm.style.display = "none";

// utk ke form masuk
signInButton.addEventListener('click', function() {
    signInForm.style.display = "block";
    signUpForm.style.display = "none";
    signInButton.classList.add('active');
    signUpButton.classList.remove('active');
});

// utk ke form daftar
signUpButton.addEventListener('click', function() {
    signUpForm.style.display = "block";
    signInForm.style.display = "none";
    signUpButton.classList.add('active');
    signInButton.classList.remove('active');
});

//  submit utk masuk
submitSignIn.addEventListener('click', function() {
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const role = document.getElementById('roleSelect').value;

    // Validasi input
    if (!email || !password || !role) {
        signInMessage.style.display = 'block';
        signInMessage.innerHTML = 'Semua kolom harus diisi!';
        return;
    }

    // utk adm lgn
    if (role === 'admin' && email === adminUsername && password === adminPassword) {
        window.location.href = 'admhome.html'; 
    } else if (role === 'user') {
        signInMessage.style.display = 'block';
        signInMessage.innerHTML = 'Login berhasil untuk user!';
    } else {
        signInMessage.style.display = 'block';
        signInMessage.innerHTML = 'Email, password, atau role tidak valid!';
    }
});

//  submit daftar
submitSignUp.addEventListener('click', function() {
    const fName = document.getElementById('fName').value;
    const lName = document.getElementById('lName').value;
    const rEmail = document.getElementById('rEmail').value;
    const rPassword = document.getElementById('rPassword').value;

    // Validasi input
    if (!fName || !lName || !rEmail || !rPassword) {
        signUpMessage.style.display = 'block';
        signUpMessage.innerHTML = 'Semua kolom harus diisi!';
        return;
    }

    // pesan
    signUpMessage.style.display = 'block';
    signUpMessage.innerHTML = 'Pendaftaran berhasil!';
});
