// admhome.js

import { initializeApp } from "https://www.gstatic.com/firebasejs/9.17.2/firebase-app.js";
import { getFirestore, collection, getDocs, deleteDoc, doc } from "https://www.gstatic.com/firebasejs/9.17.2/firebase-firestore.js";

const firebaseConfig = {
    apiKey: "AIzaSyCIg6W43aFMst9RuZhp_XzsJCAcV3bhwwc",
    authDomain: "kel777.firebaseapp.com",
    projectId: "kel777",
    storageBucket: "kel777.appspot.com",
    messagingSenderId: "54577150898",
    appId: "1:54577150898:web:a00d2714f67f0ddbf1c2ab",
    measurementId: "G-CZVEN08RZ3"
};

const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

// loaduser
async function loadUsers() {
    const userTable = document.getElementById("userTable");
    userTable.innerHTML = ""; 

    try {
        const querySnapshot = await getDocs(collection(db, "users"));
        querySnapshot.forEach((doc) => {
            const userData = doc.data();
            const userId = doc.id;

            const row = document.createElement("tr");
            row.innerHTML = `
                <td>${userData.firstName} ${userData.lastName}</td>
                <td>${userData.email}</td>
                <td>
                    <button class="delete-btn" data-user-id="${userId}">Delete</button>
                </td>
            `;
            userTable.appendChild(row);
        });
        document.querySelectorAll(".delete-btn").forEach(button => {
            button.addEventListener("click", function() {
                const userId = this.getAttribute("data-user-id");
                deleteUser(userId);
            });
        });
    } catch (error) {
        console.error("Gagal memuat data pengguna:", error);
    }
}

// hapus user
async function deleteUser(userId) {
    if (confirm("Apakah Anda yakin ingin menghapus pengguna ini?")) {
        try {
            
            await deleteDoc(doc(db, "users", userId));

            const response = await fetch("http://localhost:3000/deleteUser", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ userId }), 
            });

            if (response.ok) {
                alert("Pengguna berhasil dihapus!");
                loadUsers(); 
            } else {
                alert("Gagal menghapus pengguna di Firebase Authentication");
            }
        } catch (error) {
            console.error("Gagal menghapus pengguna:", error);
        }
    }
}


function logout() {
    alert("Anda telah logout!");
    window.location.href = "login.html"; 
}

window.onload = loadUsers;
