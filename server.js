const express = require("express");
const bodyParser = require("body-parser");
const cors = require("cors");
const admin = require("firebase-admin");

// Inisialisasi Firebase Admin SDK
const serviceAccount = require("./bcknd/bcknd.json");
admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
});

const app = express();
app.use(cors());
app.use(bodyParser.json());

// Endpoint untuk menghapus pengguna
app.post("/deleteUser", async (req, res) => {
    const { userId } = req.body;

    if (!userId) {
        return res.status(400).send({ error: "User ID tidak diberikan" });
    }

    try {
        // Hapus pengguna dari Firebase Authentication
        await admin.auth().deleteUser(userId);
        console.log(`Akun dengan UID ${userId} berhasil dihapus`);
        res.status(200).send({ message: "Pengguna berhasil dihapus" });
    } catch (error) {
        console.error("Gagal menghapus akun pengguna:", error);
        res.status(500).send({ error: "Gagal menghapus akun pengguna" });
    }
});

// Tambahkan log ketika server mulai berjalan
const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server berjalan di http://localhost:${PORT}`);
});
