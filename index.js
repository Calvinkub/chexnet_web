const express = require('express');
const path = require('path');
const multer = require('multer');
const sqlite3 = require('sqlite3').verbose();
const app = express();



app.set('view engine', 'ejs'); // กำหนดให้ใช้ EJS เป็น view engine
app.set('views', path.join(__dirname, 'views')); // ตั้งค่า path สำหรับเก็บไฟล์ EJS

const port = 3000;

// เปิดการเชื่อมต่อฐานข้อมูล
const db = new sqlite3.Database('./Image_data.db', sqlite3.OPEN_READWRITE, (err) => {
    if (err) {
        return console.error(err.message);
    }
    console.log('Connected to the Image_data.db database.');
});




// ตั้งค่า multer
const storage = multer.memoryStorage(); // ใช้ memoryStorage เพื่อเก็บไฟล์ใน memory ก่อน
const upload = multer({ storage: storage });

// ใช้งานไฟล์ static
app.use(express.static('public'));

app.get('/', (req, res) => {
    res.render('scan_chest.ejs'); // Render ไฟล์ index.ejs ที่อยู่ในโฟลเดอร์ views

});app.get('/test', (req, res) => {
    res.render('test.ejs');
});


// สร้าง route สำหรับอัปโหลด
app.post('/upload', upload.single('image'), function (req, res) {
    const imageBuffer = req.file.buffer; // ใช้ buffer ที่ได้จาก multer
    const sql = `INSERT INTO images (image) VALUES (?)`;
    db.run(sql, [imageBuffer], function(err) {
        if (err) {
            console.error('Error inserting image into SQLite:', err);
            res.status(500).send('Error inserting image into database');
            return;
        }
        console.log(`A new image has been inserted with rowid ${this.lastID}`);
        res.json({ message: 'Image uploaded successfully', id: this.lastID });
    });
});

// เริ่มเซิร์ฟเวอร์
app.listen(port, () => {
    console.log(`Listening on port ${port}`);
});

// จัดการปิดฐานข้อมูลเมื่อปิด Node.js server
process.on('SIGINT', () => {
    db.close(() => {
        console.log('Database connection closed.');
        process.exit(0);
    });
});


