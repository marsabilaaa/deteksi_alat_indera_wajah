Panduan singkat: deploy static site (OpenCV.js) ke Vercel

1. Pastikan file cascade XML ada di root repository (sudah ada):

- haarcascade_frontalface_default.xml
- haarcascade_eye.xml
- haarcascade_mcs_nose.xml
- haarcascade_mcs_mouth.xml

2. Struktur file yang digunakan oleh halaman:

- index.html
- static/detect.js
- (cascade XML berada di root dan diakses sebagai `/haarcascade_...xml`)

3. Deploy ke Vercel:

- Jika belum: `npm i -g vercel` atau gunakan `npx vercel`
- Jalankan di root repo:

```
vercel --prod
```

Catatan:

- Aplikasi ini berjalan sepenuhnya di browser menggunakan OpenCV.js; tidak ada server-side processing. Pastikan browser mengizinkan kamera.
- Jika OpenCV.js CDN gagal, pertimbangkan untuk menaruh `opencv.js` lokal di folder `static/` dan update `index.html`.
