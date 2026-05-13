# Laser Range RTSP Test

## Pokretanje
1. Aktiviraj virtual environment
2. Instaliraj dependencies:
   pip install -r requirements.txt
3. Pokreni:
   python app.py

## Komande
- q = izlaz
- a = toggle auto-align
- n = freeze/unfreeze trenutnog poravnanja (potvrdi kalibraciju)
- r = reset rezultata
- + / -      = scale up/down (ringovi veći/manji za 0.5%)
- h / l      = pomeri centar levo / desno (0.5 mm)
- k / j      = pomeri centar gore / dole (0.5 mm)
- , / .      = rotacija ±0.5°
- [ / ]      = aspect ratio Y/X ±0.5%
- 0          = reset svih ručnih tweaks-a (scale=1.0, offset=0, rot=0, aspect=1.0)

Tweaks (scale + offset) se automatski snimaju u `calibration_tweaks.json`
i učitavaju pri svakom startu — kada jednom pogodiš veličinu i centar,
ne moraš više nikad to da radiš.

## Mobile control server
Python app pri startu pokreće HTTP server (default `0.0.0.0:8080`) koji
React Native mobile aplikacija koristi za pairing, live preview i kalibraciju.

Endpoints:
- `GET  /api/health`              — health check (pairing probe)
- `POST /api/pair`                — vraća dev token (no real auth)
- `GET  /api/stream/preview.mjpeg` — MJPEG stream anotirane slike (isto kao cv2 prozor)
- `POST /api/calibration/freeze`   — programatska zamena za pritisak `n` (zaledi)
- `POST /api/calibration/unfreeze` — odledi

Konfigurabilno u `config.py` (`CONTROL_SERVER_*`, `MJPEG_QUALITY`, `AUTH_TOKEN`).

### Raspberry Pi 5 + Camera Module 3 + Wi-Fi AP
- Postavi Pi kao AP (npr. NetworkManager hotspot ili `hostapd`).
- Tipične AP IP adrese koje mobile aplikacija probova: `192.168.4.1`,
  `192.168.42.1`, `10.42.0.1`, `shooterrange.local` — sve probava i na
  portu `8080` i na `80`.
- Ako želiš port 80 (bez `:8080` u URL), pokreni Python sa pravima ili
  proksiraj kroz `nginx`/`socat`. Najjednostavnije: ostavi 8080.
- U mobilnoj aplikaciji koristi auto-discovery ili "Manual IP" → `192.168.4.1`.

## Ako stream ne radi
- Testiraj URL u VLC
- Probaj RTSP_TRANSPORT = "tcp"
- Probaj USE_FFMPEG = False