"""Arma el video final (≤ 5:00): intro, láminas con la voz de P1 y tomas de la demo con la voz de P2.

Uso: python armar_video.py <ruta a ffmpeg>
Salidas: video_final_con_musica.mp4 y video_final_sin_musica.mp4 en esta carpeta.
Los cortes y los offsets de audio salen de los tiempos de cada frase (transcripción con marcas de tiempo).
"""
import subprocess
import sys
from pathlib import Path

FF = sys.argv[1]
D = Path(__file__).parent
TMP = D / "_segmentos"
TMP.mkdir(exist_ok=True)
W, H, FPS = 1920, 1080, 30
FONDO = "0xF2F4F7"           # gris claro de las láminas, para los bordes de las tomas 16:10
VOZ = "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000"
FADE = 0.25

T1, T2, T3 = "2026-10-22 18-47-50.mp4", "2026-10-22 18-48-55.mp4", "2026-10-22 19-15-40.mp4"

# Lámina: (png, audio, inicio, fin) — inicio/fin recortan silencios de los audios de WhatsApp
LAMINAS = {
    "01": ("01_gancho.png", "P1_01_gancho.mp4", 0.90, 37.35),
    "02": ("02_fisica.png", "P1_02_fisica_y_modelos.mp4", 0.68, 15.00),
    "03": ("03_modelos.png", "P1_02_fisica_y_modelos.mp4", 15.00, 37.15),
    "04": ("04_evidencia.png", "P1_03_evidencia.mp4", 0.81, 41.35),
    "05": ("05_velovision.png", "P1_04_puente_velovision.mp4", 1.00, 12.72),
    "06": ("06_cierre.png", "P1_05_cierre.mp4", 1.11, 26.11),
}

# Toma: video = tramos (ini, fin, congelar_al_final_s);
# audio = (ini, fin, colocar_en_s[, opciones]) sobre la toma ya editada; opciones: "src" (otro archivo), "tempo"
TOMAS = {
    "t1": (T1, "p2_01.wav", [(0, 28.0, 0), (32.5, 53.5, 0)],
           [(0.0, 19.6, 1.5), (19.6, 26.1, 21.1), (26.1, 37.0, 28.0), (37.0, 46.4, 38.9)]),
    "t2": (T2, "p2_02.wav", [(0, 22.0, 0), (25.5, 33.0, 0)],
           [(0.0, 21.0, 1.5), (21.0, 25.5, 23.5)]),
    "t3": (T3, "p2_03.wav", [(0, 41.0, 5.0), (41.0, 58.0, 0)],
           [(0.0, 3.36, 4.0, {"src": "audiocorrecion.wav"}),      # frase regrabada: "clip de demostración"
            (4.4, 11.2, 8.0), (11.2, 21.7, 22.0), (21.7, 39.2, 32.5),
            (39.2, 52.4, 50.3, {"tempo": 1.1})]),                 # sin acelerar, la última frase no cabe
}
# Aviso en pantalla mientras se ve el clip de VeloVision: (toma, texto, desde_s, hasta_s)
AVISO = ("t3", "Clip de demostración generado con IA", 0, 31)
FUENTE = "C\\:/Windows/Fonts/arialbd.ttf"
MUSICA_VOL = 0.124           # −5 dB respecto a la primera versión (0.22)

ORDEN = ["intro", "01", "02", "03", "04", "t1", "t2", "05", "t3", "06"]


def run(args):
    r = subprocess.run([FF, "-hide_banner", "-v", "error", "-y", *args], capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr)


def enc(out):
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS),
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", str(out)]


def fades(dur, final=False):
    out = f"fade=t=in:st=0:d={FADE}:color=white,"
    out += f"fade=t=out:st={dur - 1.5:.2f}:d=1.5" if final else f"fade=t=out:st={dur - FADE:.2f}:d={FADE}:color=white"
    return out


def aviso(key):
    toma, texto, a, b = AVISO
    if key != toma:
        return ""
    return (f"drawtext=fontfile='{FUENTE}':text='{texto}':fontsize=34:fontcolor=white:"
            f"box=1:boxcolor=0x1B2A41@0.85:boxborderw=18:x=140:y=h-110:enable='between(t,{a},{b})',")


def intro(out):
    ini, fin = 1.2, 6.9                  # corta en el destello blanco, que enlaza con la lámina 1
    dur = fin - ini
    run(["-ss", str(ini), "-t", str(dur), "-i", str(D / "intro_anime_fixed.mp4"),
         "-vf", f"scale={W}:{H},fade=t=in:st=0:d=0.4",
         "-af", f"afade=t=in:d=0.2,afade=t=out:st={dur - 0.5:.2f}:d=0.5,{VOZ}", *enc(out)])


def lamina(key, out):
    png, aud, a, b = LAMINAS[key]
    dur = b - a + (1.5 if key == "06" else 0)
    run(["-loop", "1", "-framerate", str(FPS), "-t", f"{dur:.2f}", "-i", str(D / png),
         "-ss", str(a), "-t", f"{b - a:.2f}", "-i", str(D / aud),
         "-vf", f"scale={W}:{H}:flags=lanczos,{fades(dur, key == '06')}",
         "-af", f"{VOZ},apad", "-t", f"{dur:.2f}", *enc(out)])


def toma(key, out):
    vid, aud, tramos, chunks = TOMAS[key]
    srcs = [aud] + sorted({c[3]["src"] for c in chunks if len(c) > 3 and "src" in c[3]})
    f, vs, dur = [], [], 0.0
    for i, (a, b, hold) in enumerate(tramos):
        pad = f",tpad=stop_mode=clone:stop_duration={hold}" if hold else ""
        f.append(f"[0:v]trim={a}:{b},setpts=PTS-STARTPTS{pad}[v{i}]")
        vs.append(f"[v{i}]")
        dur += b - a + hold
    f.append(f"{''.join(vs)}concat=n={len(vs)}:v=1:a=0,fps={FPS},"
             f"scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos,"
             f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color={FONDO},{aviso(key)}{fades(dur)}[v]")
    ac = []
    for i, (a, b, at, *opt) in enumerate(chunks):
        opt = opt[0] if opt else {}
        ms = int(at * 1000)
        tempo = f",atempo={opt['tempo']}" if "tempo" in opt else ""
        f.append(f"[{1 + srcs.index(opt.get('src', aud))}:a]atrim={a}:{b},asetpts=PTS-STARTPTS{tempo},"
                 f"aformat=sample_rates=48000:channel_layouts=stereo,adelay={ms}|{ms}[a{i}]")
        ac.append(f"[a{i}]")
    f.append(f"{''.join(ac)}amix=inputs={len(ac)}:normalize=0,{VOZ},apad,atrim=0:{dur:.2f}[a]")
    entradas = [x for s in srcs for x in ("-i", str(D / s))]
    run(["-i", str(D / vid), *entradas, "-filter_complex", ";".join(f),
         "-map", "[v]", "-map", "[a]", "-t", f"{dur:.2f}", *enc(out)])


def main():
    partes = []
    for k in ORDEN:
        out = TMP / f"{k}.mp4"
        if k == "intro":
            intro(out)
        elif k.startswith("t"):
            toma(k, out)
        else:
            lamina(k, out)
        partes.append(out)
        print("listo", k, flush=True)
    lista = TMP / "lista.txt"
    lista.write_text("".join(f"file '{p.as_posix()}'\n" for p in partes), encoding="utf-8")
    sin = D / "video_final_sin_musica.mp4"
    run(["-f", "concat", "-safe", "0", "-i", str(lista), "-c", "copy", "-movflags", "+faststart", str(sin)])
    # Música: entra al terminar la intro, baja bajo la voz (ducking) y se desvanece al final
    hh, mm, ss = (subprocess.run([FF, "-i", str(sin)], capture_output=True, text=True).stderr
                  .split("Duration: ")[1].split(",")[0].split(":"))
    total = int(hh) * 3600 + int(mm) * 60 + float(ss)
    ini = 5.7
    mus = (f"[1:a]atrim=0:{total - ini:.2f},asetpts=PTS-STARTPTS,loudnorm=I=-16,volume={MUSICA_VOL},"
           f"afade=t=in:d=1.5,afade=t=out:st={total - ini - 3:.2f}:d=3,adelay={int(ini * 1000)}|{int(ini * 1000)}[m];"
           "[0:a]asplit[v1][v2];[m][v2]sidechaincompress=threshold=0.03:ratio=6:attack=20:release=400[md];"
           "[v1][md]amix=inputs=2:normalize=0,alimiter=limit=0.95[a]")
    run(["-i", str(sin), "-i", str(D / "cancion.wav"), "-filter_complex", mus, "-map", "0:v", "-map", "[a]",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(D / "video_final_con_musica.mp4")])
    print(f"total {total:.2f} s")


if __name__ == "__main__":
    main()
