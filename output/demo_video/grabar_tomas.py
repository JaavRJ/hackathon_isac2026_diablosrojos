"""Graba las tomas 1 (Simulador) y 2 (Leaderboard) del guion: Playwright maneja la app y OBS graba la pantalla.

Uso: .venv/Scripts/python output/demo_video/grabar_tomas.py [1|2|12] [--dry]
Requiere la app en http://localhost:8501 y OBS abierto con WebSocket activo (puerto 4455).
"""
import re
import sys
import time

import obsws_python as obs
from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"
OBS_PASS = "1IjZ9ECPlycKtLas"
DRY = "--dry" in sys.argv
TOMAS = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "--dry" else "12"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scroll(page, total, step=120, pausa=0.08):
    page.mouse.move(900, 500)
    n = abs(total) // step
    for _ in range(n):
        page.mouse.wheel(0, step if total > 0 else -step)
        time.sleep(pausa)


def panel(page):
    return page.locator('[role="tabpanel"]:visible')


def toma_simulador(page):
    page.get_by_role("tab", name="Inicio").click()
    time.sleep(2)
    page.get_by_role("button", name="Abrir el simulador").first.click()
    time.sleep(3)
    panel(page).get_by_role("button", name="Recta alta").first.click()
    log("preset Recta alta")
    time.sleep(8)                      # veredicto
    scroll(page, 600)
    time.sleep(7)                      # tarjetas
    scroll(page, 600)
    time.sleep(7)                      # flecha del diagnóstico
    scroll(page, -1200)
    time.sleep(2)
    giro = panel(page).get_by_role("slider").nth(1)
    giro.scroll_into_view_if_needed()
    box = giro.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=15)
    giro.focus()
    for _ in range(60):                # +600 rpm: Stuff+ CDMX 89.3 → 93.9 (con −250 coincide en 89.3)
        page.keyboard.press("ArrowRight")
        time.sleep(0.08)
    log("slider de giro")
    time.sleep(5)
    analista = panel(page).get_by_text("Modo Analista").first
    analista.scroll_into_view_if_needed()
    time.sleep(2)
    analista.click()
    time.sleep(3)
    scroll(page, 900)
    time.sleep(8)                      # SHAP y evidencia
    analista.scroll_into_view_if_needed()
    time.sleep(1)
    analista.click()
    log("modo analista abierto y cerrado")
    time.sleep(3)


def toma_leaderboard(page):
    page.get_by_role("tab", name="Leaderboard").click()
    time.sleep(3)
    p = panel(page)
    p.get_by_text("ΔStuff+ CDMX", exact=True).first.hover()   # ya viene ordenado por ΔStuff+
    log("orden por ΔStuff+")
    time.sleep(5)
    scroll(page, 500)
    time.sleep(3)
    sel = p.get_by_role("combobox").last
    sel.scroll_into_view_if_needed()
    sel.click()
    time.sleep(1.5)
    page.get_by_role("option").first.click()
    log("tarjeta de lanzador")
    time.sleep(4)
    scroll(page, 700)
    time.sleep(7)                      # tarjeta y mapa del arsenal
    btn = p.get_by_role("button", name="Llevar al Simulador")
    btn.scroll_into_view_if_needed()
    time.sleep(2)
    btn.click()
    log("llevar al simulador")
    time.sleep(6)


CLIP, ESTATURA_CM, MANO, FRAME, RADAR = "aprueba.mp4", 185, "Derecho", 82, 93


def prep_velovision(page):
    """Antes de grabar: carga el clip y los datos del lanzador y espera el procesamiento."""
    page.get_by_role("tab", name="VeloVision").click()
    time.sleep(3)
    p = panel(page)
    p.locator("label").filter(has_text="cm").first.click()
    time.sleep(1.5)
    p.locator("label").filter(has_text=MANO).first.click()
    time.sleep(1.5)
    est = p.get_by_label("Estatura del lanzador (cm)")
    est.fill(str(ESTATURA_CM))
    est.press("Enter")
    time.sleep(1.5)
    p.get_by_text("Tengo una lectura de radar de la recta").click()
    time.sleep(1.5)
    velo = p.get_by_label("Velocidad de la recta del radar (mph)")
    velo.fill(str(RADAR))
    velo.press("Enter")
    time.sleep(1.5)
    p.get_by_role("combobox", name="…o un clip de demostración local").first.click()
    page.get_by_role("option", name=CLIP).click()
    log("clip elegido; procesando")
    p.get_by_role("button", name="Confirmar soltura").wait_for(timeout=300000)
    time.sleep(3)
    page.mouse.move(900, 400)
    page.keyboard.press("Home")
    time.sleep(1)


def toma_velovision(page):
    p = panel(page)
    time.sleep(4)                      # guía de captura
    btn = p.get_by_role("button", name="Confirmar soltura")
    actual = int(btn.inner_text().split("frame")[1].strip(" )"))
    barra = p.get_by_role("slider").last
    barra.scroll_into_view_if_needed()
    time.sleep(2)
    box = barra.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=15)
    barra.focus()
    def frame_actual():
        return int(p.get_by_role("button", name="Confirmar soltura").inner_text().split("frame")[1].strip(" )"))

    def ir_a(meta):                    # cada tecla recarga la app: se relee el frame tras cada paso
        for _ in range(60):
            f = frame_actual()
            if f == meta:
                return
            p.get_by_role("slider").last.focus()
            page.keyboard.press("ArrowRight" if meta > f else "ArrowLeft")
            time.sleep(0.5)

    if actual == FRAME:                # si ya detectó el frame, igual recorre la barra para que se vea
        ir_a(FRAME - 6)
        time.sleep(1)
    ir_a(FRAME)
    p.get_by_role("button", name=f"Confirmar soltura (frame {FRAME})").wait_for(timeout=30000)
    log(f"frame {actual} -> {FRAME}")
    time.sleep(3)
    p.get_by_role("button", name=f"Confirmar soltura (frame {FRAME})").click()
    time.sleep(4)
    for txt in ("Entiendo que la medida está fuera del rango", "Entiendo que no se detectó una soltura clara"):
        chk = p.get_by_text(txt)
        if chk.count():
            chk.first.click()
            time.sleep(2)
    scroll(page, 600)
    time.sleep(8)                      # tarjetas de medidas
    scroll(page, 700)
    time.sleep(10)                     # ficha por familia
    ir = p.get_by_role("button", name="Buscar referencias factibles")
    ir.scroll_into_view_if_needed()
    time.sleep(2)
    ir.click()
    log("salto al Sculptor")
    time.sleep(6)
    scroll(page, 600)
    time.sleep(7)                      # barra p25–p75
    scroll(page, 700)
    time.sleep(7)                      # comparables
    if DRY:
        t = " ".join(panel(page).inner_text().split())
        for m in re.finditer(r"factible|soltura|ft\b", t, re.I):
            log(t[max(0, m.start() - 50):m.start() + 50])


def main():
    cli = None if DRY else obs.ReqClient(host="localhost", port=4455, password=OBS_PASS, timeout=5)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=False,
                                     args=["--kiosk", "--window-position=0,0", "--disable-infobars"],
                                     ignore_default_args=["--enable-automation"])
        page = browser.new_context(no_viewport=True).new_page()
        cdp = page.context.new_cdp_session(page)
        win = cdp.send("Browser.getWindowForTarget")["windowId"]
        cdp.send("Browser.setWindowBounds", {"windowId": win, "bounds": {"windowState": "fullscreen"}})
        page.goto(URL)
        page.get_by_role("tab", name="Simulador").wait_for(timeout=60000)
        time.sleep(5)
        for t, prep, fn in (("1", None, toma_simulador), ("2", None, toma_leaderboard),
                            ("3", prep_velovision, toma_velovision)):
            if t not in TOMAS:
                continue
            page.reload()
            page.get_by_role("tab", name="Simulador").wait_for(timeout=60000)
            time.sleep(4)
            if prep:
                prep(page)
            if cli:
                cli.start_record()
            log(f"grabando toma {t}")
            time.sleep(1.5)
            try:
                fn(page)
            finally:
                if cli:
                    out = cli.stop_record().output_path
                    log(f"toma {t} guardada en {out}")
        browser.close()


if __name__ == "__main__":
    main()
