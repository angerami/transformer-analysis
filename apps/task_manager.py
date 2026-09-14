# Run this on a separate port: streamlit run apps/task_manager.py --server.port=8502
"""Browser-based process launcher and monitor."""
import html
import os
import pty
import re
import socket
import subprocess
import threading
import time
from datetime import datetime

_ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07')
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Task Manager", layout="wide", page_icon="🚀")

TASKS = {
    "streamlit_weights": {
        "label": "Weights Dashboard",
        "cmd": [
            "streamlit", "run", "dashboards/streamlit_app.py",
            "--server.port=8502",
            "--server.headless=true",
            "--server.runOnSave=true",
        ],
        "url": "http://localhost:8502",
    },
    "streamlit_correlations": {
        "label": "Correlations Dashboard",
        "cmd": [
            "streamlit", "run", "apps/correlations_app.py",
            "--server.port=8503",
            "--server.headless=true",
            "--server.runOnSave=true",
        ],
        "url": "http://localhost:8503",
    },
    "personal_website": {
        "label": "Website",
        "cmd": [
            "bundle", "exec", "jekyll", "serve", "--livereload"
        ],
        "cwd" : "/Users/angerami/Projects/angerami.github.io",
        "url": "http://127.0.0.1:4000",
    },
    "mlflow_ui": {
        "label": "MLflow UI",
        "cmd": [
            "mlflow", "ui",
            "--backend-store-uri", "sqlite:///outputs/mlruns.db",
        ],
        "url": "http://127.0.0.1:5000",
    },
}

SECTIONS = [
    ("Dashboards", ["streamlit_weights", "streamlit_correlations"]),
    ("Experiment Tracking", ["mlflow_ui"]),
    ("Services", ["personal_website"]),
]

PREVIEW_H = 160  # px — height for both iframe thumbnail and log tail preview

# ── Persistent state ──────────────────────────────────────────────────────────

@st.cache_resource
def get_state():
    return {
        "processes": {},
        "logs": {key: [] for key in TASKS},
        "start_times": {},
        "end_times": {},
    }


def _stream(key, master_fd, proc):
    """Read from pty master, handle \\r (tqdm) as in-place line updates."""
    state = get_state()
    logs = state["logs"]
    buf = ""
    try:
        while True:
            try:
                chunk = os.read(master_fd, 4096).decode(errors="ignore")
            except OSError:
                break
            chunk = _ANSI.sub("", chunk)
            buf += chunk
            while "\n" in buf or "\r" in buf:
                ni = buf.find("\n")
                ri = buf.find("\r")
                if ni == -1 or (ri != -1 and ri < ni):
                    # \r: overwrite the last partial line (tqdm-style update)
                    line, buf = buf[:ri], buf[ri + 1:]
                    if logs[key] and not logs[key][-1].endswith("\n"):
                        logs[key][-1] = line
                    elif line:
                        logs[key].append(line)
                else:
                    # \n: commit a full line
                    line, buf = buf[:ni], buf[ni + 1:]
                    logs[key].append(line + "\n")
                logs[key] = logs[key][-200:]
    finally:
        os.close(master_fd)
    if buf.strip():
        logs[key].append(buf)
    state["end_times"][key] = datetime.now()
    state["processes"].pop(key, None)


def kill_port(port):
    """Kill any process currently listening on port (handles orphaned procs)."""
    try:
        r = subprocess.run(["lsof", "-t", f"-i:{port}"], capture_output=True, text=True)
        for pid in r.stdout.strip().split():
            subprocess.run(["kill", "-TERM", pid])
        time.sleep(0.3)
    except Exception:
        pass


def start(key):
    state = get_state()
    cfg = TASKS[key]
    if cfg["url"]:
        port = int(cfg["url"].rsplit(":", 1)[-1])
        kill_port(port)
    master_fd, slave_fd = pty.openpty()
    proc = subprocess.Popen(
        cfg["cmd"],
        cwd = os.path.abspath(cfg.get("cwd", ".")),
        stdout=slave_fd,
        stderr=slave_fd,
        stdin=subprocess.DEVNULL,
        close_fds=True,
    )
    os.close(slave_fd)
    state["processes"][key] = proc
    state["logs"][key] = []
    state["start_times"][key] = datetime.now()
    state["end_times"].pop(key, None)
    threading.Thread(target=_stream, args=(key, master_fd, proc), daemon=True).start()


def stop(key):
    state = get_state()
    proc = state["processes"].pop(key, None)
    if proc:
        proc.terminate()
        state["end_times"][key] = datetime.now()


def is_running(key):
    proc = get_state()["processes"].get(key)
    return proc is not None and proc.poll() is None


def is_url_ready(url):
    try:
        port = int(url.rsplit(":", 1)[-1])
        with socket.create_connection(("localhost", port), timeout=0.5):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def fmt_time(dt):
    return dt.strftime("%H:%M:%S")


def fmt_elapsed(start):
    s = int((datetime.now() - start).total_seconds())
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def scaled_iframe(url, height):
    """Render a scaled-down preview of url that fits in height px without scrollbars."""
    scale = 0.35
    inner_h = int(height / scale)
    inner_w = int(100 / scale)
    components.html(f"""
        <html><body style="margin:0;padding:0;overflow:hidden;height:{height}px;">
        <div style="position:relative;width:100%;height:{height}px;overflow:hidden;
                    border-radius:4px;border:1px solid #333;">
            <iframe src="{url}"
                style="position:absolute;top:0;left:0;
                       width:{inner_w}%;height:{inner_h}px;
                       transform:scale({scale});transform-origin:top left;
                       border:none;"
                scrolling="no"></iframe>
        </div></body></html>""", height=height)


def log_tail_preview(log_lines, height):
    """Terminal-style log tail preview matching the iframe height."""
    content = html.escape("".join(log_lines[-20:]))
    components.html(f"""
        <html><body style="margin:0;padding:0;overflow:hidden;">
        <pre style="margin:0;padding:8px;box-sizing:border-box;
                    background:#0e1117;color:#e0e0e0;font-family:monospace;
                    font-size:11px;line-height:1.4;height:{height}px;
                    overflow:hidden;border-radius:4px;border:1px solid #333;
                    white-space:pre-wrap;word-break:break-all;">{content}</pre>
        </body></html>""", height=height)


# ── UI defaults ───────────────────────────────────────────────────────────────

st.session_state.setdefault("show_logs", False)

# ── Button colors via JS (injected once into parent frame) ────────────────────

components.html("""
<script>
function styleButtons() {
    parent.document.querySelectorAll('button').forEach(btn => {
        const t = btn.innerText.trim();
        if (t.includes('Launch All')) {
            btn.style.setProperty('background-color', '#1b5e20', 'important');
            btn.style.setProperty('border-color', '#1b5e20', 'important');
            btn.style.setProperty('color', 'white', 'important');
        }
        if (t.includes('Stop All')) {
            btn.style.setProperty('background-color', '#b71c1c', 'important');
            btn.style.setProperty('border-color', '#b71c1c', 'important');
            btn.style.setProperty('color', 'white', 'important');
        }
    });
}
styleButtons();
new MutationObserver(styleButtons).observe(parent.document.body, {childList:true, subtree:true});
</script>
""", height=0)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Controls")

    if st.button("▶▶  Launch All", use_container_width=True, type="primary"):
        for key in TASKS:
            if not is_running(key):
                start(key)
        st.rerun()

    if st.button("⏹⏹  Stop All", use_container_width=True):
        for key in TASKS:
            if is_running(key):
                stop(key)
        st.rerun()

    st.divider()
    st.subheader("View")

    log_label = "▲  Collapse Logs" if st.session_state.show_logs else "▼  Expand Logs"
    if st.button(log_label, use_container_width=True):
        st.session_state.show_logs = not st.session_state.show_logs
        st.rerun()

# ── Cards ─────────────────────────────────────────────────────────────────────

st.title("Task Manager")

any_running = False
state = get_state()

for section_label, section_keys in SECTIONS:
    st.subheader(section_label)

    for key in section_keys:
        cfg = TASKS[key]
        running = is_running(key)
        if running:
            any_running = True

        start_time = state["start_times"].get(key)
        end_time = state["end_times"].get(key)
        url_ready = running and bool(cfg["url"]) and is_url_ready(cfg["url"])
        logs = state["logs"][key]

        with st.container(border=True):
            left, right = st.columns([3, 2])

            with left:
                st.subheader(cfg["label"])

                btn_col, _ = st.columns([1, 3])
                with btn_col:
                    if running:
                        if st.button("⏹ Stop", key=f"stop_{key}", use_container_width=True):
                            stop(key)
                            st.rerun()
                    else:
                        if st.button("▶ Start", key=f"start_{key}", use_container_width=True):
                            start(key)
                            st.rerun()

                if running:
                    link = (
                        f'&nbsp;&nbsp;<a href="{cfg["url"]}" target="_blank" '
                        f'style="font-size:1.1rem">Open ↗</a>'
                        if cfg["url"] else ""
                    )
                    st.markdown(
                        f'<p style="font-size:1.1rem;margin:4px 0">🟢&nbsp; Running{link}</p>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<p style="font-size:1.1rem;margin:4px 0">🔴&nbsp; Stopped</p>',
                        unsafe_allow_html=True,
                    )

                if start_time:
                    if running:
                        st.caption(f"Started {fmt_time(start_time)} · Running for {fmt_elapsed(start_time)}")
                    elif end_time:
                        st.caption(f"Started {fmt_time(start_time)} · Ended {fmt_time(end_time)}")

            with right:
                if cfg["url"]:
                    if url_ready:
                        scaled_iframe(cfg["url"], PREVIEW_H)
                    elif running:
                        st.caption("⏳ Starting…")
                else:
                    if logs:
                        log_tail_preview(logs, PREVIEW_H)
                    elif running:
                        st.caption("⏳ Waiting for output…")

        if logs:
            with st.expander("Logs", expanded=st.session_state.show_logs):
                st.code("".join(logs[-80:]), language=None)

# ── Auto-refresh ──────────────────────────────────────────────────────────────

if any_running:
    time.sleep(3)
    st.rerun()
else:
    # Ensure we stop spinning when nothing is running
    time.sleep(0.1)
