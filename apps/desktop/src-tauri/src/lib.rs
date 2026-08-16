use std::{
    net::{SocketAddr, TcpStream},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::Mutex,
    time::Duration,
};

use tauri::Manager;

struct EngineProcess(Mutex<Option<Child>>);

impl EngineProcess {
    fn start() -> Self {
        if engine_is_running() {
            return Self(Mutex::new(None));
        }

        let child = start_development_engine().map_err(|error| {
            eprintln!("Could not start the local analysis engine: {error}");
            error
        });

        Self(Mutex::new(child.ok()))
    }

    fn stop(&self) {
        if let Ok(mut guard) = self.0.lock() {
            if let Some(child) = guard.as_mut() {
                let _ = child.kill();
                let _ = child.wait();
            }
            *guard = None;
        }
    }
}

fn engine_is_running() -> bool {
    let address = SocketAddr::from(([127, 0, 0, 1], 8000));
    TcpStream::connect_timeout(&address, Duration::from_millis(150)).is_ok()
}

fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("src-tauri must live inside apps/desktop")
        .to_path_buf()
}

fn start_development_engine() -> std::io::Result<Child> {
    let root = repository_root();
    let python = root.join(".venv/bin/python");
    Command::new(python)
        .args([
            "-m",
            "uvicorn",
            "futbol_video_analyst.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ])
        .current_dir(root)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let application = tauri::Builder::default()
        .manage(EngineProcess::start())
        .plugin(tauri_plugin_dialog::init())
        .build(tauri::generate_context!())
        .expect("error while building Futbol Video Analyst");

    application.run(|app_handle, event| {
        if matches!(event, tauri::RunEvent::Exit) {
            app_handle.state::<EngineProcess>().stop();
        }
    });
}
