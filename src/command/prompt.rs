use crossbeam_channel::{Receiver, Sender};
use rustyline::error::ReadlineError;

use super::parser::parse_command;
use super::{Command, CommandResponse};

/// Run the interactive prompt loop on a worker thread.
/// Blocks until EOF, Ctrl-C, or the main thread drops the channel.
pub fn run_prompt(cmd_tx: Sender<Command>, resp_rx: Receiver<CommandResponse>) {
    let mut rl = match rustyline::DefaultEditor::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("rustyline init error: {e}");
            return;
        }
    };

    loop {
        match rl.readline("rusmol> ") {
            Ok(line) => {
                let line = line.trim().to_string();
                if line.is_empty() {
                    continue;
                }
                let _ = rl.add_history_entry(&line);

                match parse_command(&line) {
                    Ok(cmd) => {
                        let is_quit = matches!(cmd, Command::Quit);
                        if cmd_tx.send(cmd).is_err() {
                            break; // main thread gone
                        }
                        // Wait for response from main thread
                        match resp_rx.recv() {
                            Ok(CommandResponse::Ok(msg)) => {
                                if !msg.is_empty() {
                                    println!("{msg}");
                                }
                            }
                            Ok(CommandResponse::Error(msg)) => {
                                eprintln!("Error: {msg}");
                            }
                            Err(_) => break, // main thread gone
                        }
                        if is_quit {
                            break;
                        }
                    }
                    Err(e) => eprintln!("Parse error: {e}"),
                }
            }
            Err(ReadlineError::Interrupted | ReadlineError::Eof) => {
                let _ = cmd_tx.send(Command::Quit);
                break;
            }
            Err(e) => {
                eprintln!("Readline error: {e}");
                break;
            }
        }
    }
}
