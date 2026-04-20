//! Typed representation of commands sent to `babeld`'s local socket.
//!
//! This is the outbound counterpart to [`crate::babel::line`]:
//!
//! - [`crate::babel::line`] models what `babeld` emits
//! - this module models the runtime control lines that `babblerd` sends
//!
//! The scope here is intentionally narrow: this module only models the local-socket
//! commands that `babblerd` currently issues at runtime.
//!
//! NOTE: spawn-time `-C` configuration strings are still assembled in the runtime layer for now.
//!       If you want to push the protocol model further, the next obvious extraction is a typed
//!       configuration/config-statement layer rather than more runtime socket commands.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BabelCommand {
    Dump,
    Monitor,
    Unmonitor,
    Quit,
    Interface(Box<str>),
}

impl BabelCommand {
    /// Encode this command for the local `babeld` socket, including line framing.
    #[must_use]
    pub fn encode(&self) -> String {
        format!("{self}\n")
    }
}

impl fmt::Display for BabelCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dump => f.write_str("dump"),
            Self::Monitor => f.write_str("monitor"),
            Self::Unmonitor => f.write_str("unmonitor"),
            Self::Quit => f.write_str("quit"),
            Self::Interface(ifname) => write!(f, "interface {ifname}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BabelCommand;

    #[test]
    fn renders_commands() {
        assert_eq!(BabelCommand::Dump.to_string(), "dump");
        assert_eq!(BabelCommand::Monitor.to_string(), "monitor");
        assert_eq!(BabelCommand::Unmonitor.to_string(), "unmonitor");
        assert_eq!(BabelCommand::Quit.to_string(), "quit");
        assert_eq!(
            BabelCommand::Interface("en2".into()).to_string(),
            "interface en2"
        );
    }

    #[test]
    fn encodes_commands() {
        assert_eq!(BabelCommand::Dump.encode(), "dump\n");
        assert_eq!(BabelCommand::Monitor.encode(), "monitor\n");
        assert_eq!(
            BabelCommand::Interface("en2".into()).encode(),
            "interface en2\n"
        );
    }
}
