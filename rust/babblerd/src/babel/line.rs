//! Typed representation of lines emitted by `babeld`'s local socket.
//!
//! This module models the inbound side of the Babel local control protocol:
//!
//! - [`BabelLine`] is one parsed wire line.
//! - [`HeaderLine`] covers the connection prelude.
//! - [`Status`] covers command completion lines such as `ok`, `bad`, and `no ...`.
//! - [`Event`] and its associated structs cover the asynchronous routing/interface updates
//!   emitted by `dump` and `monitor`.
//!
//! The sibling parser lives in [`parse`]. Its job is to turn raw socket lines into these domain
//! types. Higher layers such as the Babel session/process code should depend on this module's
//! types, and keep raw strings only at the actual socket boundary.
//!
//! More concretely:
//!
//! - use [`parse::parse_line`] when reading from `babeld`
//! - reduce [`Event`] values into in-memory state
//! - treat [`Status`] as command acknowledgements
//! - keep outbound socket/config commands in a separate module rather than mixing them into
//!   this inbound line model

use ipnet::IpNet;
use std::net::{IpAddr, Ipv4Addr};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BabelLine {
    Header(HeaderLine),
    Status(Status),
    Event(Event),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeaderLine {
    Banner { major: u8, minor: u8 },
    Version(Box<str>),
    Host(Box<str>),
    MyId(Eui64),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    Ok,
    Bad,
    No(Option<Box<str>>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    Interface(InterfaceEvent),
    Neighbour(NeighbourEvent),
    XRoute(XRouteEvent),
    Route(RouteEvent),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    Add,
    Change,
    Flush,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterfaceEvent {
    pub kind: EventKind,
    pub ifname: Box<str>,
    pub up: bool,
    pub ipv6: Option<IpAddr>,
    pub ipv4: Option<Ipv4Addr>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NeighbourEvent {
    pub kind: EventKind,
    pub handle: u64,
    pub address: IpAddr,
    pub ifname: Box<str>,
    pub reach: u16,
    pub ureach: u16,
    pub rxcost: u32,
    pub txcost: u32,
    pub rtt_millis: Option<u32>,
    pub rttcost: Option<u32>,
    pub cost: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XRouteEvent {
    pub kind: EventKind,
    pub prefix: IpNet,
    pub from: IpNet,
    pub metric: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteEvent {
    pub kind: EventKind,
    pub handle: u64,
    pub prefix: IpNet,
    pub from: IpNet,
    pub installed: bool,
    pub id: Eui64,
    pub metric: u32,
    pub refmetric: u32,
    pub via: IpAddr,
    pub ifname: Box<str>,
}

/// An EUI-64 type aliased to [`macaddr::MacAddr8`].
pub type Eui64 = macaddr::MacAddr8;

/// Parser for `babeld`'s local socket output.
///
/// This submodule is the wire-format counterpart to the parent [`crate::babel::line`] domain
/// types. It turns raw socket text into [`BabelLine`] values.
///
/// The local socket protocol implemented in `networking-related/babeld/local.c` is line-oriented
/// ASCII. The parser is split into two layers:
///
/// - [`RawLines`] does zero-copy line framing over buffered bytes with [`memchr`].
/// - [`parse_line`] parses one complete line with [`winnow`].
/// - [`ParsedLines`] is a convenience adapter for buffered transcripts such as `dump` output.
///
/// `monitor` mode uses the exact same line grammar as `dump`; it simply keeps emitting event lines
/// after the initial snapshot.
///
/// The accepted grammar is:
///
/// ```text
/// stream     ::= (line "\n")* line?
/// line       ::= header | status | event
///
/// header     ::= banner | version | host | my-id
/// banner     ::= "BABEL " uint "." uint
/// version    ::= "version " text
/// host       ::= "host " text
/// my-id      ::= "my-id " eui64
///
/// status     ::= "ok" | "bad" | ("no" (" " text)?)
///
/// event      ::= kind " " (interface | neighbour | xroute | route)
/// kind       ::= "add" | "change" | "flush"
///
/// interface  ::= "interface " ifname " up " bool
///               (" ipv6 " ip)?
///               (" ipv4 " ipv4)?
///
/// neighbour  ::= "neighbour " hex " address " ip " if " ifname
///               " reach " hex " ureach " hex
///               " rxcost " uint " txcost " uint
///               (" rtt " millis " rttcost " uint)?
///               " cost " uint
///
/// xroute     ::= "xroute " prefix "-" prefix
///               " prefix " prefix " from " prefix " metric " uint
///
/// route      ::= "route " hex
///               " prefix " prefix " from " prefix
///               " installed " yesno
///               " id " eui64
///               " metric " uint " refmetric " uint
///               " via " ip " if " ifname
/// ```
///
/// The accepted grammar is written in a regex/BNF-ish notation:
///
/// - `e1 e2` means concatenation
/// - `e1 | e2` means choice
/// - `e*` means zero or more
/// - `e+` means one or more
/// - `e?` means optional
/// - `(e)` groups expressions
///
/// # Notes
///
/// - The `xroute` summary `prefix-from` token is parsed only to consume the wire format;
///   the later `prefix` and `from` fields are treated as the authoritative values.
/// - The parser is intentionally strict about the documented token set. Internal defensive
///   fallbacks in `babeld` such as `???` are not treated as part of the formal grammar.
pub mod parse {
    use crate::babel::line::{
        BabelLine, Eui64, Event, EventKind, HeaderLine, InterfaceEvent, NeighbourEvent, RouteEvent,
        Status, XRouteEvent,
    };
    use ipnet::IpNet;
    use memchr::memchr;
    use std::{
        net::{IpAddr, Ipv4Addr},
        str::FromStr,
    };
    use thiserror::Error;
    use winnow::{
        ascii::{dec_uint, hex_uint, space1},
        combinator::{alt, eof, opt, preceded, terminated},
        error::ContextError,
        prelude::*,
        token::{rest, take_till},
    };

    #[derive(Error, Debug)]
    pub enum ParseError {
        #[error("invalid utf8 in babeld output: {0}")]
        InvalidUtf8(#[from] std::str::Utf8Error),
        #[error("failed to parse babeld line {line:?}: {error}")]
        Syntax { line: String, error: String },
    }

    /// Zero-copy line framing for already-buffered socket output.
    ///
    /// This is the `stream = { line }` part of the grammar: framing happens first,
    /// then each line is parsed independently by `parse_line`.
    #[derive(Debug, Clone)]
    pub struct RawLines<'a> {
        remaining: &'a [u8],
    }

    impl<'a> RawLines<'a> {
        pub fn new(bytes: &'a [u8]) -> Self {
            Self { remaining: bytes }
        }
    }

    impl<'a> Iterator for RawLines<'a> {
        type Item = Result<&'a str, ParseError>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.remaining.is_empty() {
                return None;
            }

            let split = memchr(b'\n', self.remaining);
            let (line, rest_bytes) = match split {
                Some(idx) => (&self.remaining[..idx], &self.remaining[idx + 1..]),
                None => (self.remaining, &[][..]),
            };
            self.remaining = rest_bytes;

            let line = if let Some(stripped) = line.strip_suffix(b"\r") {
                stripped
            } else {
                line
            };

            Some(std::str::from_utf8(line).map_err(ParseError::InvalidUtf8))
        }
    }

    /// Convenience adapter for parsing a fully buffered transcript, e.g. a dump.
    #[derive(Debug, Clone)]
    pub struct ParsedLines<'a> {
        raw: RawLines<'a>,
    }

    impl<'a> ParsedLines<'a> {
        pub fn new(bytes: &'a [u8]) -> Self {
            Self {
                raw: RawLines::new(bytes),
            }
        }
    }

    impl<'a> Iterator for ParsedLines<'a> {
        type Item = Result<BabelLine, ParseError>;

        fn next(&mut self) -> Option<Self::Item> {
            self.raw.next().map(|line| line.and_then(parse_line))
        }
    }

    pub fn parse_line(line: &str) -> Result<BabelLine, ParseError> {
        terminated(parse_babel_line, eof)
            .parse(line)
            .map_err(|err| ParseError::Syntax {
                line: line.to_owned(),
                error: err.to_string(),
            })
    }

    fn parse_babel_line(input: &mut &str) -> ModalResult<BabelLine> {
        alt((
            parse_banner,
            parse_version,
            parse_host,
            parse_my_id,
            parse_ok,
            parse_bad,
            parse_no,
            parse_event,
        ))
        .parse_next(input)
    }

    fn parse_banner(input: &mut &str) -> ModalResult<BabelLine> {
        let _ = "BABEL ".parse_next(input)?;
        let major = dec_uint::<_, u8, _>.parse_next(input)?;
        let _ = '.'.parse_next(input)?;
        let minor = dec_uint::<_, u8, _>.parse_next(input)?;
        Ok(BabelLine::Header(HeaderLine::Banner { major, minor }))
    }

    fn parse_version(input: &mut &str) -> ModalResult<BabelLine> {
        let _ = "version ".parse_next(input)?;
        let version = Box::<str>::from(rest.parse_next(input)?);
        Ok(BabelLine::Header(HeaderLine::Version(version)))
    }

    fn parse_host(input: &mut &str) -> ModalResult<BabelLine> {
        let _ = "host ".parse_next(input)?;
        let host = Box::<str>::from(rest.parse_next(input)?);
        Ok(BabelLine::Header(HeaderLine::Host(host)))
    }

    fn parse_my_id(input: &mut &str) -> ModalResult<BabelLine> {
        let _ = "my-id ".parse_next(input)?;
        let id = parse_eui64.parse_next(input)?;
        Ok(BabelLine::Header(HeaderLine::MyId(id)))
    }

    fn parse_ok(input: &mut &str) -> ModalResult<BabelLine> {
        let _ = "ok".parse_next(input)?;
        Ok(BabelLine::Status(Status::Ok))
    }

    fn parse_bad(input: &mut &str) -> ModalResult<BabelLine> {
        let _ = "bad".parse_next(input)?;
        Ok(BabelLine::Status(Status::Bad))
    }

    fn parse_no(input: &mut &str) -> ModalResult<BabelLine> {
        let _ = "no".parse_next(input)?;
        let message = opt(preceded(space1, rest)).parse_next(input)?;
        let message = message.filter(|msg| !msg.is_empty()).map(Into::into);
        Ok(BabelLine::Status(Status::No(message)))
    }

    fn parse_event(input: &mut &str) -> ModalResult<BabelLine> {
        let kind = parse_kind.parse_next(input)?;
        let _ = ' '.parse_next(input)?;
        let entity = parse_word.parse_next(input)?;

        match entity {
            "interface" => parse_interface_event(kind, input).map(Event::Interface),
            "neighbour" => parse_neighbour_event(kind, input).map(Event::Neighbour),
            "xroute" => parse_xroute_event(kind, input).map(Event::XRoute),
            "route" => parse_route_event(kind, input).map(Event::Route),
            _ => Err(winnow::error::ErrMode::Backtrack(ContextError::new())),
        }
        .map(BabelLine::Event)
    }

    fn parse_interface_event(kind: EventKind, input: &mut &str) -> ModalResult<InterfaceEvent> {
        let _ = ' '.parse_next(input)?;
        let ifname = parse_word.parse_next(input)?;
        let _ = " up ".parse_next(input)?;
        let up = parse_bool.parse_next(input)?;
        let ipv6 = opt(preceded(" ipv6 ", parse_ip_addr)).parse_next(input)?;
        let ipv4 = opt(preceded(" ipv4 ", parse_ipv4_addr)).parse_next(input)?;

        Ok(InterfaceEvent {
            kind,
            ifname: ifname.into(),
            up,
            ipv6,
            ipv4,
        })
    }

    fn parse_neighbour_event(kind: EventKind, input: &mut &str) -> ModalResult<NeighbourEvent> {
        let _ = ' '.parse_next(input)?;
        let handle = parse_hex_u64.parse_next(input)?;
        let _ = " address ".parse_next(input)?;
        let address = parse_ip_addr.parse_next(input)?;
        let _ = " if ".parse_next(input)?;
        let ifname = parse_word.parse_next(input)?;
        let _ = " reach ".parse_next(input)?;
        let reach = parse_hex_u16.parse_next(input)?;
        let _ = " ureach ".parse_next(input)?;
        let ureach = parse_hex_u16.parse_next(input)?;
        let _ = " rxcost ".parse_next(input)?;
        let rxcost = dec_uint::<_, u32, _>.parse_next(input)?;
        let _ = " txcost ".parse_next(input)?;
        let txcost = dec_uint::<_, u32, _>.parse_next(input)?;
        let rtt = opt(parse_rtt_clause).parse_next(input)?;
        let _ = " cost ".parse_next(input)?;
        let cost = dec_uint::<_, u32, _>.parse_next(input)?;

        Ok(NeighbourEvent {
            kind,
            handle,
            address,
            ifname: ifname.into(),
            reach,
            ureach,
            rxcost,
            txcost,
            rtt_millis: rtt.map(|(millis, _)| millis),
            rttcost: rtt.map(|(_, cost)| cost),
            cost,
        })
    }

    fn parse_xroute_event(kind: EventKind, input: &mut &str) -> ModalResult<XRouteEvent> {
        let _ = ' '.parse_next(input)?;
        let _summary_prefix = parse_prefix_until('-').parse_next(input)?;
        let _ = '-'.parse_next(input)?;
        let _summary_from = parse_prefix.parse_next(input)?;
        let _ = " prefix ".parse_next(input)?;
        let prefix = parse_prefix.parse_next(input)?;
        let _ = " from ".parse_next(input)?;
        let from = parse_prefix.parse_next(input)?;
        let _ = " metric ".parse_next(input)?;
        let metric = dec_uint::<_, u32, _>.parse_next(input)?;

        Ok(XRouteEvent {
            kind,
            prefix,
            from,
            metric,
        })
    }

    fn parse_route_event<'a>(kind: EventKind, input: &mut &'a str) -> ModalResult<RouteEvent> {
        let _ = ' '.parse_next(input)?;
        let handle = parse_hex_u64.parse_next(input)?;
        let _ = " prefix ".parse_next(input)?;
        let prefix = parse_prefix.parse_next(input)?;
        let _ = " from ".parse_next(input)?;
        let from = parse_prefix.parse_next(input)?;
        let _ = " installed ".parse_next(input)?;
        let installed = parse_yes_no.parse_next(input)?;
        let _ = " id ".parse_next(input)?;
        let id = parse_eui64.parse_next(input)?;
        let _ = " metric ".parse_next(input)?;
        let metric = dec_uint::<_, u32, _>.parse_next(input)?;
        let _ = " refmetric ".parse_next(input)?;
        let refmetric = dec_uint::<_, u32, _>.parse_next(input)?;
        let _ = " via ".parse_next(input)?;
        let via = parse_ip_addr.parse_next(input)?;
        let _ = " if ".parse_next(input)?;
        let ifname = parse_word.parse_next(input)?;

        Ok(RouteEvent {
            kind,
            handle,
            prefix,
            from,
            installed,
            id,
            metric,
            refmetric,
            via,
            ifname: ifname.into(),
        })
    }

    fn parse_rtt_clause(input: &mut &str) -> ModalResult<(u32, u32)> {
        let _ = " rtt ".parse_next(input)?;
        let millis = parse_millis.parse_next(input)?;
        let _ = " rttcost ".parse_next(input)?;
        let rttcost = dec_uint::<_, u32, _>.parse_next(input)?;
        Ok((millis, rttcost))
    }

    fn parse_kind(input: &mut &str) -> ModalResult<EventKind> {
        alt((
            "add".value(EventKind::Add),
            "change".value(EventKind::Change),
            "flush".value(EventKind::Flush),
        ))
        .parse_next(input)
    }

    fn parse_bool(input: &mut &str) -> ModalResult<bool> {
        alt(("true".value(true), "false".value(false))).parse_next(input)
    }

    fn parse_yes_no(input: &mut &str) -> ModalResult<bool> {
        alt(("yes".value(true), "no".value(false))).parse_next(input)
    }

    fn parse_ip_addr(input: &mut &str) -> ModalResult<IpAddr> {
        parse_word.try_map(IpAddr::from_str).parse_next(input)
    }

    fn parse_ipv4_addr(input: &mut &str) -> ModalResult<Ipv4Addr> {
        parse_word.try_map(Ipv4Addr::from_str).parse_next(input)
    }

    fn parse_prefix(input: &mut &str) -> ModalResult<IpNet> {
        parse_word.try_map(IpNet::from_str).parse_next(input)
    }

    fn parse_prefix_until(separator: char) -> impl FnMut(&mut &str) -> ModalResult<IpNet> {
        move |input: &mut &str| {
            let token = take_till(1.., |c: char| c == separator).parse_next(input)?;
            IpNet::from_str(token)
                .map_err(|_| winnow::error::ErrMode::Backtrack(ContextError::new()))
        }
    }
    fn parse_eui64(input: &mut &str) -> ModalResult<Eui64> {
        parse_word.try_map(Eui64::from_str).parse_next(input)
    }

    fn parse_hex_u64(input: &mut &str) -> ModalResult<u64> {
        hex_uint.parse_next(input)
    }

    fn parse_hex_u16(input: &mut &str) -> ModalResult<u16> {
        hex_uint.parse_next(input)
    }

    fn parse_millis(input: &mut &str) -> ModalResult<u32> {
        let word = parse_word.parse_next(input)?;
        parse_millis_str(word).map_err(|_| winnow::error::ErrMode::Backtrack(ContextError::new()))
    }

    fn parse_word<'a>(input: &mut &'a str) -> ModalResult<&'a str> {
        take_till(1.., |c: char| c == ' ').parse_next(input)
    }

    fn parse_millis_str(value: &str) -> Result<u32, &'static str> {
        let (secs, millis) = value
            .split_once('.')
            .ok_or("missing milliseconds separator")?;
        if millis.len() != 3 || !millis.bytes().all(|b| b.is_ascii_digit()) {
            return Err("expected 3-digit millisecond suffix");
        }
        let secs = secs
            .parse::<u32>()
            .map_err(|_| "invalid seconds field in rtt value")?;
        let millis = millis
            .parse::<u32>()
            .map_err(|_| "invalid milliseconds field in rtt value")?;
        secs.checked_mul(1000)
            .and_then(|s| s.checked_add(millis))
            .ok_or("rtt value overflowed u32")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::babel::line::parse::{ParsedLines, parse_line};
    use std::str::FromStr;

    #[test]
    fn parse_header_banner() {
        assert_eq!(
            parse_line("BABEL 1.0").unwrap(),
            BabelLine::Header(HeaderLine::Banner { major: 1, minor: 0 })
        );
    }

    #[test]
    fn parse_header_metadata() {
        assert_eq!(
            parse_line("version babeld-1.13.1").unwrap(),
            BabelLine::Header(HeaderLine::Version("babeld-1.13.1".into()))
        );
        assert_eq!(
            parse_line("host e2").unwrap(),
            BabelLine::Header(HeaderLine::Host("e2".into()))
        );
        assert_eq!(
            parse_line("my-id 02:00:00:00:00:00:00:01").unwrap(),
            BabelLine::Header(HeaderLine::MyId(Eui64::new(2, 0, 0, 0, 0, 0, 0, 1)))
        );
    }

    #[test]
    fn parse_status_lines() {
        assert_eq!(parse_line("ok").unwrap(), BabelLine::Status(Status::Ok));
        assert_eq!(parse_line("bad").unwrap(), BabelLine::Status(Status::Bad));
        assert_eq!(
            parse_line("no No such interface").unwrap(),
            BabelLine::Status(Status::No(Some("No such interface".into())))
        );
    }

    #[test]
    fn parse_interface_event() {
        assert_eq!(
            parse_line("add interface en2 up true ipv6 fe80::1 ipv4 169.254.1.2").unwrap(),
            BabelLine::Event(Event::Interface(InterfaceEvent {
                kind: EventKind::Add,
                ifname: "en2".into(),
                up: true,
                ipv6: Some(IpAddr::from_str("fe80::1").unwrap()),
                ipv4: Some(Ipv4Addr::new(169, 254, 1, 2)),
            }))
        );
        assert_eq!(
            parse_line("change interface en3 up false").unwrap(),
            BabelLine::Event(Event::Interface(InterfaceEvent {
                kind: EventKind::Change,
                ifname: "en3".into(),
                up: false,
                ipv6: None,
                ipv4: None,
            }))
        );
    }

    #[test]
    fn parse_neighbour_event() {
        assert_eq!(
            parse_line(
                "add neighbour 7ffdeadbeef address fe80::1 if en2 reach 00ff ureach 000f rxcost 256 txcost 96 rtt 0.123 rttcost 32 cost 128"
            )
                .unwrap(),
            BabelLine::Event(Event::Neighbour(NeighbourEvent {
                kind: EventKind::Add,
                handle: 0x7ffdeadbeef,
                address: IpAddr::from_str("fe80::1").unwrap(),
                ifname: "en2".into(),
                reach: 0x00ff,
                ureach: 0x000f,
                rxcost: 256,
                txcost: 96,
                rtt_millis: Some(123),
                rttcost: Some(32),
                cost: 128,
            }))
        );
    }

    #[test]
    fn parse_xroute_event() {
        assert_eq!(
            parse_line(
                "add xroute fd00::1/128-fd00::/64 prefix fd00::1/128 from fd00::/64 metric 0"
            )
            .unwrap(),
            BabelLine::Event(Event::XRoute(XRouteEvent {
                kind: EventKind::Add,
                prefix: IpNet::from_str("fd00::1/128").unwrap(),
                from: IpNet::from_str("fd00::/64").unwrap(),
                metric: 0,
            }))
        );
    }

    #[test]
    fn parse_route_event() {
        assert_eq!(
            parse_line(
                "change route 7ffdeadbeef prefix fd00::1/128 from fd00::/64 installed yes id 02:00:00:00:00:00:00:01 metric 96 refmetric 0 via fe80::2 if en2"
            )
                .unwrap(),
            BabelLine::Event(Event::Route(RouteEvent {
                kind: EventKind::Change,
                handle: 0x7ffdeadbeef,
                prefix: IpNet::from_str("fd00::1/128").unwrap(),
                from: IpNet::from_str("fd00::/64").unwrap(),
                installed: true,
                id: Eui64::new(2, 0, 0, 0, 0, 0, 0, 1),
                metric: 96,
                refmetric: 0,
                via: IpAddr::from_str("fe80::2").unwrap(),
                ifname: "en2".into(),
            }))
        );
    }

    #[test]
    fn raw_lines_uses_memchr_framing() {
        let bytes = b"BABEL 1.0\nok\nadd interface en2 up false\n";
        let parsed = ParsedLines::new(bytes)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(
            parsed,
            vec![
                BabelLine::Header(HeaderLine::Banner { major: 1, minor: 0 }),
                BabelLine::Status(Status::Ok),
                BabelLine::Event(Event::Interface(InterfaceEvent {
                    kind: EventKind::Add,
                    ifname: "en2".into(),
                    up: false,
                    ipv6: None,
                    ipv4: None,
                })),
            ]
        );
    }
}
