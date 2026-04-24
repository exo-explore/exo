use std::mem::size_of;

use thiserror::Error;
use zerocopy::byteorder::{NetworkEndian, U16, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

type U16Be = U16<NetworkEndian>;
type U32Be = U32<NetworkEndian>;
type U64Be = U64<NetworkEndian>;

pub const HEADER_LEN: usize = size_of::<WireHeader>();
pub const SUMMARY_BODY_LEN: usize = size_of::<WireSummaryBody>();
pub const SUMMARY_PACKET_LEN: usize = HEADER_LEN + SUMMARY_BODY_LEN;

const MAGIC: &[u8; 4] = b"BBLP";
const VERSION: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PacketKind {
    EchoRequest = 1,
    EchoReply = 2,
    Train = 3,
    SummaryRequest = 4,
    SummaryReply = 5,
}

impl TryFrom<u8> for PacketKind {
    type Error = ProtocolError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::EchoRequest),
            2 => Ok(Self::EchoReply),
            3 => Ok(Self::Train),
            4 => Ok(Self::SummaryRequest),
            5 => Ok(Self::SummaryReply),
            other => Err(ProtocolError::UnknownKind(other)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub kind: PacketKind,
    pub run_id: u64,
    pub seq: u32,
    pub count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SummaryBody {
    pub received_packets: u32,
    pub received_bytes: u64,
    pub span_nanos: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, Immutable, IntoBytes, KnownLayout)]
struct WireHeader {
    magic: [u8; 4],
    version: u8,
    kind: u8,
    flags: U16Be,
    run_id: U64Be,
    seq: U32Be,
    count: U32Be,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, Immutable, IntoBytes, KnownLayout)]
struct WireSummaryBody {
    received_packets: U32Be,
    received_bytes: U64Be,
    span_nanos: U64Be,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ProtocolError {
    #[error("packet is too short")]
    TooShort,
    #[error("packet buffer is too small")]
    BufferTooSmall,
    #[error("bad profiling packet magic")]
    BadMagic,
    #[error("unsupported profiling protocol version {0}")]
    BadVersion(u8),
    #[error("unknown profiling packet kind {0}")]
    UnknownKind(u8),
}

pub fn encode_header(dst: &mut [u8], header: Header) -> Result<usize, ProtocolError> {
    write_bytes(dst, WireHeader::from_header(header).as_bytes())
}

pub fn decode_header(src: &[u8]) -> Result<Header, ProtocolError> {
    let (wire, _) = WireHeader::read_from_prefix(src).map_err(|_| ProtocolError::TooShort)?;
    wire.decode()
}

pub fn encode_summary(
    dst: &mut [u8],
    header: Header,
    body: SummaryBody,
) -> Result<usize, ProtocolError> {
    if dst.len() < SUMMARY_PACKET_LEN {
        return Err(ProtocolError::BufferTooSmall);
    }

    let cursor = encode_header(dst, header)?;
    write_bytes(&mut dst[cursor..], WireSummaryBody::from(body).as_bytes())?;
    Ok(SUMMARY_PACKET_LEN)
}

pub fn decode_summary_body(src: &[u8]) -> Result<SummaryBody, ProtocolError> {
    let body_src = src.get(HEADER_LEN..).ok_or(ProtocolError::TooShort)?;
    let (wire, _) =
        WireSummaryBody::read_from_prefix(body_src).map_err(|_| ProtocolError::TooShort)?;
    Ok(SummaryBody::from(wire))
}

impl WireHeader {
    fn from_header(header: Header) -> Self {
        Self {
            magic: *MAGIC,
            version: VERSION,
            kind: header.kind as u8,
            flags: U16Be::ZERO,
            run_id: U64Be::new(header.run_id),
            seq: U32Be::new(header.seq),
            count: U32Be::new(header.count),
        }
    }

    fn decode(self) -> Result<Header, ProtocolError> {
        if self.magic != *MAGIC {
            return Err(ProtocolError::BadMagic);
        }
        if self.version != VERSION {
            return Err(ProtocolError::BadVersion(self.version));
        }

        Ok(Header {
            kind: PacketKind::try_from(self.kind)?,
            run_id: self.run_id.get(),
            seq: self.seq.get(),
            count: self.count.get(),
        })
    }
}

impl From<SummaryBody> for WireSummaryBody {
    fn from(body: SummaryBody) -> Self {
        Self {
            received_packets: U32Be::new(body.received_packets),
            received_bytes: U64Be::new(body.received_bytes),
            span_nanos: U64Be::new(body.span_nanos),
        }
    }
}

impl From<WireSummaryBody> for SummaryBody {
    fn from(wire: WireSummaryBody) -> Self {
        Self {
            received_packets: wire.received_packets.get(),
            received_bytes: wire.received_bytes.get(),
            span_nanos: wire.span_nanos.get(),
        }
    }
}

fn write_bytes(dst: &mut [u8], src: &[u8]) -> Result<usize, ProtocolError> {
    let Some(slot) = dst.get_mut(..src.len()) else {
        return Err(ProtocolError::BufferTooSmall);
    };
    slot.copy_from_slice(src);
    Ok(src.len())
}

#[cfg(test)]
mod tests {
    use super::{
        HEADER_LEN, Header, PacketKind, SUMMARY_PACKET_LEN, SummaryBody, decode_header,
        decode_summary_body, encode_header, encode_summary,
    };

    #[test]
    fn layouts_are_stable() {
        assert_eq!(HEADER_LEN, 24);
        assert_eq!(SUMMARY_PACKET_LEN, 44);
    }

    #[test]
    fn header_round_trips() {
        let header = Header {
            kind: PacketKind::Train,
            run_id: 42,
            seq: 7,
            count: 64,
        };
        let mut buf = [0_u8; HEADER_LEN];

        let encoded = encode_header(&mut buf, header);
        assert_eq!(encoded, Ok(HEADER_LEN));
        assert_eq!(decode_header(&buf), Ok(header));
    }

    #[test]
    fn summary_round_trips() {
        let header = Header {
            kind: PacketKind::SummaryReply,
            run_id: 99,
            seq: 0,
            count: 64,
        };
        let body = SummaryBody {
            received_packets: 63,
            received_bytes: 91_476,
            span_nanos: 725_000,
        };
        let mut buf = [0_u8; SUMMARY_PACKET_LEN];

        let encoded = encode_summary(&mut buf, header, body);
        assert_eq!(encoded, Ok(SUMMARY_PACKET_LEN));
        assert_eq!(decode_header(&buf), Ok(header));
        assert_eq!(decode_summary_body(&buf), Ok(body));
    }
}
