use std::mem::size_of;
use std::time::Duration;

use thiserror::Error;
use zerocopy::byteorder::{NetworkEndian, U16, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

type U16Be = U16<NetworkEndian>;
type U32Be = U32<NetworkEndian>;
type U64Be = U64<NetworkEndian>;

pub const HEADER_LEN: usize = size_of::<WireHeader>();
pub const RESULT_BODY_LEN: usize = size_of::<WireResultBody>();
pub const RESULT_PACKET_LEN: usize = HEADER_LEN + RESULT_BODY_LEN;
pub const BULK_STATS_BODY_LEN: usize = size_of::<WireBulkStatsBody>();
pub const BULK_STATS_PACKET_LEN: usize = HEADER_LEN + BULK_STATS_BODY_LEN;

const MAGIC: &[u8; 4] = b"BBPB";
const VERSION: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PacketKind {
    Start = 1,
    StartAck = 2,
    Rts = 3,
    Bulk = 4,
    Result = 5,
    End = 6,
    ErrorMessage = 7,
    BulkStats = 8,
}

impl TryFrom<u8> for PacketKind {
    type Error = ProtocolError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Start),
            2 => Ok(Self::StartAck),
            3 => Ok(Self::Rts),
            4 => Ok(Self::Bulk),
            5 => Ok(Self::Result),
            6 => Ok(Self::End),
            7 => Ok(Self::ErrorMessage),
            8 => Ok(Self::BulkStats),
            other => Err(ProtocolError::UnknownKind(other)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub kind: PacketKind,
    pub run_id: u64,
    pub sample_id: u32,
    pub seq: u32,
    pub bulk_len: u32,
    pub sample_count: u32,
    pub ip_packet_bytes: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResultBody {
    pub attempts: u32,
    pub lost_samples: u32,
    pub selected_sample_id: u32,
    pub accepted_samples: u32,
    pub delay_sum: Duration,
    pub dispersion: Duration,
    pub min_dispersion: Duration,
    pub capacity_mbps: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BulkStatsBody {
    pub server_issue_duration: Duration,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, Immutable, IntoBytes, KnownLayout)]
struct WireHeader {
    magic: [u8; 4],
    version: u8,
    kind: u8,
    flags: U16Be,
    run_id: U64Be,
    sample_id: U32Be,
    seq: U32Be,
    bulk_len: U32Be,
    sample_count: U32Be,
    ip_packet_bytes: U32Be,
    reserved: U32Be,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, Immutable, IntoBytes, KnownLayout)]
struct WireResultBody {
    attempts: U32Be,
    lost_samples: U32Be,
    selected_sample_id: U32Be,
    accepted_samples: U32Be,
    delay_sum_nanos: U64Be,
    dispersion_nanos: U64Be,
    min_dispersion_nanos: U64Be,
    capacity_mbps_bits: U64Be,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, Immutable, IntoBytes, KnownLayout)]
struct WireBulkStatsBody {
    server_issue_nanos: U64Be,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ProtocolError {
    #[error("packet is too short")]
    TooShort,
    #[error("packet buffer is too small")]
    BufferTooSmall,
    #[error("bad PBProbe packet magic")]
    BadMagic,
    #[error("unsupported PBProbe protocol version {0}")]
    BadVersion(u8),
    #[error("unknown PBProbe packet kind {0}")]
    UnknownKind(u8),
}

pub fn encode_header(dst: &mut [u8], header: Header) -> Result<usize, ProtocolError> {
    write_bytes(dst, WireHeader::from(header).as_bytes())
}

pub fn decode_header(src: &[u8]) -> Result<Header, ProtocolError> {
    let (wire, _) = WireHeader::read_from_prefix(src).map_err(|_| ProtocolError::TooShort)?;
    Header::try_from(wire)
}

pub fn encode_result(
    dst: &mut [u8],
    header: Header,
    body: ResultBody,
) -> Result<usize, ProtocolError> {
    if dst.len() < RESULT_PACKET_LEN {
        return Err(ProtocolError::BufferTooSmall);
    }

    let cursor = encode_header(dst, header)?;
    write_bytes(&mut dst[cursor..], WireResultBody::from(body).as_bytes())?;
    Ok(RESULT_PACKET_LEN)
}

pub fn decode_result_body(src: &[u8]) -> Result<ResultBody, ProtocolError> {
    let body_src = src.get(HEADER_LEN..).ok_or(ProtocolError::TooShort)?;
    let (wire, _) =
        WireResultBody::read_from_prefix(body_src).map_err(|_| ProtocolError::TooShort)?;
    Ok(ResultBody::from(wire))
}

pub fn encode_bulk_stats(
    dst: &mut [u8],
    header: Header,
    body: BulkStatsBody,
) -> Result<usize, ProtocolError> {
    if dst.len() < BULK_STATS_PACKET_LEN {
        return Err(ProtocolError::BufferTooSmall);
    }

    let cursor = encode_header(dst, header)?;
    write_bytes(&mut dst[cursor..], WireBulkStatsBody::from(body).as_bytes())?;
    Ok(BULK_STATS_PACKET_LEN)
}

pub fn decode_bulk_stats_body(src: &[u8]) -> Result<BulkStatsBody, ProtocolError> {
    let body_src = src.get(HEADER_LEN..).ok_or(ProtocolError::TooShort)?;
    let (wire, _) =
        WireBulkStatsBody::read_from_prefix(body_src).map_err(|_| ProtocolError::TooShort)?;
    Ok(BulkStatsBody::from(wire))
}

pub fn duration_nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

impl From<Header> for WireHeader {
    fn from(header: Header) -> Self {
        Self {
            magic: *MAGIC,
            version: VERSION,
            kind: header.kind as u8,
            flags: U16Be::ZERO,
            run_id: U64Be::new(header.run_id),
            sample_id: U32Be::new(header.sample_id),
            seq: U32Be::new(header.seq),
            bulk_len: U32Be::new(header.bulk_len),
            sample_count: U32Be::new(header.sample_count),
            ip_packet_bytes: U32Be::new(header.ip_packet_bytes),
            reserved: U32Be::ZERO,
        }
    }
}

impl TryFrom<WireHeader> for Header {
    type Error = ProtocolError;

    fn try_from(wire: WireHeader) -> Result<Self, Self::Error> {
        if wire.magic != *MAGIC {
            return Err(ProtocolError::BadMagic);
        }
        if wire.version != VERSION {
            return Err(ProtocolError::BadVersion(wire.version));
        }

        Ok(Self {
            kind: PacketKind::try_from(wire.kind)?,
            run_id: wire.run_id.get(),
            sample_id: wire.sample_id.get(),
            seq: wire.seq.get(),
            bulk_len: wire.bulk_len.get(),
            sample_count: wire.sample_count.get(),
            ip_packet_bytes: wire.ip_packet_bytes.get(),
        })
    }
}

impl From<ResultBody> for WireResultBody {
    fn from(body: ResultBody) -> Self {
        Self {
            attempts: U32Be::new(body.attempts),
            lost_samples: U32Be::new(body.lost_samples),
            selected_sample_id: U32Be::new(body.selected_sample_id),
            accepted_samples: U32Be::new(body.accepted_samples),
            delay_sum_nanos: U64Be::new(duration_nanos(body.delay_sum)),
            dispersion_nanos: U64Be::new(duration_nanos(body.dispersion)),
            min_dispersion_nanos: U64Be::new(duration_nanos(body.min_dispersion)),
            capacity_mbps_bits: U64Be::new(body.capacity_mbps.to_bits()),
        }
    }
}

impl From<WireResultBody> for ResultBody {
    fn from(wire: WireResultBody) -> Self {
        Self {
            attempts: wire.attempts.get(),
            lost_samples: wire.lost_samples.get(),
            selected_sample_id: wire.selected_sample_id.get(),
            accepted_samples: wire.accepted_samples.get(),
            delay_sum: Duration::from_nanos(wire.delay_sum_nanos.get()),
            dispersion: Duration::from_nanos(wire.dispersion_nanos.get()),
            min_dispersion: Duration::from_nanos(wire.min_dispersion_nanos.get()),
            capacity_mbps: f64::from_bits(wire.capacity_mbps_bits.get()),
        }
    }
}

impl From<BulkStatsBody> for WireBulkStatsBody {
    fn from(body: BulkStatsBody) -> Self {
        Self {
            server_issue_nanos: U64Be::new(duration_nanos(body.server_issue_duration)),
        }
    }
}

impl From<WireBulkStatsBody> for BulkStatsBody {
    fn from(wire: WireBulkStatsBody) -> Self {
        Self {
            server_issue_duration: Duration::from_nanos(wire.server_issue_nanos.get()),
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
    use std::time::Duration;

    use super::{
        BULK_STATS_PACKET_LEN, BulkStatsBody, HEADER_LEN, Header, PacketKind, RESULT_PACKET_LEN,
        ResultBody, decode_bulk_stats_body, decode_header, decode_result_body, encode_bulk_stats,
        encode_header, encode_result,
    };

    #[test]
    fn header_layout_is_stable() {
        assert_eq!(HEADER_LEN, 40);
    }

    #[test]
    fn header_round_trips() {
        let header = Header {
            kind: PacketKind::Bulk,
            run_id: 7,
            sample_id: 11,
            seq: 3,
            bulk_len: 100,
            sample_count: 200,
            ip_packet_bytes: 1500,
        };
        let mut buf = [0_u8; HEADER_LEN];

        assert_eq!(encode_header(&mut buf, header), Ok(HEADER_LEN));
        assert_eq!(decode_header(&buf), Ok(header));
    }

    #[test]
    fn result_round_trips() {
        let header = Header {
            kind: PacketKind::Result,
            run_id: 9,
            sample_id: 0,
            seq: 0,
            bulk_len: 100,
            sample_count: 200,
            ip_packet_bytes: 1500,
        };
        let body = ResultBody {
            attempts: 210,
            lost_samples: 10,
            selected_sample_id: 42,
            accepted_samples: 200,
            delay_sum: Duration::from_micros(123),
            dispersion: Duration::from_micros(1200),
            min_dispersion: Duration::from_micros(1100),
            capacity_mbps: 1000.25,
        };
        let mut buf = [0_u8; RESULT_PACKET_LEN];

        assert_eq!(encode_result(&mut buf, header, body), Ok(RESULT_PACKET_LEN));
        assert_eq!(decode_header(&buf), Ok(header));
        assert_eq!(decode_result_body(&buf), Ok(body));
    }

    #[test]
    fn bulk_stats_round_trips() {
        let header = Header {
            kind: PacketKind::BulkStats,
            run_id: 9,
            sample_id: 41,
            seq: 0,
            bulk_len: 100,
            sample_count: 200,
            ip_packet_bytes: 1500,
        };
        let body = BulkStatsBody {
            server_issue_duration: Duration::from_micros(900),
        };
        let mut buf = [0_u8; BULK_STATS_PACKET_LEN];

        assert_eq!(
            encode_bulk_stats(&mut buf, header, body),
            Ok(BULK_STATS_PACKET_LEN)
        );
        assert_eq!(decode_header(&buf), Ok(header));
        assert_eq!(decode_bulk_stats_body(&buf), Ok(body));
    }
}
