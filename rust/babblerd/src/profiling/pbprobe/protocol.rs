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
    encode_header_with_aux(dst, header, 0)
}

pub fn encode_header_with_aux(
    dst: &mut [u8],
    header: Header,
    aux: u32,
) -> Result<usize, ProtocolError> {
    write_bytes(dst, WireHeader::from_header(header, aux).as_bytes())
}

pub fn decode_header(src: &[u8]) -> Result<Header, ProtocolError> {
    decode_header_with_aux(src).map(|(header, _aux)| header)
}

pub fn decode_header_with_aux(src: &[u8]) -> Result<(Header, u32), ProtocolError> {
    let (wire, _) = WireHeader::read_from_prefix(src).map_err(|_| ProtocolError::TooShort)?;
    wire.decode()
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

pub fn duration_nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

impl WireHeader {
    fn from_header(header: Header, aux: u32) -> Self {
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
            reserved: U32Be::new(aux),
        }
    }

    fn decode(self) -> Result<(Header, u32), ProtocolError> {
        if self.magic != *MAGIC {
            return Err(ProtocolError::BadMagic);
        }
        if self.version != VERSION {
            return Err(ProtocolError::BadVersion(self.version));
        }

        Ok((
            Header {
                kind: PacketKind::try_from(self.kind)?,
                run_id: self.run_id.get(),
                sample_id: self.sample_id.get(),
                seq: self.seq.get(),
                bulk_len: self.bulk_len.get(),
                sample_count: self.sample_count.get(),
                ip_packet_bytes: self.ip_packet_bytes.get(),
            },
            self.reserved.get(),
        ))
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
        HEADER_LEN, Header, PacketKind, RESULT_PACKET_LEN, ResultBody, decode_header,
        decode_header_with_aux, decode_result_body, encode_header, encode_header_with_aux,
        encode_result,
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
    fn header_aux_round_trips() {
        let header = Header {
            kind: PacketKind::Bulk,
            run_id: 7,
            sample_id: 11,
            seq: 100,
            bulk_len: 100,
            sample_count: 200,
            ip_packet_bytes: 1500,
        };
        let mut buf = [0_u8; HEADER_LEN];

        assert_eq!(
            encode_header_with_aux(&mut buf, header, 12_345),
            Ok(HEADER_LEN)
        );
        assert_eq!(decode_header_with_aux(&buf), Ok((header, 12_345)));
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
}
