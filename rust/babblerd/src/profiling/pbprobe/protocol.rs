use std::time::Duration;

use thiserror::Error;

pub const HEADER_LEN: usize = 40;
pub const RESULT_BODY_LEN: usize = 48;
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
    if dst.len() < HEADER_LEN {
        return Err(ProtocolError::BufferTooSmall);
    }

    let mut cursor = 0;
    put_slice(dst, &mut cursor, MAGIC.as_slice())?;
    put_slice(dst, &mut cursor, &[VERSION])?;
    put_slice(dst, &mut cursor, &[header.kind as u8])?;
    put_slice(dst, &mut cursor, &0_u16.to_be_bytes())?;
    put_slice(dst, &mut cursor, &header.run_id.to_be_bytes())?;
    put_slice(dst, &mut cursor, &header.sample_id.to_be_bytes())?;
    put_slice(dst, &mut cursor, &header.seq.to_be_bytes())?;
    put_slice(dst, &mut cursor, &header.bulk_len.to_be_bytes())?;
    put_slice(dst, &mut cursor, &header.sample_count.to_be_bytes())?;
    put_slice(dst, &mut cursor, &header.ip_packet_bytes.to_be_bytes())?;
    put_slice(dst, &mut cursor, &0_u32.to_be_bytes())?;
    Ok(cursor)
}

pub fn decode_header(src: &[u8]) -> Result<Header, ProtocolError> {
    let mut cursor = 0;
    let magic = take(src, &mut cursor, MAGIC.len())?;
    if magic != MAGIC.as_slice() {
        return Err(ProtocolError::BadMagic);
    }

    let version = read_u8(src, &mut cursor)?;
    if version != VERSION {
        return Err(ProtocolError::BadVersion(version));
    }

    let kind = PacketKind::try_from(read_u8(src, &mut cursor)?)?;
    let _flags = read_u16(src, &mut cursor)?;
    let run_id = read_u64(src, &mut cursor)?;
    let sample_id = read_u32(src, &mut cursor)?;
    let seq = read_u32(src, &mut cursor)?;
    let bulk_len = read_u32(src, &mut cursor)?;
    let sample_count = read_u32(src, &mut cursor)?;
    let ip_packet_bytes = read_u32(src, &mut cursor)?;
    let _reserved = read_u32(src, &mut cursor)?;

    Ok(Header {
        kind,
        run_id,
        sample_id,
        seq,
        bulk_len,
        sample_count,
        ip_packet_bytes,
    })
}

pub fn encode_result(
    dst: &mut [u8],
    header: Header,
    body: ResultBody,
) -> Result<usize, ProtocolError> {
    if dst.len() < RESULT_PACKET_LEN {
        return Err(ProtocolError::BufferTooSmall);
    }

    let mut cursor = encode_header(dst, header)?;
    put_slice(dst, &mut cursor, &body.attempts.to_be_bytes())?;
    put_slice(dst, &mut cursor, &body.lost_samples.to_be_bytes())?;
    put_slice(dst, &mut cursor, &body.selected_sample_id.to_be_bytes())?;
    put_slice(dst, &mut cursor, &body.accepted_samples.to_be_bytes())?;
    put_slice(
        dst,
        &mut cursor,
        &duration_nanos(body.delay_sum).to_be_bytes(),
    )?;
    put_slice(
        dst,
        &mut cursor,
        &duration_nanos(body.dispersion).to_be_bytes(),
    )?;
    put_slice(
        dst,
        &mut cursor,
        &duration_nanos(body.min_dispersion).to_be_bytes(),
    )?;
    put_slice(
        dst,
        &mut cursor,
        &body.capacity_mbps.to_bits().to_be_bytes(),
    )?;
    Ok(cursor)
}

pub fn decode_result_body(src: &[u8]) -> Result<ResultBody, ProtocolError> {
    let mut cursor = HEADER_LEN;
    let attempts = read_u32(src, &mut cursor)?;
    let lost_samples = read_u32(src, &mut cursor)?;
    let selected_sample_id = read_u32(src, &mut cursor)?;
    let accepted_samples = read_u32(src, &mut cursor)?;
    let delay_sum = Duration::from_nanos(read_u64(src, &mut cursor)?);
    let dispersion = Duration::from_nanos(read_u64(src, &mut cursor)?);
    let min_dispersion = Duration::from_nanos(read_u64(src, &mut cursor)?);
    let capacity_mbps = f64::from_bits(read_u64(src, &mut cursor)?);

    Ok(ResultBody {
        attempts,
        lost_samples,
        selected_sample_id,
        accepted_samples,
        delay_sum,
        dispersion,
        min_dispersion,
        capacity_mbps,
    })
}

pub fn duration_nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn put_slice(dst: &mut [u8], cursor: &mut usize, src: &[u8]) -> Result<(), ProtocolError> {
    let Some(end) = cursor.checked_add(src.len()) else {
        return Err(ProtocolError::BufferTooSmall);
    };
    let Some(slot) = dst.get_mut(*cursor..end) else {
        return Err(ProtocolError::BufferTooSmall);
    };
    slot.copy_from_slice(src);
    *cursor = end;
    Ok(())
}

fn take<'a>(src: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8], ProtocolError> {
    let Some(end) = cursor.checked_add(len) else {
        return Err(ProtocolError::TooShort);
    };
    let Some(chunk) = src.get(*cursor..end) else {
        return Err(ProtocolError::TooShort);
    };
    *cursor = end;
    Ok(chunk)
}

fn read_u8(src: &[u8], cursor: &mut usize) -> Result<u8, ProtocolError> {
    let chunk = take(src, cursor, 1)?;
    chunk.first().copied().ok_or(ProtocolError::TooShort)
}

fn read_u16(src: &[u8], cursor: &mut usize) -> Result<u16, ProtocolError> {
    let mut bytes = [0_u8; 2];
    let len = bytes.len();
    bytes.copy_from_slice(take(src, cursor, len)?);
    Ok(u16::from_be_bytes(bytes))
}

fn read_u32(src: &[u8], cursor: &mut usize) -> Result<u32, ProtocolError> {
    let mut bytes = [0_u8; 4];
    let len = bytes.len();
    bytes.copy_from_slice(take(src, cursor, len)?);
    Ok(u32::from_be_bytes(bytes))
}

fn read_u64(src: &[u8], cursor: &mut usize) -> Result<u64, ProtocolError> {
    let mut bytes = [0_u8; 8];
    let len = bytes.len();
    bytes.copy_from_slice(take(src, cursor, len)?);
    Ok(u64::from_be_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{
        HEADER_LEN, Header, PacketKind, RESULT_PACKET_LEN, ResultBody, decode_header,
        decode_result_body, encode_header, encode_result,
    };

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
}
