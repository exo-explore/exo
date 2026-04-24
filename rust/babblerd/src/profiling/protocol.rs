use thiserror::Error;

pub const HEADER_LEN: usize = 24;
pub const SUMMARY_BODY_LEN: usize = 20;
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
    if dst.len() < HEADER_LEN {
        return Err(ProtocolError::BufferTooSmall);
    }

    let mut cursor = 0;
    put_slice(dst, &mut cursor, MAGIC.as_slice())?;
    put_slice(dst, &mut cursor, &[VERSION])?;
    put_slice(dst, &mut cursor, &[header.kind as u8])?;
    put_slice(dst, &mut cursor, &0_u16.to_be_bytes())?;
    put_slice(dst, &mut cursor, &header.run_id.to_be_bytes())?;
    put_slice(dst, &mut cursor, &header.seq.to_be_bytes())?;
    put_slice(dst, &mut cursor, &header.count.to_be_bytes())?;
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
    let seq = read_u32(src, &mut cursor)?;
    let count = read_u32(src, &mut cursor)?;

    Ok(Header {
        kind,
        run_id,
        seq,
        count,
    })
}

pub fn encode_summary(
    dst: &mut [u8],
    header: Header,
    body: SummaryBody,
) -> Result<usize, ProtocolError> {
    if dst.len() < SUMMARY_PACKET_LEN {
        return Err(ProtocolError::BufferTooSmall);
    }

    let mut cursor = encode_header(dst, header)?;
    put_slice(dst, &mut cursor, &body.received_packets.to_be_bytes())?;
    put_slice(dst, &mut cursor, &body.received_bytes.to_be_bytes())?;
    put_slice(dst, &mut cursor, &body.span_nanos.to_be_bytes())?;
    Ok(cursor)
}

pub fn decode_summary_body(src: &[u8]) -> Result<SummaryBody, ProtocolError> {
    let mut cursor = HEADER_LEN;
    let received_packets = read_u32(src, &mut cursor)?;
    let received_bytes = read_u64(src, &mut cursor)?;
    let span_nanos = read_u64(src, &mut cursor)?;

    Ok(SummaryBody {
        received_packets,
        received_bytes,
        span_nanos,
    })
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
    use super::{
        HEADER_LEN, Header, PacketKind, SUMMARY_PACKET_LEN, SummaryBody, decode_header,
        decode_summary_body, encode_header, encode_summary,
    };

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
