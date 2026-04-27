use std::collections::BTreeSet;
use std::mem::size_of;

use thiserror::Error;
use zerocopy::byteorder::little_endian::{U16, U32};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

pub const DEFAULT_NTB_MAX_SIZE: usize = 16 * 1024;
pub const DEFAULT_NTB_MAX_SIZE_U32: u32 = 16 * 1024;
pub const DEFAULT_DATAGRAM_ALIGNMENT: usize = 4;
pub const ETHERNET_HEADER_LEN: usize = 14;
pub const ETHERNET_HEADER_LEN_U16: u16 = 14;

const NTH16_SIGNATURE: u32 = u32::from_le_bytes(*b"NCMH");
const NDP16_NO_CRC_SIGNATURE: u32 = u32::from_le_bytes(*b"NCM0");
const NDP16_CRC_SIGNATURE: u32 = u32::from_le_bytes(*b"NCM1");
const NTH16_LEN: usize = size_of::<Nth16>();
const NDP16_LEN: usize = size_of::<Ndp16>();
const DPE16_LEN: usize = size_of::<Dpe16>();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NtbBuildConfig {
    pub max_size: usize,
    pub datagram_alignment: usize,
    pub datagram_remainder: usize,
}

impl Default for NtbBuildConfig {
    fn default() -> Self {
        Self {
            max_size: DEFAULT_NTB_MAX_SIZE,
            datagram_alignment: DEFAULT_DATAGRAM_ALIGNMENT,
            datagram_remainder: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NtbParseConfig {
    pub max_size: usize,
    pub datagram_alignment: usize,
    pub min_datagram_size: usize,
}

impl Default for NtbParseConfig {
    fn default() -> Self {
        Self {
            max_size: DEFAULT_NTB_MAX_SIZE,
            datagram_alignment: 1,
            min_datagram_size: ETHERNET_HEADER_LEN,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct ParsedNtb<'a> {
    pub sequence: u16,
    pub frames: Vec<&'a [u8]>,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum NcmError {
    #[error("NTB is shorter than NTH16 header")]
    ShortNtb,
    #[error("NTB length {actual} exceeds configured maximum {max}")]
    NtbTooLarge { actual: usize, max: usize },
    #[error("unexpected NTH16 signature {0:#010x}")]
    InvalidNthSignature(u32),
    #[error("unexpected NTH16 header length {0}")]
    InvalidNthLength(u16),
    #[error("NTH16 block length {block_length} is outside transfer length {transfer_length}")]
    InvalidBlockLength {
        block_length: usize,
        transfer_length: usize,
    },
    #[error("NTH16 has no NDP index")]
    MissingNdp,
    #[error("descriptor at offset {offset} with length {length} exceeds NTB length {ntb_length}")]
    OutOfBounds {
        offset: usize,
        length: usize,
        ntb_length: usize,
    },
    #[error("unexpected NDP16 signature {0:#010x}")]
    InvalidNdpSignature(u32),
    #[error("CRC NDP16 is not supported")]
    CrcUnsupported,
    #[error("invalid NDP16 length {0}")]
    InvalidNdpLength(u16),
    #[error("loop in NDP16 chain at offset {0}")]
    NdpLoop(usize),
    #[error("NDP16 is missing a zero terminator")]
    MissingDpeTerminator,
    #[error("datagram index {index} is not aligned to {alignment} bytes")]
    MisalignedDatagram { index: usize, alignment: usize },
    #[error("datagram index {index} length {length} exceeds NTB length {ntb_length}")]
    DatagramOutOfBounds {
        index: usize,
        length: usize,
        ntb_length: usize,
    },
    #[error("datagram length {actual} is smaller than minimum {minimum}")]
    DatagramTooShort { actual: usize, minimum: usize },
    #[error("cannot build NTB with no Ethernet frames")]
    NoFrames,
    #[error("too many Ethernet frames for a single NDP16: {0}")]
    TooManyFrames(usize),
    #[error("Ethernet frame length {0} does not fit in DPE16")]
    FrameTooLarge(usize),
    #[error("built NTB length {actual} exceeds configured maximum {max}")]
    BuiltNtbTooLarge { actual: usize, max: usize },
}

#[derive(Clone, Copy, Debug, FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned)]
#[repr(C)]
struct Nth16 {
    dw_signature: U32,
    w_header_length: U16,
    w_sequence: U16,
    w_block_length: U16,
    w_ndp_index: U16,
}

#[derive(Clone, Copy, Debug, FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned)]
#[repr(C)]
struct Ndp16 {
    dw_signature: U32,
    w_length: U16,
    w_next_ndp_index: U16,
}

#[derive(Clone, Copy, Debug, FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned)]
#[repr(C)]
struct Dpe16 {
    w_datagram_index: U16,
    w_datagram_length: U16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FramePlan {
    offset: usize,
    length: usize,
}

pub fn parse_ntb16<'a>(bytes: &'a [u8], config: NtbParseConfig) -> Result<ParsedNtb<'a>, NcmError> {
    if bytes.len() > config.max_size {
        return Err(NcmError::NtbTooLarge {
            actual: bytes.len(),
            max: config.max_size,
        });
    }
    let nth = read_exact::<Nth16>(bytes, 0).ok_or(NcmError::ShortNtb)?;
    if nth.dw_signature.get() != NTH16_SIGNATURE {
        return Err(NcmError::InvalidNthSignature(nth.dw_signature.get()));
    }
    if nth.w_header_length.get() as usize != NTH16_LEN {
        return Err(NcmError::InvalidNthLength(nth.w_header_length.get()));
    }

    let block_length = usize::from(nth.w_block_length.get());
    if !(NTH16_LEN..=bytes.len()).contains(&block_length) {
        return Err(NcmError::InvalidBlockLength {
            block_length,
            transfer_length: bytes.len(),
        });
    }
    let ntb = &bytes[..block_length];
    let mut ndp_index = usize::from(nth.w_ndp_index.get());
    if ndp_index == 0 {
        return Err(NcmError::MissingNdp);
    }

    let mut visited = BTreeSet::new();
    let mut frames = Vec::new();
    while ndp_index != 0 {
        if !visited.insert(ndp_index) {
            return Err(NcmError::NdpLoop(ndp_index));
        }
        parse_ndp16(ntb, ndp_index, config, &mut frames)?;
        let ndp = read_bounded::<Ndp16>(ntb, ndp_index)?;
        ndp_index = usize::from(ndp.w_next_ndp_index.get());
    }

    Ok(ParsedNtb {
        sequence: nth.w_sequence.get(),
        frames,
    })
}

pub fn build_ntb16(
    sequence: u16,
    frames: &[&[u8]],
    config: NtbBuildConfig,
) -> Result<Vec<u8>, NcmError> {
    if frames.is_empty() {
        return Err(NcmError::NoFrames);
    }
    if frames.len() > (usize::from(u16::MAX) / DPE16_LEN).saturating_sub(3) {
        return Err(NcmError::TooManyFrames(frames.len()));
    }

    let ndp_index = NTH16_LEN;
    let ndp_entries_len = frames
        .len()
        .checked_add(1)
        .and_then(|entries| entries.checked_mul(DPE16_LEN))
        .ok_or(NcmError::BuiltNtbTooLarge {
            actual: usize::MAX,
            max: usize::from(u16::MAX),
        })?;
    let ndp_length = NDP16_LEN
        .checked_add(ndp_entries_len)
        .ok_or(NcmError::BuiltNtbTooLarge {
            actual: usize::MAX,
            max: usize::from(u16::MAX),
        })?;
    if ndp_length > usize::from(u16::MAX) {
        return Err(NcmError::BuiltNtbTooLarge {
            actual: ndp_length,
            max: usize::from(u16::MAX),
        });
    }
    let data_start = checked_align_up_to_remainder(
        ndp_index
            .checked_add(ndp_length)
            .ok_or(NcmError::BuiltNtbTooLarge {
                actual: usize::MAX,
                max: usize::from(u16::MAX),
            })?,
        config.datagram_alignment,
        config.datagram_remainder,
    )?;
    let mut entries = Vec::with_capacity(frames.len());
    let mut frame_plans = Vec::with_capacity(frames.len());
    let mut next_offset = data_start;

    for frame in frames {
        let frame_len = frame.len();
        if frame_len > usize::from(u16::MAX) {
            return Err(NcmError::FrameTooLarge(frame_len));
        }
        let frame_offset = checked_align_up_to_remainder(
            next_offset,
            config.datagram_alignment,
            config.datagram_remainder,
        )?;
        let frame_end = frame_offset
            .checked_add(frame_len)
            .ok_or(NcmError::BuiltNtbTooLarge {
                actual: usize::MAX,
                max: config.max_size,
            })?;
        if frame_end > config.max_size {
            return Err(NcmError::BuiltNtbTooLarge {
                actual: frame_end,
                max: config.max_size,
            });
        }
        if frame_end > usize::from(u16::MAX) {
            return Err(NcmError::BuiltNtbTooLarge {
                actual: frame_end,
                max: usize::from(u16::MAX),
            });
        }
        entries.push(Dpe16 {
            w_datagram_index: U16::new(u16::try_from(frame_offset).map_err(|_| {
                NcmError::BuiltNtbTooLarge {
                    actual: frame_offset,
                    max: usize::from(u16::MAX),
                }
            })?),
            w_datagram_length: U16::new(
                u16::try_from(frame_len).map_err(|_| NcmError::FrameTooLarge(frame_len))?,
            ),
        });
        frame_plans.push(FramePlan {
            offset: frame_offset,
            length: frame_len,
        });
        next_offset = frame_end;
    }

    if next_offset > config.max_size {
        return Err(NcmError::BuiltNtbTooLarge {
            actual: next_offset,
            max: config.max_size,
        });
    }
    if next_offset > usize::from(u16::MAX) {
        return Err(NcmError::BuiltNtbTooLarge {
            actual: next_offset,
            max: usize::from(u16::MAX),
        });
    }

    let mut out = vec![0; next_offset];
    for (frame, plan) in frames.iter().zip(frame_plans) {
        let frame_end = plan.offset + plan.length;
        out[plan.offset..frame_end].copy_from_slice(frame);
    }

    let nth = Nth16 {
        dw_signature: U32::new(NTH16_SIGNATURE),
        w_header_length: U16::new(u16::try_from(NTH16_LEN).expect("NTH16 length fits u16")),
        w_sequence: U16::new(sequence),
        w_block_length: U16::new(u16::try_from(next_offset).expect("checked above")),
        w_ndp_index: U16::new(u16::try_from(ndp_index).expect("NDP index fits u16")),
    };
    out[..NTH16_LEN].copy_from_slice(nth.as_bytes());

    let ndp = Ndp16 {
        dw_signature: U32::new(NDP16_NO_CRC_SIGNATURE),
        w_length: U16::new(u16::try_from(ndp_length).expect("NDP length fits u16")),
        w_next_ndp_index: U16::new(0),
    };
    out[ndp_index..ndp_index + NDP16_LEN].copy_from_slice(ndp.as_bytes());

    let mut entry_offset = ndp_index + NDP16_LEN;
    for entry in entries {
        out[entry_offset..entry_offset + DPE16_LEN].copy_from_slice(entry.as_bytes());
        entry_offset += DPE16_LEN;
    }
    let terminator = Dpe16 {
        w_datagram_index: U16::new(0),
        w_datagram_length: U16::new(0),
    };
    out[entry_offset..entry_offset + DPE16_LEN].copy_from_slice(terminator.as_bytes());

    Ok(out)
}

fn parse_ndp16<'a>(
    ntb: &'a [u8],
    ndp_index: usize,
    config: NtbParseConfig,
    frames: &mut Vec<&'a [u8]>,
) -> Result<(), NcmError> {
    let ndp = read_bounded::<Ndp16>(ntb, ndp_index)?;
    match ndp.dw_signature.get() {
        NDP16_NO_CRC_SIGNATURE => {}
        NDP16_CRC_SIGNATURE => return Err(NcmError::CrcUnsupported),
        signature => return Err(NcmError::InvalidNdpSignature(signature)),
    }

    let length = usize::from(ndp.w_length.get());
    if length < NDP16_LEN + DPE16_LEN || length % DPE16_LEN != 0 {
        return Err(NcmError::InvalidNdpLength(ndp.w_length.get()));
    }
    let end = checked_range(ntb, ndp_index, length)?;
    let mut saw_terminator = false;

    let entries = &ntb[ndp_index + NDP16_LEN..end];
    for entry in entries.chunks_exact(DPE16_LEN) {
        let dpe = Dpe16::read_from_bytes(entry).expect("chunks_exact yields DPE16-sized slices");
        let index = usize::from(dpe.w_datagram_index.get());
        let length = usize::from(dpe.w_datagram_length.get());
        if index == 0 && length == 0 {
            saw_terminator = true;
            break;
        }
        if config.datagram_alignment > 1 && index % config.datagram_alignment != 0 {
            return Err(NcmError::MisalignedDatagram {
                index,
                alignment: config.datagram_alignment,
            });
        }
        if length < config.min_datagram_size {
            return Err(NcmError::DatagramTooShort {
                actual: length,
                minimum: config.min_datagram_size,
            });
        }
        let frame_end = index
            .checked_add(length)
            .filter(|end| *end <= ntb.len())
            .ok_or(NcmError::DatagramOutOfBounds {
                index,
                length,
                ntb_length: ntb.len(),
            })?;
        frames.push(&ntb[index..frame_end]);
    }

    if !saw_terminator {
        return Err(NcmError::MissingDpeTerminator);
    }

    Ok(())
}

fn read_bounded<T>(bytes: &[u8], offset: usize) -> Result<T, NcmError>
where
    T: FromBytes + KnownLayout + Immutable,
{
    let end = checked_range(bytes, offset, size_of::<T>())?;
    Ok(T::read_from_bytes(&bytes[offset..end]).expect("checked exact struct size"))
}

fn read_exact<T>(bytes: &[u8], offset: usize) -> Option<T>
where
    T: FromBytes + KnownLayout + Immutable,
{
    let end = offset.checked_add(size_of::<T>())?;
    bytes
        .get(offset..end)
        .and_then(|slice| T::read_from_bytes(slice).ok())
}

fn checked_range(bytes: &[u8], offset: usize, length: usize) -> Result<usize, NcmError> {
    offset
        .checked_add(length)
        .filter(|end| *end <= bytes.len())
        .ok_or(NcmError::OutOfBounds {
            offset,
            length,
            ntb_length: bytes.len(),
        })
}

fn checked_align_up_to_remainder(
    value: usize,
    alignment: usize,
    remainder: usize,
) -> Result<usize, NcmError> {
    if alignment <= 1 {
        return Ok(value);
    }
    let remainder = remainder % alignment;
    let current = value % alignment;
    if current == remainder {
        Ok(value)
    } else {
        let delta = if current < remainder {
            remainder - current
        } else {
            alignment - current + remainder
        };
        value.checked_add(delta).ok_or(NcmError::BuiltNtbTooLarge {
            actual: usize::MAX,
            max: usize::from(u16::MAX),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ethernet_frame(seed: u8, payload_len: usize) -> Vec<u8> {
        let mut frame = vec![0; ETHERNET_HEADER_LEN + payload_len];
        for (idx, byte) in frame.iter_mut().enumerate() {
            *byte = seed.wrapping_add(u8::try_from(idx % 251).unwrap());
        }
        frame
    }

    #[test]
    fn tx_round_trips_single_frame() {
        let frame = ethernet_frame(0x10, 46);
        let ntb = build_ntb16(7, &[&frame], NtbBuildConfig::default()).unwrap();
        let parsed = parse_ntb16(&ntb, NtbParseConfig::default()).unwrap();

        assert_eq!(parsed.sequence, 7);
        assert_eq!(parsed.frames, vec![frame.as_slice()]);
    }

    #[test]
    fn tx_round_trips_multiple_frames() {
        let first = ethernet_frame(0x20, 46);
        let second = ethernet_frame(0x40, 100);
        let ntb = build_ntb16(8, &[&first, &second], NtbBuildConfig::default()).unwrap();
        let parsed = parse_ntb16(&ntb, NtbParseConfig::default()).unwrap();

        assert_eq!(parsed.frames, vec![first.as_slice(), second.as_slice()]);
    }

    #[test]
    fn rejects_bad_nth_signature() {
        let frame = ethernet_frame(0x30, 46);
        let mut ntb = build_ntb16(1, &[&frame], NtbBuildConfig::default()).unwrap();
        ntb[0] = b'X';

        assert!(matches!(
            parse_ntb16(&ntb, NtbParseConfig::default()),
            Err(NcmError::InvalidNthSignature(_))
        ));
    }

    #[test]
    fn rejects_datagram_out_of_bounds() {
        let frame = ethernet_frame(0x40, 46);
        let mut ntb = build_ntb16(1, &[&frame], NtbBuildConfig::default()).unwrap();
        let dpe_length_offset = NTH16_LEN + NDP16_LEN + 2;
        ntb[dpe_length_offset..dpe_length_offset + 2].copy_from_slice(&u16::MAX.to_le_bytes());

        assert!(matches!(
            parse_ntb16(&ntb, NtbParseConfig::default()),
            Err(NcmError::DatagramOutOfBounds { .. })
        ));
    }

    #[test]
    fn rejects_missing_dpe_terminator() {
        let frame = ethernet_frame(0x50, 46);
        let mut ntb = build_ntb16(1, &[&frame], NtbBuildConfig::default()).unwrap();
        let first_entry_offset = NTH16_LEN + NDP16_LEN;
        let terminator_offset = NTH16_LEN + NDP16_LEN + DPE16_LEN;
        let first_entry = ntb[first_entry_offset..first_entry_offset + DPE16_LEN].to_vec();
        ntb[terminator_offset..terminator_offset + DPE16_LEN].copy_from_slice(&first_entry);

        assert!(matches!(
            parse_ntb16(&ntb, NtbParseConfig::default()),
            Err(NcmError::MissingDpeTerminator)
        ));
    }

    #[test]
    fn rejects_ntb_larger_than_configured_max() {
        let frame = ethernet_frame(0x60, 46);
        let ntb = build_ntb16(1, &[&frame], NtbBuildConfig::default()).unwrap();
        let config = NtbParseConfig {
            max_size: ntb.len() - 1,
            ..NtbParseConfig::default()
        };

        assert!(matches!(
            parse_ntb16(&ntb, config),
            Err(NcmError::NtbTooLarge { .. })
        ));
    }

    #[test]
    fn builder_enforces_max_size() {
        let frame = ethernet_frame(0x70, 100);
        let config = NtbBuildConfig {
            max_size: 32,
            ..NtbBuildConfig::default()
        };

        assert!(matches!(
            build_ntb16(1, &[&frame], config),
            Err(NcmError::BuiltNtbTooLarge { .. })
        ));
    }
}
