//! Bencode encoding for BitTorrent tracker responses
//!
//! Implements the subset of bencoding needed for tracker announce responses.

use std::collections::BTreeMap;

/// Parameters from a tracker announce request
#[derive(Debug, Clone)]
pub struct AnnounceParams {
    /// 20-byte info hash of the torrent
    pub info_hash: [u8; 20],
    /// 20-byte peer ID of the client
    pub peer_id: [u8; 20],
    /// Port the client is listening on
    pub port: u16,
    /// Total bytes uploaded
    pub uploaded: u64,
    /// Total bytes downloaded
    pub downloaded: u64,
    /// Bytes remaining to download
    pub left: u64,
    /// Whether to return compact peer list (6 bytes per peer)
    pub compact: bool,
    /// Optional event (started, stopped, completed)
    pub event: Option<AnnounceEvent>,
}

/// Announce event types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnounceEvent {
    Started,
    Stopped,
    Completed,
}

/// A bencoded value
#[derive(Debug, Clone)]
pub enum BencodeValue {
    Integer(i64),
    Bytes(Vec<u8>),
    List(Vec<BencodeValue>),
    Dict(BTreeMap<Vec<u8>, BencodeValue>),
}

impl BencodeValue {
    /// Create a string value from a &str
    #[inline]
    pub fn string(s: &str) -> Self {
        Self::Bytes(s.as_bytes().to_vec())
    }

    /// Create an integer value
    #[inline]
    pub fn integer(i: i64) -> Self {
        Self::Integer(i)
    }

    /// Create an empty list
    #[inline]
    pub fn list() -> Self {
        Self::List(Vec::new())
    }

    /// Create an empty dict
    #[inline]
    pub fn dict() -> Self {
        Self::Dict(BTreeMap::new())
    }

    /// Add an item to a list (builder pattern)
    #[inline]
    pub fn push(mut self, value: BencodeValue) -> Self {
        if let Self::List(ref mut list) = self {
            list.push(value);
        }
        self
    }

    /// Insert a key-value pair into a dict (builder pattern)
    #[inline]
    pub fn insert(mut self, key: &str, value: BencodeValue) -> Self {
        if let Self::Dict(ref mut dict) = self {
            dict.insert(key.as_bytes().to_vec(), value);
        }
        self
    }

    /// Encode to bencoded bytes
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.encode_into(&mut buf);
        buf
    }

    /// Encode into an existing buffer
    pub fn encode_into(&self, buf: &mut Vec<u8>) {
        match self {
            Self::Integer(i) => {
                buf.push(b'i');
                buf.extend_from_slice(i.to_string().as_bytes());
                buf.push(b'e');
            }
            Self::Bytes(bytes) => {
                buf.extend_from_slice(bytes.len().to_string().as_bytes());
                buf.push(b':');
                buf.extend_from_slice(bytes);
            }
            Self::List(list) => {
                buf.push(b'l');
                for item in list {
                    item.encode_into(buf);
                }
                buf.push(b'e');
            }
            Self::Dict(dict) => {
                buf.push(b'd');
                // BTreeMap keeps keys sorted
                for (key, value) in dict {
                    buf.extend_from_slice(key.len().to_string().as_bytes());
                    buf.push(b':');
                    buf.extend_from_slice(key);
                    value.encode_into(buf);
                }
                buf.push(b'e');
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_integer() {
        assert_eq!(BencodeValue::integer(42).encode(), b"i42e");
        assert_eq!(BencodeValue::integer(-1).encode(), b"i-1e");
        assert_eq!(BencodeValue::integer(0).encode(), b"i0e");
    }

    #[test]
    fn test_encode_string() {
        assert_eq!(BencodeValue::string("spam").encode(), b"4:spam");
        assert_eq!(BencodeValue::string("").encode(), b"0:");
    }

    #[test]
    fn test_encode_list() {
        let list = BencodeValue::list()
            .push(BencodeValue::string("spam"))
            .push(BencodeValue::integer(42));
        assert_eq!(list.encode(), b"l4:spami42ee");
    }

    #[test]
    fn test_encode_dict() {
        let dict = BencodeValue::dict()
            .insert("bar", BencodeValue::string("spam"))
            .insert("foo", BencodeValue::integer(42));
        assert_eq!(dict.encode(), b"d3:bar4:spam3:fooi42ee");
    }
}
