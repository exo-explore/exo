use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::time::Duration;

use color_eyre::eyre::{self, Context as _, OptionExt as _};
use nusb::descriptors::{ConfigurationDescriptor, InterfaceDescriptor, TransferType};
use nusb::{DeviceInfo, MaybeFuture as _};

pub const APPLE_VENDOR_ID: u16 = 0x05ac;
pub const APPLE_MAC_PRODUCT_ID: u16 = 0x1905;

const USB_CLASS_COMM: u8 = 0x02;
const USB_CLASS_CDC_DATA: u8 = 0x0a;
const USB_CDC_SUBCLASS_NCM: u8 = 0x0d;
const USB_CDC_PROTO_NONE: u8 = 0x00;
const USB_CDC_PROTO_NCM_DATA: u8 = 0x01;
const USB_DT_CS_INTERFACE: u8 = 0x24;
const USB_CDC_UNION_TYPE: u8 = 0x06;
const USB_CDC_ETHERNET_TYPE: u8 = 0x0f;
const USB_CDC_NCM_TYPE: u8 = 0x1a;
const STRING_TIMEOUT: Duration = Duration::from_millis(500);
const US_ENGLISH: u16 = 0x0409;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UsbSelector {
    pub vendor_id: u16,
    pub product_id: u16,
}

impl UsbSelector {
    fn matches(self, device: &DeviceInfo) -> bool {
        device.vendor_id() == self.vendor_id && device.product_id() == self.product_id
    }
}

#[derive(Debug)]
pub struct DeviceSummary {
    pub vendor_id: u16,
    pub product_id: u16,
    pub bus_id: String,
    pub busnum: u8,
    pub address: u8,
    pub speed: Option<String>,
    pub manufacturer: Option<String>,
    pub product: Option<String>,
    pub serial: Option<String>,
    pub interfaces: Vec<InterfaceSummary>,
}

#[derive(Debug)]
pub struct InterfaceSummary {
    pub number: u8,
    pub class: u8,
    pub subclass: u8,
    pub protocol: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct ProbeOptions {
    pub selector: UsbSelector,
    pub claim: bool,
    pub detach_kernel_driver: bool,
}

#[derive(Debug)]
pub struct ProbeReport {
    pub device: DeviceSummary,
    pub configuration: ConfigurationReport,
    pub ncm_pairs: Vec<NcmPair>,
    pub claim_results: Vec<ClaimResult>,
}

#[derive(Debug)]
pub struct ConfigurationReport {
    pub configuration_value: u8,
    pub num_interfaces: u8,
    pub alt_settings: Vec<InterfaceAltSummary>,
}

#[derive(Debug)]
pub struct InterfaceAltSummary {
    pub number: u8,
    pub alternate_setting: u8,
    pub class: u8,
    pub subclass: u8,
    pub protocol: u8,
    pub num_endpoints: u8,
    pub cdc: CdcInterfaceInfo,
    pub endpoints: Vec<EndpointSummary>,
}

#[derive(Clone, Debug, Default)]
pub struct CdcInterfaceInfo {
    pub union_master: Option<u8>,
    pub union_slaves: Vec<u8>,
    pub ethernet_mac_string_index: Option<u8>,
    pub ethernet_mac: Option<String>,
    pub max_segment_size: Option<u16>,
    pub ncm_version_bcd: Option<u16>,
    pub ncm_capabilities: Option<u8>,
}

#[derive(Debug)]
pub struct EndpointSummary {
    pub address: u8,
    pub direction: EndpointDirection,
    pub transfer_type: EndpointTransferType,
    pub max_packet_size: usize,
    pub interval: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EndpointDirection {
    In,
    Out,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EndpointTransferType {
    Bulk,
    Interrupt,
    Isochronous,
    Control,
}

#[derive(Clone, Debug)]
pub struct NcmPair {
    pub control_interface: u8,
    pub data_interface: u8,
    pub control_alt_setting: u8,
    pub data_alt_setting: u8,
    pub bulk_in: Option<u8>,
    pub bulk_out: Option<u8>,
    pub mac_string_index: Option<u8>,
    pub mac: Option<String>,
    pub control_has_status_endpoint: bool,
}

#[derive(Debug)]
pub struct ClaimResult {
    pub interface_number: u8,
    pub detached_kernel_driver: bool,
    pub result: Result<(), String>,
}

/// List USB devices visible to the current process.
///
/// # Errors
///
/// Returns an error if the OS USB device list cannot be read.
pub fn list_devices(selector: UsbSelector, all: bool) -> eyre::Result<Vec<DeviceSummary>> {
    let devices = nusb::list_devices()
        .wait()
        .wrap_err("failed to list USB devices")?;

    devices
        .filter(|device| all || selector.matches(device))
        .map(|device| Ok(summarize_device_info(&device)))
        .collect()
}

/// Open and inspect the selected USB device.
///
/// # Errors
///
/// Returns an error if the device is missing, cannot be opened, or its active configuration cannot
/// be read.
pub fn probe(options: ProbeOptions) -> eyre::Result<ProbeReport> {
    let device_info = find_single_device(options.selector)?;
    let device_summary = summarize_device_info(&device_info);
    let device = device_info
        .open()
        .wait()
        .wrap_err_with(|| format!("failed to open {}", format_device_id(&device_summary)))?;
    let configuration = device
        .active_configuration()
        .wrap_err("failed to read active USB configuration")?;
    let configuration_report = summarize_configuration(&device, &configuration);
    let ncm_pairs = detect_ncm_pairs(&configuration_report);

    let claim_results = if options.claim {
        claim_ncm_interfaces(&device, &ncm_pairs, options.detach_kernel_driver)
    } else {
        Vec::new()
    };

    Ok(ProbeReport {
        device: device_summary,
        configuration: configuration_report,
        ncm_pairs,
        claim_results,
    })
}

#[must_use]
pub fn render_device_list(devices: &[DeviceSummary]) -> String {
    if devices.is_empty() {
        return "no matching USB devices found\n".to_owned();
    }

    let mut out = String::new();
    for device in devices {
        let _ = writeln!(out, "{}", format_device_id(device));
        if let Some(speed) = &device.speed {
            let _ = writeln!(out, "  speed: {speed}");
        }
        if let Some(manufacturer) = &device.manufacturer {
            let _ = writeln!(out, "  manufacturer: {manufacturer}");
        }
        if let Some(product) = &device.product {
            let _ = writeln!(out, "  product: {product}");
        }
        if let Some(serial) = &device.serial {
            let _ = writeln!(out, "  serial: {serial}");
        }
        if device.interfaces.is_empty() {
            let _ = writeln!(out, "  interfaces: none reported by OS cache");
        } else {
            let _ = writeln!(out, "  interfaces:");
            for interface in &device.interfaces {
                let _ = writeln!(
                    out,
                    "    if{} class={:#04x} subclass={:#04x} protocol={:#04x}",
                    interface.number, interface.class, interface.subclass, interface.protocol
                );
            }
        }
    }
    out
}

#[must_use]
pub fn render_probe_report(report: &ProbeReport) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "device: {}", format_device_id(&report.device));
    let _ = writeln!(
        out,
        "configuration: value={} interfaces={}",
        report.configuration.configuration_value, report.configuration.num_interfaces
    );

    let _ = writeln!(out, "alternate settings:");
    for alt in &report.configuration.alt_settings {
        let _ = writeln!(
            out,
            "  if{} alt{} class={:#04x} subclass={:#04x} protocol={:#04x} endpoints={}",
            alt.number,
            alt.alternate_setting,
            alt.class,
            alt.subclass,
            alt.protocol,
            alt.num_endpoints
        );

        if let Some(master) = alt.cdc.union_master {
            let _ = writeln!(
                out,
                "    cdc union: master={master} slaves={:?}",
                alt.cdc.union_slaves
            );
        }
        if let Some(index) = alt.cdc.ethernet_mac_string_index {
            let mac = alt.cdc.mac_display();
            let _ = writeln!(
                out,
                "    cdc ethernet: mac_index={} mac={} max_segment_size={}",
                index,
                mac,
                optional_u16(alt.cdc.max_segment_size)
            );
        }
        if let Some(version) = alt.cdc.ncm_version_bcd {
            let capabilities = optional_hex_u8(alt.cdc.ncm_capabilities);
            let _ = writeln!(
                out,
                "    cdc ncm: version_bcd={version:#06x} capabilities={capabilities}"
            );
        }
        for endpoint in &alt.endpoints {
            let _ = writeln!(
                out,
                "    ep {:#04x} {:?} {:?} max_packet={} interval={}",
                endpoint.address,
                endpoint.direction,
                endpoint.transfer_type,
                endpoint.max_packet_size,
                endpoint.interval
            );
        }
    }

    if report.ncm_pairs.is_empty() {
        let _ = writeln!(out, "detected ncm pairs: none");
    } else {
        let _ = writeln!(out, "detected ncm pairs:");
        for pair in &report.ncm_pairs {
            let status = if pair.control_has_status_endpoint {
                "has-status-endpoint"
            } else {
                "missing-status-endpoint"
            };
            let _ = writeln!(
                out,
                "  control if{} alt{} -> data if{} alt{} in={} out={} mac={} {}",
                pair.control_interface,
                pair.control_alt_setting,
                pair.data_interface,
                pair.data_alt_setting,
                optional_endpoint(pair.bulk_in),
                optional_endpoint(pair.bulk_out),
                pair.mac.as_deref().unwrap_or("<unknown>"),
                status
            );
        }
    }

    if !report.claim_results.is_empty() {
        let _ = writeln!(out, "claim results:");
        for claim in &report.claim_results {
            match &claim.result {
                Ok(()) => {
                    let detach = if claim.detached_kernel_driver {
                        " detach"
                    } else {
                        ""
                    };
                    let _ = writeln!(out, "  if{}: ok{}", claim.interface_number, detach);
                }
                Err(err) => {
                    let _ = writeln!(out, "  if{}: error: {err}", claim.interface_number);
                }
            }
        }
    }

    out
}

impl CdcInterfaceInfo {
    fn mac_display(&self) -> &str {
        self.ethernet_mac.as_deref().unwrap_or("<unread>")
    }
}

fn summarize_device_info(device: &DeviceInfo) -> DeviceSummary {
    DeviceSummary {
        vendor_id: device.vendor_id(),
        product_id: device.product_id(),
        bus_id: device.bus_id().to_owned(),
        busnum: device.busnum(),
        address: device.device_address(),
        speed: device.speed().map(|speed| format!("{speed:?}")),
        manufacturer: device.manufacturer_string().map(ToOwned::to_owned),
        product: device.product_string().map(ToOwned::to_owned),
        serial: device.serial_number().map(ToOwned::to_owned),
        interfaces: device
            .interfaces()
            .map(|interface| InterfaceSummary {
                number: interface.interface_number(),
                class: interface.class(),
                subclass: interface.subclass(),
                protocol: interface.protocol(),
            })
            .collect(),
    }
}

fn find_single_device(selector: UsbSelector) -> eyre::Result<DeviceInfo> {
    let matches: Vec<_> = nusb::list_devices()
        .wait()
        .wrap_err("failed to list USB devices")?
        .filter(|device| selector.matches(device))
        .collect();

    match matches.len() {
        0 => Err(eyre::eyre!(
            "USB device {:#06x}:{:#06x} not found",
            selector.vendor_id,
            selector.product_id
        )),
        1 => matches
            .into_iter()
            .next()
            .ok_or_eyre("matched device disappeared"),
        count => Err(eyre::eyre!(
            "found {count} matching USB devices for {:#06x}:{:#06x}; selection by bus/address is not implemented yet",
            selector.vendor_id,
            selector.product_id
        )),
    }
}

fn summarize_configuration(
    device: &nusb::Device,
    configuration: &ConfigurationDescriptor<'_>,
) -> ConfigurationReport {
    let alt_settings = configuration
        .interface_alt_settings()
        .map(|alt| summarize_alt_setting(device, &alt))
        .collect();

    ConfigurationReport {
        configuration_value: configuration.configuration_value(),
        num_interfaces: configuration.num_interfaces(),
        alt_settings,
    }
}

fn summarize_alt_setting(
    device: &nusb::Device,
    alt: &InterfaceDescriptor<'_>,
) -> InterfaceAltSummary {
    let cdc = parse_cdc_interface_info(device, alt);
    let endpoints = alt
        .endpoints()
        .map(|endpoint| EndpointSummary {
            address: endpoint.address(),
            direction: if endpoint.address() & 0x80 == 0 {
                EndpointDirection::Out
            } else {
                EndpointDirection::In
            },
            transfer_type: map_transfer_type(endpoint.transfer_type()),
            max_packet_size: endpoint.max_packet_size(),
            interval: endpoint.interval(),
        })
        .collect();

    InterfaceAltSummary {
        number: alt.interface_number(),
        alternate_setting: alt.alternate_setting(),
        class: alt.class(),
        subclass: alt.subclass(),
        protocol: alt.protocol(),
        num_endpoints: alt.num_endpoints(),
        cdc,
        endpoints,
    }
}

fn parse_cdc_interface_info(
    device: &nusb::Device,
    alt: &InterfaceDescriptor<'_>,
) -> CdcInterfaceInfo {
    let mut cdc = CdcInterfaceInfo::default();

    for descriptor in alt.descriptors() {
        if descriptor.descriptor_type() != USB_DT_CS_INTERFACE || descriptor.len() < 3 {
            continue;
        }

        let Some(subtype) = descriptor.get(2).copied() else {
            continue;
        };

        match subtype {
            USB_CDC_UNION_TYPE if descriptor.len() >= 5 => {
                cdc.union_master = descriptor.get(3).copied();
                cdc.union_slaves = descriptor.get(4..).unwrap_or_default().to_vec();
            }
            USB_CDC_ETHERNET_TYPE if descriptor.len() >= 13 => {
                let Some(string_index) = descriptor.get(3).copied() else {
                    continue;
                };
                cdc.ethernet_mac_string_index = Some(string_index);
                cdc.max_segment_size = descriptor
                    .get(8..10)
                    .and_then(|bytes| bytes.try_into().ok())
                    .map(u16::from_le_bytes);
                cdc.ethernet_mac = read_string_descriptor(device, string_index);
            }
            USB_CDC_NCM_TYPE if descriptor.len() >= 6 => {
                cdc.ncm_version_bcd = descriptor
                    .get(3..5)
                    .and_then(|bytes| bytes.try_into().ok())
                    .map(u16::from_le_bytes);
                cdc.ncm_capabilities = descriptor.get(5).copied();
            }
            _ => {}
        }
    }

    cdc
}

fn read_string_descriptor(device: &nusb::Device, index: u8) -> Option<String> {
    let index = std::num::NonZeroU8::new(index)?;
    device
        .get_string_descriptor(index, US_ENGLISH, STRING_TIMEOUT)
        .wait()
        .ok()
}

fn detect_ncm_pairs(configuration: &ConfigurationReport) -> Vec<NcmPair> {
    let mut alt_by_key: BTreeMap<(u8, u8), &InterfaceAltSummary> = BTreeMap::new();
    for alt in &configuration.alt_settings {
        alt_by_key.insert((alt.number, alt.alternate_setting), alt);
    }

    let mut pairs = Vec::new();
    for control in configuration.alt_settings.iter().filter(|alt| {
        alt.class == USB_CLASS_COMM
            && alt.subclass == USB_CDC_SUBCLASS_NCM
            && alt.protocol == USB_CDC_PROTO_NONE
    }) {
        for slave in &control.cdc.union_slaves {
            if let Some(pair) = build_ncm_pair(control, *slave, &alt_by_key) {
                pairs.push(pair);
            }
        }
    }
    pairs
}

fn build_ncm_pair(
    control: &InterfaceAltSummary,
    data_interface: u8,
    alt_by_key: &BTreeMap<(u8, u8), &InterfaceAltSummary>,
) -> Option<NcmPair> {
    let data = alt_by_key
        .values()
        .copied()
        .filter(|alt| {
            alt.number == data_interface
                && alt.class == USB_CLASS_CDC_DATA
                && alt.protocol == USB_CDC_PROTO_NCM_DATA
        })
        .max_by_key(|alt| (alt.endpoints.len(), alt.alternate_setting))?;

    let bulk_in = data
        .endpoints
        .iter()
        .find(|endpoint| {
            endpoint.transfer_type == EndpointTransferType::Bulk
                && endpoint.direction == EndpointDirection::In
        })
        .map(|endpoint| endpoint.address);
    let bulk_out = data
        .endpoints
        .iter()
        .find(|endpoint| {
            endpoint.transfer_type == EndpointTransferType::Bulk
                && endpoint.direction == EndpointDirection::Out
        })
        .map(|endpoint| endpoint.address);
    let control_has_status_endpoint = control.endpoints.iter().any(|endpoint| {
        endpoint.transfer_type == EndpointTransferType::Interrupt
            && endpoint.direction == EndpointDirection::In
    });

    Some(NcmPair {
        control_interface: control.number,
        data_interface,
        control_alt_setting: control.alternate_setting,
        data_alt_setting: data.alternate_setting,
        bulk_in,
        bulk_out,
        mac_string_index: control.cdc.ethernet_mac_string_index,
        mac: control.cdc.ethernet_mac.clone(),
        control_has_status_endpoint,
    })
}

fn claim_ncm_interfaces(
    device: &nusb::Device,
    pairs: &[NcmPair],
    detach_kernel_driver: bool,
) -> Vec<ClaimResult> {
    let interfaces: BTreeSet<_> = pairs
        .iter()
        .flat_map(|pair| [pair.control_interface, pair.data_interface])
        .collect();
    let mut claimed = Vec::new();
    let mut results = Vec::new();

    for interface_number in interfaces {
        let result = if detach_kernel_driver {
            device.detach_and_claim_interface(interface_number).wait()
        } else {
            device.claim_interface(interface_number).wait()
        };

        match result {
            Ok(interface) => {
                claimed.push(interface);
                results.push(ClaimResult {
                    interface_number,
                    detached_kernel_driver: detach_kernel_driver,
                    result: Ok(()),
                });
            }
            Err(err) => results.push(ClaimResult {
                interface_number,
                detached_kernel_driver: detach_kernel_driver,
                result: Err(err.to_string()),
            }),
        }
    }

    drop(claimed);
    results
}

const fn map_transfer_type(transfer_type: TransferType) -> EndpointTransferType {
    match transfer_type {
        TransferType::Bulk => EndpointTransferType::Bulk,
        TransferType::Interrupt => EndpointTransferType::Interrupt,
        TransferType::Isochronous => EndpointTransferType::Isochronous,
        TransferType::Control => EndpointTransferType::Control,
    }
}

fn format_device_id(device: &DeviceSummary) -> String {
    let mut text = format!(
        "{:04x}:{:04x} bus={} busnum={} address={}",
        device.vendor_id, device.product_id, device.bus_id, device.busnum, device.address
    );
    if let Some(product) = &device.product {
        let _ = write!(text, " product={product:?}");
    }
    text
}

fn optional_endpoint(value: Option<u8>) -> String {
    value.map_or_else(
        || "<none>".to_owned(),
        |endpoint| format!("{endpoint:#04x}"),
    )
}

fn optional_u16(value: Option<u16>) -> String {
    value.map_or_else(|| "<none>".to_owned(), |value| value.to_string())
}

fn optional_hex_u8(value: Option<u8>) -> String {
    value.map_or_else(|| "<none>".to_owned(), |value| format!("{value:#04x}"))
}
