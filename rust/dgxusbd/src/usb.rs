use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::mem::size_of;
use std::time::Duration;

use color_eyre::eyre::{self, Context as _, OptionExt as _};
use nusb::descriptors::{ConfigurationDescriptor, InterfaceDescriptor, TransferType, language_id};
use nusb::transfer::{Bulk, ControlIn, ControlOut, ControlType, In, Out, Recipient, TransferError};
use nusb::{DeviceInfo, MaybeFuture as _};
use zerocopy::byteorder::little_endian::{U16, U32};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use crate::ncm::{DEFAULT_NTB_MAX_SIZE, NtbBuildConfig, NtbParseConfig};

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
const CONTROL_TIMEOUT: Duration = Duration::from_secs(1);
const CDC_GET_NTB_PARAMETERS: u8 = 0x80;
const CDC_SET_NTB_FORMAT: u8 = 0x84;
const CDC_GET_MAX_DATAGRAM_SIZE: u8 = 0x87;
const CDC_SET_MAX_DATAGRAM_SIZE: u8 = 0x88;
const CDC_SET_CRC_MODE: u8 = 0x8a;
const CDC_SET_ETHERNET_PACKET_FILTER: u8 = 0x43;
const CDC_SET_NTB_INPUT_SIZE: u8 = 0x86;
const CDC_NCM_NTB16_FORMAT: u16 = 0x0000;
const CDC_NCM_NTB32_SUPPORTED: u16 = 0x0002;
const CDC_NCM_CRC_NOT_APPENDED: u16 = 0x0000;
const CDC_PACKET_TYPE_DIRECTED: u16 = 0x0001;
const CDC_PACKET_TYPE_ALL_MULTICAST: u16 = 0x0004;
const CDC_PACKET_TYPE_BROADCAST: u16 = 0x0008;
const CDC_NCM_NCAP_ETH_FILTER: u8 = 0x01;
const CDC_NCM_NCAP_MAX_DATAGRAM_SIZE: u8 = 0x08;
const CDC_NCM_NCAP_CRC_MODE: u8 = 0x10;
const CDC_NCM_NCAP_NTB_INPUT_SIZE: u8 = 0x20;

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
    pub max_segment_size: Option<u16>,
    pub ncm_capabilities: Option<u8>,
    pub control_has_status_endpoint: bool,
}

#[derive(Debug)]
pub struct ClaimResult {
    pub interface_number: u8,
    pub used_detach_and_claim: bool,
    pub result: Result<(), String>,
}

#[derive(Clone, Copy, Debug)]
pub struct OpenPairOptions {
    pub selector: UsbSelector,
    pub pair_index: usize,
    pub detach_kernel_driver: bool,
    pub initialize_ncm: bool,
    pub require_ncm_setup_success: bool,
    pub ntb_input_size: u32,
    pub max_datagram_size: u16,
}

#[derive(Debug)]
pub struct OpenNcmPair {
    pub pair: NcmPair,
    pub device_summary: DeviceSummary,
    pub control_interface: nusb::Interface,
    pub data_interface: nusb::Interface,
    pub setup_report: NcmSetupReport,
}

#[derive(Clone, Debug, Default)]
pub struct NcmSetupReport {
    pub ntb_parameters: Option<NtbParametersReport>,
    pub steps: Vec<ControlStep>,
}

#[derive(Clone, Debug)]
pub struct ControlStep {
    pub name: &'static str,
    pub result: Result<String, String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NtbParametersReport {
    pub formats_supported: u16,
    pub ntb_in_max_size: u32,
    pub ndp_in_divisor: u16,
    pub ndp_in_payload_remainder: u16,
    pub ndp_in_alignment: u16,
    pub ntb_out_max_size: u32,
    pub ndp_out_divisor: u16,
    pub ndp_out_payload_remainder: u16,
    pub ndp_out_alignment: u16,
    pub ntb_out_max_datagrams: u16,
}

#[derive(Debug)]
pub struct UsbSmokeOptions {
    pub open: OpenPairOptions,
    pub read_timeout: Duration,
}

#[derive(Debug)]
pub struct UsbSmokeReport {
    pub open: OpenNcmPair,
    pub bulk_in: u8,
    pub bulk_out: u8,
    pub bulk_in_max_packet_size: usize,
    pub bulk_out_max_packet_size: usize,
    pub read_result: Result<usize, String>,
}

#[derive(Clone, Copy, Debug, FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned)]
#[repr(C)]
struct NtbParametersRaw {
    w_length: U16,
    bm_ntb_formats_supported: U16,
    dw_ntb_in_max_size: U32,
    w_ndp_in_divisor: U16,
    w_ndp_in_payload_remainder: U16,
    w_ndp_in_alignment: U16,
    w_reserved: U16,
    dw_ntb_out_max_size: U32,
    w_ndp_out_divisor: U16,
    w_ndp_out_payload_remainder: U16,
    w_ndp_out_alignment: U16,
    w_ntb_out_max_datagrams: U16,
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

/// Open one detected CDC-NCM pair, run minimal class setup, and select the data altsetting.
///
/// # Errors
///
/// Returns an error if the USB device is absent, the selected pair is missing, interface claiming
/// fails, required NCM parameters cannot be read, or the data altsetting cannot be selected.
pub fn open_ncm_pair(options: OpenPairOptions) -> eyre::Result<OpenNcmPair> {
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
    let pairs = detect_ncm_pairs(&configuration_report);
    let pair = pairs.get(options.pair_index).cloned().ok_or_else(|| {
        eyre::eyre!(
            "pair index {} is missing; detected {} NCM pair(s)",
            options.pair_index,
            pairs.len()
        )
    })?;
    ensure_pair_has_bulk_endpoints(&pair)?;

    let control_interface = claim_interface(
        &device,
        pair.control_interface,
        options.detach_kernel_driver,
    )?;
    let data_interface =
        claim_interface(&device, pair.data_interface, options.detach_kernel_driver)?;

    let setup_report = if options.initialize_ncm {
        initialize_ncm_control(&control_interface, &pair, options)?
    } else {
        NcmSetupReport::default()
    };
    if options.require_ncm_setup_success {
        setup_report.ensure_success()?;
    }

    data_interface
        .set_alt_setting(pair.data_alt_setting)
        .wait()
        .wrap_err_with(|| {
            format!(
                "failed to set data interface {} altsetting {}",
                pair.data_interface, pair.data_alt_setting
            )
        })?;

    Ok(OpenNcmPair {
        pair,
        device_summary,
        control_interface,
        data_interface,
        setup_report,
    })
}

/// Claim the selected pair, select altsetting 1, open bulk endpoints, and try one timed read.
///
/// # Errors
///
/// Returns an error when pair setup or endpoint opening fails.
pub fn smoke_usb_data_path(options: UsbSmokeOptions) -> eyre::Result<UsbSmokeReport> {
    let open = open_ncm_pair(options.open)?;
    let bulk_in = open
        .pair
        .bulk_in
        .ok_or_eyre("selected pair has no bulk IN endpoint")?;
    let bulk_out = open
        .pair
        .bulk_out
        .ok_or_eyre("selected pair has no bulk OUT endpoint")?;
    let mut ep_in = open
        .data_interface
        .endpoint::<Bulk, In>(bulk_in)
        .wrap_err_with(|| format!("failed to open bulk IN endpoint {bulk_in:#04x}"))?;
    let ep_out = open
        .data_interface
        .endpoint::<Bulk, Out>(bulk_out)
        .wrap_err_with(|| format!("failed to open bulk OUT endpoint {bulk_out:#04x}"))?;

    let bulk_in_max_packet_size = ep_in.max_packet_size();
    let bulk_out_max_packet_size = ep_out.max_packet_size();
    let completion = ep_in.transfer_blocking(
        nusb::transfer::Buffer::new(DEFAULT_NTB_MAX_SIZE),
        options.read_timeout,
    );
    let read_result = completion
        .status
        .map(|()| completion.actual_len)
        .map_err(|err| err.to_string());

    Ok(UsbSmokeReport {
        open,
        bulk_in,
        bulk_out,
        bulk_in_max_packet_size,
        bulk_out_max_packet_size,
        read_result,
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
                    let detach = if claim.used_detach_and_claim {
                        " detach-and-claim"
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

#[must_use]
pub fn render_usb_smoke_report(report: &UsbSmokeReport) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "device: {}",
        format_device_id(&report.open.device_summary)
    );
    let _ = writeln!(
        out,
        "pair: control if{} -> data if{} alt{}",
        report.open.pair.control_interface,
        report.open.pair.data_interface,
        report.open.pair.data_alt_setting
    );
    let _ = writeln!(
        out,
        "bulk endpoints: in={:#04x} max_packet={} out={:#04x} max_packet={}",
        report.bulk_in,
        report.bulk_in_max_packet_size,
        report.bulk_out,
        report.bulk_out_max_packet_size
    );
    render_setup_report(&mut out, &report.open.setup_report);
    match &report.read_result {
        Ok(len) => {
            let _ = writeln!(out, "timed bulk read: ok {len} bytes");
        }
        Err(err) => {
            let _ = writeln!(out, "timed bulk read: {err}");
        }
    }
    out
}

pub fn render_setup_report(out: &mut String, report: &NcmSetupReport) {
    if let Some(parameters) = report.ntb_parameters {
        let _ = writeln!(
            out,
            "ntb parameters: formats={:#06x} in_max={} out_max={} out_datagrams={} payload_mod={} payload_remainder={} ndp_align={}",
            parameters.formats_supported,
            parameters.ntb_in_max_size,
            parameters.ntb_out_max_size,
            parameters.ntb_out_max_datagrams,
            parameters.ndp_out_divisor,
            parameters.ndp_out_payload_remainder,
            parameters.ndp_out_alignment
        );
    }
    if !report.steps.is_empty() {
        let _ = writeln!(out, "ncm setup:");
        for step in &report.steps {
            match &step.result {
                Ok(message) => {
                    let _ = writeln!(out, "  {}: ok {message}", step.name);
                }
                Err(err) => {
                    let _ = writeln!(out, "  {}: error {err}", step.name);
                }
            }
        }
    }
}

impl CdcInterfaceInfo {
    fn mac_display(&self) -> &str {
        self.ethernet_mac.as_deref().unwrap_or("<unread>")
    }
}

impl NcmSetupReport {
    fn ensure_success(&self) -> eyre::Result<()> {
        let failures: Vec<_> = self
            .steps
            .iter()
            .filter_map(|step| {
                step.result
                    .as_ref()
                    .err()
                    .map(|err| format!("{}: {err}", step.name))
            })
            .collect();

        if failures.is_empty() {
            Ok(())
        } else {
            Err(eyre::eyre!(
                "required CDC-NCM setup failed: {}",
                failures.join("; ")
            ))
        }
    }
}

impl NtbParametersRaw {
    fn into_report(self) -> NtbParametersReport {
        NtbParametersReport {
            formats_supported: self.bm_ntb_formats_supported.get(),
            ntb_in_max_size: self.dw_ntb_in_max_size.get(),
            ndp_in_divisor: self.w_ndp_in_divisor.get(),
            ndp_in_payload_remainder: self.w_ndp_in_payload_remainder.get(),
            ndp_in_alignment: self.w_ndp_in_alignment.get(),
            ntb_out_max_size: self.dw_ntb_out_max_size.get(),
            ndp_out_divisor: self.w_ndp_out_divisor.get(),
            ndp_out_payload_remainder: self.w_ndp_out_payload_remainder.get(),
            ndp_out_alignment: self.w_ndp_out_alignment.get(),
            ntb_out_max_datagrams: self.w_ntb_out_max_datagrams.get(),
        }
    }
}

impl NtbParametersReport {
    #[must_use]
    pub fn rx_parse_config(self) -> NtbParseConfig {
        NtbParseConfig {
            max_size: usize::try_from(self.ntb_in_max_size).unwrap_or(DEFAULT_NTB_MAX_SIZE),
            datagram_alignment: 1,
            ..NtbParseConfig::default()
        }
    }

    #[must_use]
    pub fn tx_build_config(self) -> NtbBuildConfig {
        let datagram_alignment = sanitize_alignment(self.ndp_out_divisor);
        NtbBuildConfig {
            max_size: usize::try_from(self.ntb_out_max_size).unwrap_or(DEFAULT_NTB_MAX_SIZE),
            datagram_alignment,
            datagram_remainder: adjusted_payload_remainder(
                self.ndp_out_payload_remainder,
                datagram_alignment,
            ),
        }
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
        .get_string_descriptor(index, language_id::US_ENGLISH, STRING_TIMEOUT)
        .wait()
        .ok()
}

fn ensure_pair_has_bulk_endpoints(pair: &NcmPair) -> eyre::Result<()> {
    if pair.bulk_in.is_none() {
        return Err(eyre::eyre!(
            "selected pair control if{} data if{} has no bulk IN endpoint",
            pair.control_interface,
            pair.data_interface
        ));
    }
    if pair.bulk_out.is_none() {
        return Err(eyre::eyre!(
            "selected pair control if{} data if{} has no bulk OUT endpoint",
            pair.control_interface,
            pair.data_interface
        ));
    }
    Ok(())
}

fn claim_interface(
    device: &nusb::Device,
    interface_number: u8,
    detach_kernel_driver: bool,
) -> eyre::Result<nusb::Interface> {
    let result = if detach_kernel_driver {
        device.detach_and_claim_interface(interface_number).wait()
    } else {
        device.claim_interface(interface_number).wait()
    };

    result.wrap_err_with(|| {
        let mode = if detach_kernel_driver {
            "detach-and-claim"
        } else {
            "claim"
        };
        format!("failed to {mode} interface {interface_number}")
    })
}

fn initialize_ncm_control(
    control_interface: &nusb::Interface,
    pair: &NcmPair,
    options: OpenPairOptions,
) -> eyre::Result<NcmSetupReport> {
    let mut report = NcmSetupReport::default();
    let parameters = read_ntb_parameters(control_interface)?;
    report.ntb_parameters = Some(parameters);

    if pair
        .ncm_capabilities
        .is_some_and(|capabilities| capabilities & CDC_NCM_NCAP_CRC_MODE != 0)
    {
        report.steps.push(control_out_value(
            control_interface,
            "set-crc-mode",
            CDC_SET_CRC_MODE,
            CDC_NCM_CRC_NOT_APPENDED,
            &[],
        ));
    }

    if parameters.formats_supported & CDC_NCM_NTB32_SUPPORTED != 0 {
        report.steps.push(control_out_value(
            control_interface,
            "set-ntb16-format",
            CDC_SET_NTB_FORMAT,
            CDC_NCM_NTB16_FORMAT,
            &[],
        ));
    }

    if pair
        .ncm_capabilities
        .is_some_and(|capabilities| capabilities & CDC_NCM_NCAP_NTB_INPUT_SIZE != 0)
    {
        let requested = options.ntb_input_size.min(parameters.ntb_in_max_size);
        report.steps.push(control_out_value(
            control_interface,
            "set-ntb-input-size",
            CDC_SET_NTB_INPUT_SIZE,
            0,
            &requested.to_le_bytes(),
        ));
    }

    if pair
        .ncm_capabilities
        .is_some_and(|capabilities| capabilities & CDC_NCM_NCAP_MAX_DATAGRAM_SIZE != 0)
    {
        let max_datagram_size = options
            .max_datagram_size
            .min(pair.max_segment_size.unwrap_or(options.max_datagram_size));
        report.steps.push(control_in_u16(
            control_interface,
            "get-max-datagram-size",
            CDC_GET_MAX_DATAGRAM_SIZE,
        ));
        report.steps.push(control_out_value(
            control_interface,
            "set-max-datagram-size",
            CDC_SET_MAX_DATAGRAM_SIZE,
            0,
            &max_datagram_size.to_le_bytes(),
        ));
    }

    if pair
        .ncm_capabilities
        .is_some_and(|capabilities| capabilities & CDC_NCM_NCAP_ETH_FILTER != 0)
    {
        report.steps.push(control_out_value(
            control_interface,
            "set-packet-filter",
            CDC_SET_ETHERNET_PACKET_FILTER,
            CDC_PACKET_TYPE_DIRECTED | CDC_PACKET_TYPE_ALL_MULTICAST | CDC_PACKET_TYPE_BROADCAST,
            &[],
        ));
    }

    Ok(report)
}

fn read_ntb_parameters(control_interface: &nusb::Interface) -> eyre::Result<NtbParametersReport> {
    let bytes = control_interface
        .control_in(
            ControlIn {
                control_type: ControlType::Class,
                recipient: Recipient::Interface,
                request: CDC_GET_NTB_PARAMETERS,
                value: 0,
                index: u16::from(control_interface.interface_number()),
                length: u16::try_from(size_of::<NtbParametersRaw>())
                    .expect("NtbParametersRaw length fits u16"),
            },
            CONTROL_TIMEOUT,
        )
        .wait()
        .map_err(control_error)
        .wrap_err("failed GET_NTB_PARAMETERS")?;
    let raw = NtbParametersRaw::read_from_bytes(bytes.as_slice())
        .map_err(|err| eyre::eyre!("GET_NTB_PARAMETERS returned malformed length: {err}"))?;
    Ok(raw.into_report())
}

fn control_out_value(
    control_interface: &nusb::Interface,
    name: &'static str,
    request: u8,
    value: u16,
    data: &[u8],
) -> ControlStep {
    let result = control_interface
        .control_out(
            ControlOut {
                control_type: ControlType::Class,
                recipient: Recipient::Interface,
                request,
                value,
                index: u16::from(control_interface.interface_number()),
                data,
            },
            CONTROL_TIMEOUT,
        )
        .wait()
        .map(|()| {
            if data.is_empty() {
                format!("value={value:#06x}")
            } else {
                format!("value={value:#06x} bytes={}", data.len())
            }
        })
        .map_err(|err| control_error(err).to_string());

    ControlStep { name, result }
}

fn control_in_u16(
    control_interface: &nusb::Interface,
    name: &'static str,
    request: u8,
) -> ControlStep {
    let result = control_interface
        .control_in(
            ControlIn {
                control_type: ControlType::Class,
                recipient: Recipient::Interface,
                request,
                value: 0,
                index: u16::from(control_interface.interface_number()),
                length: 2,
            },
            CONTROL_TIMEOUT,
        )
        .wait()
        .map_err(control_error)
        .and_then(|bytes| {
            let value = bytes
                .as_slice()
                .try_into()
                .map(u16::from_le_bytes)
                .map_err(|_| eyre::eyre!("expected 2 bytes, got {}", bytes.len()))?;
            Ok(format!("value={value}"))
        })
        .map_err(|err| err.to_string());

    ControlStep { name, result }
}

fn control_error(err: TransferError) -> eyre::Report {
    eyre::eyre!("{err}")
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
        max_segment_size: control.cdc.max_segment_size,
        ncm_capabilities: control.cdc.ncm_capabilities,
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
                    used_detach_and_claim: detach_kernel_driver,
                    result: Ok(()),
                });
            }
            Err(err) => results.push(ClaimResult {
                interface_number,
                used_detach_and_claim: detach_kernel_driver,
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

fn sanitize_alignment(value: u16) -> usize {
    let alignment = usize::from(value);
    if alignment >= 4 && alignment.is_power_of_two() {
        alignment
    } else {
        4
    }
}

fn adjusted_payload_remainder(remainder: u16, alignment: usize) -> usize {
    let alignment = alignment.max(1);
    let ethernet_header_remainder = crate::ncm::ETHERNET_HEADER_LEN % alignment;
    (usize::from(remainder) + alignment - ethernet_header_remainder) % alignment
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
