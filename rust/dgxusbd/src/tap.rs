use color_eyre::eyre::{self, Context as _};
use tun_rs::{DeviceBuilder, Layer, SyncDevice};

pub const DEFAULT_TAP_NAME: &str = "dgxusb0";
pub const DEFAULT_TAP_MTU: u16 = 1500;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TapOptions {
    pub name: String,
    pub mtu: u16,
    pub nonblocking: bool,
    pub multi_queue: bool,
}

impl Default for TapOptions {
    fn default() -> Self {
        Self {
            name: DEFAULT_TAP_NAME.to_owned(),
            mtu: DEFAULT_TAP_MTU,
            nonblocking: false,
            multi_queue: false,
        }
    }
}

pub struct TapDevice {
    pub name: String,
    pub mtu: u16,
    pub device: SyncDevice,
}

impl std::fmt::Debug for TapDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TapDevice")
            .field("name", &self.name)
            .field("mtu", &self.mtu)
            .finish_non_exhaustive()
    }
}

pub fn create_tap(options: &TapOptions) -> eyre::Result<TapDevice> {
    let builder = DeviceBuilder::new()
        .name(options.name.clone())
        .layer(Layer::L2)
        .mtu(options.mtu)
        .packet_information(false);
    #[cfg(target_os = "linux")]
    let builder = builder.multi_queue(options.multi_queue);
    let device = builder
        .build_sync()
        .wrap_err_with(|| format!("failed to create TAP interface {}", options.name))?;
    if options.nonblocking {
        device
            .set_nonblocking(true)
            .wrap_err("failed to put TAP device in nonblocking mode")?;
    }
    let name = device.name().unwrap_or_else(|_| options.name.clone());

    Ok(TapDevice {
        name,
        mtu: options.mtu,
        device,
    })
}

#[must_use]
pub fn render_tap_smoke(device: &TapDevice) -> String {
    format!("tap: name={} mtu={}\n", device.name, device.mtu)
}
