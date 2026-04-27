#[cfg(not(target_os = "linux"))]
compile_error!("dgxusbd is linux-only");

pub fn foo() -> i32 {
    return 42;
}
