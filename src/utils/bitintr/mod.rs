mod arch {
    #[cfg(target_arch = "x86")]
    pub use core::arch::x86::*;

    #[cfg(target_arch = "x86_64")]
    pub use core::arch::x86_64::*;

    #[cfg(target_arch = "arm")]
    pub use core::arch::arm::*;

    #[cfg(target_arch = "aarch64")]
    pub use core::arch::aarch64::*;
}

#[macro_use]
mod macros;

mod pdep;
mod pext;

pub use pdep::Pdep;
pub use pext::Pext;
