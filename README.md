# crc24-openpgp-fast

SIMD implementation of the CRC24 checksum as used in OpenPGP, for the x86-64 architecture.

## CPU requirements

x86-64 with the following extensions:
- pclmulqdq
- sse2
- sse4.1

NOTE: Will fallback to a non-SIMD implementation if CPU features are not present, so don't worry about using this library for code that potentially goes into non-compatible CPUs.

## Usage

`let res: u32 = crc24_openpgp_fast::hash_raw(&my_binary_slice);`

There is no "update"-like functionality yet, since doing this with arbitrarily lengths can be tricky with SIMD and destroy performance.
