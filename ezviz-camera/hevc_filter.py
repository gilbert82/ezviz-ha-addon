#!/usr/bin/env python3
"""
HEVC NAL unit filter for EZVIZ proprietary H.265 streams.

The EZVIZ HP2 (and possibly other models) use a proprietary H.265 variant
with non-standard NAL unit types that cause ffmpeg to fail.

This filter:
1. Strip any data before the first valid HEVC start code (proprietary preamble)
2. Strip NAL units with type > 40 (proprietary, e.g. type 48)
3. Replace entire VPS with a valid Main profile VPS (EZVIZ VPS is too
   non-standard to patch - bogus vps_id, layers, reserved fields, profile)
4. Patch SPS to reference VPS 0 and use Main profile
5. Cache and re-inject VPS/SPS/PPS before IDR frames
"""

import sys


# Standard HEVC NAL unit types (0-40)
MAX_STANDARD_NAL_TYPE = 40

# NAL types
NAL_VPS = 32
NAL_SPS = 33
NAL_PPS = 34

# IDR/keyframe NAL types that need VPS/SPS/PPS prepended
IDR_NAL_TYPES = {16, 17, 18, 19, 20, 21}  # BLA, IDR, CRA


def find_start_codes(data):
    """Find all HEVC start code positions in the data.
    Returns list of (position, start_code_length) tuples.
    """
    positions = []
    i = 0
    dlen = len(data)
    while i < dlen - 3:
        if data[i] == 0 and data[i + 1] == 0:
            if data[i + 2] == 1:
                positions.append((i, 3))
                i += 3
            elif data[i + 2] == 0 and i + 3 < dlen and data[i + 3] == 1:
                positions.append((i, 4))
                i += 4
            else:
                i += 1
        else:
            i += 1
    return positions


def get_nal_type(nal_header_byte):
    """Extract NAL unit type from first byte of NAL unit header.
    HEVC: forbidden(1) | nal_unit_type(6) | nuh_layer_id(6) | nuh_temporal_id_plus1(3)
    """
    return (nal_header_byte >> 1) & 0x3F


def ensure_4byte_sc(nal_with_sc, sc_len):
    """Ensure NAL unit uses 4-byte start code (0x00000001)."""
    if sc_len == 4:
        return nal_with_sc
    return bytearray(b'\x00') + nal_with_sc


# Pre-built valid VPS NAL for HEVC Main profile, Level 4.0
# MUST have max_sub_layers_minus1=3 to match the EZVIZ SPS
# (SPS has max_sub_layers=3, changing it would shift all VLC-coded fields)
_REPLACEMENT_VPS = bytearray(
    b'\x00\x00\x00\x01'  # 4-byte start code
    b'\x40\x01'          # NAL header: type=32 (VPS), layer_id=0, tid=1
    b'\x0c\x07'          # vps_id=0, reserved_3=3, max_layers=0, max_sub=3, nesting=1
    b'\xff\xff'          # reserved_0xffff
    b'\x01'              # PTL: profile_space=0, tier=0, profile_idc=1 (Main)
    b'\x60\x00\x00\x00' # profile_compat: Main + Main10 bits set
    b'\xb0'              # progressive=1, interlaced=0, non_packed=1, frame_only=1
    b'\x00\x00\x00\x00\x00'  # constraint flags (zeros)
    b'\x78'              # level_idc=120 (Level 4.0)
    b'\x00\x00'          # sub_layer flags: all 0 (no sub-layer PTL data)
    b'\x17\x02\x40'      # ordering_info + max_layer + trailing bits
)


def replace_vps(nal_data, sc_len):
    """Replace the entire VPS NAL with a valid minimal VPS.

    EZVIZ cameras send a heavily non-standard VPS with bogus values
    for nearly every field. Rather than patching individual fields
    (which can leave inconsistencies), we replace the entire VPS
    with a pre-built valid one for HEVC Main profile, Level 4.0.
    """
    return bytearray(_REPLACEMENT_VPS), True


def patch_sps(nal_data, sc_len):
    """Patch SPS to reference VPS 0 and use Main profile.

    SPS RBSP layout (after 2-byte NAL header):
      Byte 0: sps_video_parameter_set_id(4) | sps_max_sub_layers_minus1(3) | temporal_nesting(1)
      Byte 1: general_profile_space(2) | general_tier_flag(1) | general_profile_idc(5)
      ...

    We ONLY change:
    - sps_video_parameter_set_id to 0 (match our replacement VPS)
    - general_profile_idc to 1 (Main)
    We do NOT touch max_sub_layers_minus1 or anything else, because
    changing it would shift all VLC-coded fields that follow the PTL.
    """
    rbsp_start = sc_len + 2  # skip start code + 2-byte NAL header
    if len(nal_data) < rbsp_start + 2:
        return nal_data, False

    patched = False

    # Fix sps_video_parameter_set_id to 0 (bits [7:4] of RBSP byte 0)
    byte0 = nal_data[rbsp_start]
    sps_vps_id = (byte0 >> 4) & 0x0F
    if sps_vps_id != 0:
        nal_data[rbsp_start] = byte0 & 0x0F  # clear upper 4 bits
        patched = True

    # Fix general_profile_idc to Main (1) in PTL byte 0
    ptl_byte = nal_data[rbsp_start + 1]
    profile_idc = ptl_byte & 0x1F
    if profile_idc != 1:
        nal_data[rbsp_start + 1] = (ptl_byte & 0xE0) | 1
        patched = True

    return nal_data, patched


def filter_hevc_stream():
    """Read raw HEVC from stdin, filter NAL units, output to stdout."""
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    buffer = bytearray()
    total_in = 0
    total_out = 0
    total_filtered = 0
    total_preamble = 0
    chunk_count = 0

    # Cache parameter sets for re-injection before IDR frames
    last_vps = None
    last_sps = None
    last_pps = None
    injections = 0

    CHUNK_SIZE = 32768  # 32KB
    MAX_BUFFER = 2 * 1024 * 1024  # 2MB safety limit

    while True:
        try:
            chunk = stdin.read(CHUNK_SIZE)
            if not chunk:
                break

            buffer.extend(chunk)
            total_in += len(chunk)
            chunk_count += 1

            # Safety: prevent buffer from growing too large
            if len(buffer) > MAX_BUFFER:
                positions = find_start_codes(buffer)
                if positions:
                    buffer = buffer[positions[-1][0]:]
                else:
                    buffer = buffer[-4:]
                continue

            # Find first start code and strip any preamble before it
            positions = find_start_codes(buffer)
            if not positions:
                # No start codes found yet, keep buffering
                # But don't let pre-start-code data grow huge
                if len(buffer) > 65536:
                    buffer = buffer[-4:]
                continue

            first_pos = positions[0][0]
            if first_pos > 0:
                total_preamble += first_pos
                if chunk_count <= 20:
                    preview = buffer[:min(16, first_pos)].hex()
                    print(
                        f"Stripped {first_pos} preamble bytes (hex: {preview})",
                        file=sys.stderr,
                    )
                buffer = buffer[first_pos:]
                # Recalculate positions after stripping
                positions = find_start_codes(buffer)

            if len(positions) < 2:
                continue

            output = bytearray()

            for i in range(len(positions) - 1):
                pos, sc_len = positions[i]
                next_pos, _ = positions[i + 1]

                nal_with_sc = bytearray(buffer[pos:next_pos])

                if len(nal_with_sc) <= sc_len + 1:
                    continue

                first_byte = nal_with_sc[sc_len]

                # Check forbidden zero bit
                if first_byte & 0x80:
                    continue

                nal_type = get_nal_type(first_byte)

                # Check temporal_id_plus1 > 0
                temporal_id = nal_with_sc[sc_len + 1] & 0x07
                if temporal_id == 0:
                    continue

                # Filter proprietary NAL types (> 40)
                if nal_type > MAX_STANDARD_NAL_TYPE:
                    total_filtered += 1
                    if total_filtered <= 5 or total_filtered % 500 == 0:
                        print(
                            f"Filtered NAL type {nal_type} "
                            f"({len(nal_with_sc)} bytes) [total: {total_filtered}]",
                            file=sys.stderr,
                        )
                    continue

                # Normalize to 4-byte start code and patch VPS if needed
                normalized = ensure_4byte_sc(nal_with_sc, sc_len)

                if nal_type == NAL_VPS:
                    normalized, was_replaced = replace_vps(normalized, 4)
                    if was_replaced and chunk_count <= 10:
                        print(
                            f"Replaced VPS with standard Main profile "
                            f"({len(normalized)} bytes)",
                            file=sys.stderr,
                        )
                    last_vps = bytearray(normalized)
                    if chunk_count <= 5:
                        print(f"Cached VPS ({len(last_vps)} bytes)", file=sys.stderr)
                elif nal_type == NAL_SPS:
                    normalized, was_patched = patch_sps(normalized, 4)
                    if was_patched and chunk_count <= 10:
                        print(
                            f"Patched SPS vps_id=0, profile=Main "
                            f"(header: {normalized[6:8].hex()})",
                            file=sys.stderr,
                        )
                    last_sps = bytearray(normalized)
                    if chunk_count <= 5:
                        print(f"Cached SPS ({len(last_sps)} bytes)", file=sys.stderr)
                elif nal_type == NAL_PPS:
                    last_pps = bytearray(normalized)
                    if chunk_count <= 5:
                        print(f"Cached PPS ({len(last_pps)} bytes)", file=sys.stderr)

                # Before IDR frames, re-inject cached VPS/SPS/PPS
                if nal_type in IDR_NAL_TYPES:
                    if last_vps and last_sps and last_pps:
                        output.extend(last_vps)
                        output.extend(last_sps)
                        output.extend(last_pps)
                        injections += 1
                        if injections <= 3 or injections % 100 == 0:
                            print(
                                f"Injected VPS/SPS/PPS before IDR type {nal_type} "
                                f"[#{injections}]",
                                file=sys.stderr,
                            )

                output.extend(normalized)

            if output:
                stdout.write(bytes(output))
                stdout.flush()
                total_out += len(output)

            # Keep from last start code onwards
            if positions:
                last_pos = positions[-1][0]
                buffer = buffer[last_pos:]
            else:
                if len(buffer) > 4:
                    buffer = buffer[-4:]

            if chunk_count % 500 == 0:
                print(
                    f"HEVC filter stats: in={total_in // 1024}KB "
                    f"out={total_out // 1024}KB "
                    f"filtered={total_filtered} "
                    f"preamble={total_preamble}B "
                    f"injections={injections}",
                    file=sys.stderr,
                )

        except BrokenPipeError:
            break
        except Exception as e:
            print(f"HEVC filter error: {e}", file=sys.stderr)
            buffer = bytearray()

    # Flush remaining
    if buffer:
        try:
            stdout.write(bytes(buffer))
            stdout.flush()
        except:
            pass

    print(
        f"HEVC filter done: in={total_in // 1024}KB out={total_out // 1024}KB "
        f"filtered={total_filtered} preamble={total_preamble}B "
        f"injections={injections}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    filter_hevc_stream()
