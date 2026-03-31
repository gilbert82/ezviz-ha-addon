#!/usr/bin/env python3
"""
HEVC NAL unit filter for EZVIZ proprietary H.265 streams.

The EZVIZ HP2 (and possibly other models) use a proprietary H.265 variant
with non-standard NAL unit types, profiles, and parameter sets that cause
ffmpeg to fail with errors like:
  - "PPS id out of range"
  - "VPS/SPS 0 does not exist"
  - "Unknown HEVC profile: 23/6"
  - "Invalid NAL unit 0, skipping"

EZVIZ VTDU packets on channel 0x01 contain raw video data, but each packet
may include proprietary preamble bytes before the actual HEVC annexB data.
Some packets start with 00 00 01 FC which indicates an extended proprietary
header. This filter handles all of that.

This filter:
1. Scans for valid HEVC start codes and strips any proprietary preamble
2. Strips proprietary NAL units (types > 40, including type 48 commonly seen)
3. Patches VPS/SPS profile to Main (1) so ffmpeg accepts the stream
4. Injects VPS/SPS/PPS before IDR frames if they were received separately
5. Outputs a clean annexB stream to stdout
"""

import sys


# Standard HEVC NAL unit types (0-40)
MAX_STANDARD_NAL_TYPE = 40

# NAL types we care about
NAL_VPS = 32
NAL_SPS = 33
NAL_PPS = 34
NAL_AUD = 35
NAL_PREFIX_SEI = 39
NAL_SUFFIX_SEI = 40

# IDR/keyframe NAL types that need VPS/SPS/PPS prepended
IDR_NAL_TYPES = {16, 17, 18, 19, 20, 21}  # BLA, IDR, CRA

# Target profile: Main = 1 (standard, widely supported)
TARGET_PROFILE = 1


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
    HEVC NAL header: forbidden(1) | nal_unit_type(6) | nuh_layer_id(6) | nuh_temporal_id_plus1(3)
    """
    return (nal_header_byte >> 1) & 0x3F


def patch_profile(nal_data, sc_len, nal_type):
    """Patch the general_profile_idc in VPS or SPS NAL units."""
    if nal_type == NAL_VPS:
        profile_offset = sc_len + 6
    elif nal_type == NAL_SPS:
        profile_offset = sc_len + 3
    else:
        return nal_data, False

    if profile_offset >= len(nal_data):
        return nal_data, False

    old_byte = nal_data[profile_offset]
    old_profile = old_byte & 0x1F

    if old_profile == TARGET_PROFILE:
        return nal_data, False

    new_byte = (old_byte & 0xE0) | TARGET_PROFILE
    nal_data[profile_offset] = new_byte

    return nal_data, True


def ensure_4byte_sc(nal_with_sc, sc_len):
    """Ensure NAL unit uses 4-byte start code (0x00000001)."""
    if sc_len == 4:
        return nal_with_sc
    return bytearray(b'\x00') + nal_with_sc


def filter_hevc_stream():
    """Read raw HEVC from stdin, filter and patch NAL units, output to stdout."""
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    buffer = bytearray()
    total_in = 0
    total_out = 0
    total_filtered = 0
    total_patched = 0
    total_preamble_stripped = 0
    chunk_count = 0

    # Store last seen parameter sets so we can re-inject before IDR
    last_vps = None
    last_sps = None
    last_pps = None
    param_sets_injected = 0

    CHUNK_SIZE = 32768  # 32KB
    # Max buffer size to prevent unbounded growth (2MB)
    MAX_BUFFER = 2 * 1024 * 1024

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
                # Find last start code and keep from there
                positions = find_start_codes(buffer)
                if positions:
                    buffer = buffer[positions[-1][0]:]
                else:
                    buffer = buffer[-4:]
                continue

            # Strip any data before the first start code
            # This handles EZVIZ proprietary preamble bytes
            first_sc = -1
            for i in range(len(buffer) - 3):
                if buffer[i] == 0 and buffer[i+1] == 0:
                    if buffer[i+2] == 1 or (buffer[i+2] == 0 and i+3 < len(buffer) and buffer[i+3] == 1):
                        first_sc = i
                        break
            if first_sc > 0:
                total_preamble_stripped += first_sc
                if chunk_count <= 10:
                    print(
                        f"Stripped {first_sc} preamble bytes "
                        f"(first bytes: {buffer[:min(8,first_sc)].hex()})",
                        file=sys.stderr,
                    )
                buffer = buffer[first_sc:]

            positions = find_start_codes(buffer)

            if len(positions) < 2:
                continue

            output = bytearray()

            for i in range(len(positions) - 1):
                pos, sc_len = positions[i]
                next_pos, _ = positions[i + 1]

                nal_with_sc = bytearray(buffer[pos:next_pos])

                if len(nal_with_sc) <= sc_len:
                    continue

                first_byte = nal_with_sc[sc_len]

                # Check forbidden zero bit - must be 0 for valid HEVC
                if first_byte & 0x80:
                    continue

                nal_type = get_nal_type(first_byte)

                # Check temporal_id_plus1 (must be > 0)
                if len(nal_with_sc) > sc_len + 1:
                    temporal_id = nal_with_sc[sc_len + 1] & 0x07
                    if temporal_id == 0:
                        continue

                # Filter proprietary NAL types (> 40)
                if nal_type > MAX_STANDARD_NAL_TYPE:
                    total_filtered += 1
                    if total_filtered <= 5 or total_filtered % 500 == 0:
                        print(
                            f"Filtered proprietary NAL type {nal_type} "
                            f"({len(nal_with_sc)} bytes) [total: {total_filtered}]",
                            file=sys.stderr,
                        )
                    continue

                # Store parameter sets
                if nal_type == NAL_VPS:
                    nal_with_sc, was_patched = patch_profile(nal_with_sc, sc_len, nal_type)
                    if was_patched:
                        total_patched += 1
                    last_vps = ensure_4byte_sc(bytearray(nal_with_sc), sc_len)
                    if chunk_count <= 3:
                        print(f"Captured VPS ({len(last_vps)} bytes)", file=sys.stderr)

                elif nal_type == NAL_SPS:
                    nal_with_sc, was_patched = patch_profile(nal_with_sc, sc_len, nal_type)
                    if was_patched:
                        total_patched += 1
                    last_sps = ensure_4byte_sc(bytearray(nal_with_sc), sc_len)
                    if chunk_count <= 3:
                        print(f"Captured SPS ({len(last_sps)} bytes)", file=sys.stderr)

                elif nal_type == NAL_PPS:
                    last_pps = ensure_4byte_sc(bytearray(nal_with_sc), sc_len)
                    if chunk_count <= 3:
                        print(f"Captured PPS ({len(last_pps)} bytes)", file=sys.stderr)

                # Before IDR frames, inject VPS/SPS/PPS if we have them
                # This ensures FFmpeg always has parameter sets when decoding keyframes
                if nal_type in IDR_NAL_TYPES and last_vps and last_sps and last_pps:
                    if nal_type not in (NAL_VPS, NAL_SPS, NAL_PPS):
                        output.extend(last_vps)
                        output.extend(last_sps)
                        output.extend(last_pps)
                        param_sets_injected += 1
                        if param_sets_injected <= 3:
                            print(
                                f"Injected VPS/SPS/PPS before IDR (NAL type {nal_type}) "
                                f"[total injections: {param_sets_injected}]",
                                file=sys.stderr,
                            )

                # Normalize start code and add to output
                nal_with_sc = ensure_4byte_sc(nal_with_sc, sc_len)
                output.extend(nal_with_sc)

            if output:
                stdout.write(bytes(output))
                stdout.flush()
                total_out += len(output)

            # Keep unprocessed data (from last start code onwards)
            if positions:
                last_pos = positions[-1][0]
                buffer = buffer[last_pos:]
            else:
                if len(buffer) > 4:
                    buffer = buffer[-4:]

            if chunk_count % 500 == 0:
                print(
                    f"HEVC filter: in={total_in // 1024}KB out={total_out // 1024}KB "
                    f"filtered={total_filtered} patched={total_patched} "
                    f"preamble={total_preamble_stripped}B "
                    f"injected={param_sets_injected}",
                    file=sys.stderr,
                )

        except BrokenPipeError:
            break
        except Exception as e:
            print(f"HEVC filter error: {e}", file=sys.stderr)
            try:
                stdout.write(bytes(buffer))
                stdout.flush()
            except:
                pass
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
        f"filtered={total_filtered} patched={total_patched} "
        f"preamble={total_preamble_stripped}B injected={param_sets_injected}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    filter_hevc_stream()
