#!/usr/bin/env python3
"""
HEVC NAL unit filter for EZVIZ proprietary H.265 streams.

The EZVIZ HP2 (and possibly other models) use a proprietary H.265 variant
with non-standard NAL unit types, profiles, and parameter sets that cause
ffmpeg to fail with errors like:
  - "PPS id out of range"
  - "VPS/SPS 0 does not exist"
  - "Unknown HEVC profile: 23/6"

This filter:
1. Strips proprietary NAL units (types > 40)
2. Patches VPS/SPS profile to Main (1) so ffmpeg accepts the stream
3. Outputs a clean annexB stream to stdout
"""

import sys


# Standard HEVC NAL unit types (0-40)
MAX_STANDARD_NAL_TYPE = 40

# NAL types that contain profile info we need to patch
NAL_VPS = 32
NAL_SPS = 33

# Target profile: Main = 1 (standard, widely supported)
TARGET_PROFILE = 1


def find_start_codes(data):
    """Find all HEVC start code positions in the data.
    Returns list of (position, start_code_length) tuples.
    """
    positions = []
    i = 0
    while i < len(data) - 3:
        if data[i] == 0 and data[i + 1] == 0:
            if data[i + 2] == 1:
                positions.append((i, 3))
                i += 3
            elif data[i + 2] == 0 and i + 3 < len(data) and data[i + 3] == 1:
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
    """Patch the general_profile_idc in VPS or SPS NAL units.

    VPS structure after NAL header (2 bytes):
        - 2 bytes: vps_id(4) + flags(2) + max_layers(6) + max_sub_layers(3) + nesting(1)
        - 2 bytes: reserved 0xFFFF
        - profile_tier_level byte: profile_space(2) | tier_flag(1) | profile_idc(5)
        => Profile byte at offset: sc_len + 2 + 4 = sc_len + 6

    SPS structure after NAL header (2 bytes):
        - 1 byte: sps_vps_id(4) + max_sub_layers(3) + nesting(1)
        - profile_tier_level byte: profile_space(2) | tier_flag(1) | profile_idc(5)
        => Profile byte at offset: sc_len + 2 + 1 = sc_len + 3
    """
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
        return nal_data, False  # Already correct

    # Patch: keep upper 3 bits (profile_space + tier_flag), set profile to Main
    new_byte = (old_byte & 0xE0) | TARGET_PROFILE
    nal_data[profile_offset] = new_byte

    return nal_data, True


def filter_hevc_stream():
    """Read raw HEVC from stdin, filter and patch NAL units, output to stdout."""
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    buffer = bytearray()
    total_in = 0
    total_out = 0
    total_filtered = 0
    total_patched = 0
    chunk_count = 0

    CHUNK_SIZE = 32768  # 32KB

    while True:
        try:
            chunk = stdin.read(CHUNK_SIZE)
            if not chunk:
                break

            buffer.extend(chunk)
            total_in += len(chunk)
            chunk_count += 1

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

                nal_type = get_nal_type(nal_with_sc[sc_len])

                # Filter proprietary NAL types (> 40)
                if nal_type > MAX_STANDARD_NAL_TYPE:
                    total_filtered += 1
                    if total_filtered <= 10 or total_filtered % 100 == 0:
                        print(
                            f"Filtered proprietary NAL type {nal_type} "
                            f"({len(nal_with_sc)} bytes) [total: {total_filtered}]",
                            file=sys.stderr,
                        )
                    continue

                # Patch VPS/SPS profile if non-standard
                if nal_type in (NAL_VPS, NAL_SPS):
                    nal_with_sc, was_patched = patch_profile(nal_with_sc, sc_len, nal_type)
                    if was_patched:
                        total_patched += 1
                        nal_name = "VPS" if nal_type == NAL_VPS else "SPS"
                        if total_patched <= 5:
                            print(
                                f"Patched {nal_name} profile to Main (1) [total: {total_patched}]",
                                file=sys.stderr,
                            )

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
                    f"filtered={total_filtered} patched={total_patched}",
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
        f"filtered={total_filtered} patched={total_patched}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    filter_hevc_stream()
