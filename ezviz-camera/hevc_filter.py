#!/usr/bin/env python3
"""
HEVC NAL unit filter for EZVIZ proprietary H.265 streams.

The EZVIZ HP2 (and possibly other models) use a proprietary H.265 variant
with non-standard NAL unit types, profiles, and parameter sets that cause
ffmpeg to fail with errors like:
  - "PPS id out of range"
  - "VPS/SPS 0 does not exist"
  - "Unknown HEVC profile: 23/6"

This filter reads raw H.265 annexB bitstream from stdin, strips out
proprietary/invalid NAL units, and outputs a clean stream to stdout
that ffmpeg can decode.
"""

import sys
import struct

# Standard HEVC NAL unit types we want to keep
VALID_NAL_TYPES = {
    0: "TRAIL_N",
    1: "TRAIL_R",
    2: "TSA_N",
    3: "TSA_R",
    4: "STSA_N",
    5: "STSA_R",
    6: "RADL_N",
    7: "RADL_R",
    8: "RASL_N",
    9: "RASL_R",
    16: "BLA_W_LP",
    17: "BLA_W_RADL",
    18: "BLA_N_LP",
    19: "IDR_W_RADL",
    20: "IDR_N_LP",
    21: "CRA_NUT",
    32: "VPS",
    33: "SPS",
    34: "PPS",
    35: "AUD",
    36: "EOS",
    37: "EOB",
    38: "FD",
    39: "PREFIX_SEI",
    40: "SUFFIX_SEI",
}

MAX_STANDARD_NAL_TYPE = 40


def find_start_codes(data):
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
    return (nal_header_byte >> 1) & 0x3F


def filter_hevc_stream():
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    buffer = bytearray()
    total_in = 0
    total_out = 0
    total_filtered = 0
    chunk_count = 0
    CHUNK_SIZE = 32768

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
                nal_with_sc = buffer[pos:next_pos]
                if len(nal_with_sc) <= sc_len:
                    continue
                nal_type = get_nal_type(nal_with_sc[sc_len])
                if nal_type <= MAX_STANDARD_NAL_TYPE:
                    output.extend(nal_with_sc)
                else:
                    total_filtered += 1
                    if total_filtered <= 10 or total_filtered % 100 == 0:
                        print(f"Filtered proprietary NAL type {nal_type} ({len(nal_with_sc)} bytes) [total: {total_filtered}]", file=sys.stderr)
            if output:
                stdout.write(bytes(output))
                stdout.flush()
                total_out += len(output)
            if positions:
                buffer = buffer[positions[-1][0]:]
            elif len(buffer) > 4:
                buffer = buffer[-4:]
            if chunk_count % 500 == 0:
                print(f"HEVC filter: in={total_in // 1024}KB out={total_out // 1024}KB filtered={total_filtered}", file=sys.stderr)
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
    if buffer:
        try:
            stdout.write(bytes(buffer))
            stdout.flush()
        except:
            pass
    print(f"HEVC filter done: in={total_in // 1024}KB out={total_out // 1024}KB filtered={total_filtered}", file=sys.stderr)


if __name__ == "__main__":
    filter_hevc_stream()
