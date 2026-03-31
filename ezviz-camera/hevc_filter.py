#!/usr/bin/env python3
"""
HEVC NAL unit filter for EZVIZ proprietary H.265 streams.

The EZVIZ HP2 (and possibly other models) use a proprietary H.265 variant
with non-standard NAL unit types and heavily non-standard VPS/SPS/PPS
parameter sets that FFmpeg cannot decode.

Previous approaches (v2.4.0-v2.4.6) tried patching individual fields in
VPS/SPS/PPS headers.  This failed because the entire internal structure
of the EZVIZ SPS (profile-tier-level sub-layers, VLC-coded fields) is
non-standard, causing cascading decode failures.

NEW APPROACH (v3.0):
1. Strip proprietary preamble and NAL types > 40
2. Parse the ORIGINAL SPS just enough to extract resolution, chroma format,
   and bit depth (these are the fields that MUST match the slice data)
3. Generate COMPLETE replacement VPS, SPS, and PPS that are fully
   standard-compliant but use the correct resolution/format from step 2
4. Replace all VPS/SPS/PPS NAL units with the generated replacements
5. Cache and re-inject VPS/SPS/PPS before IDR frames
"""

import sys
import struct


# Standard HEVC NAL unit types (0-40)
MAX_STANDARD_NAL_TYPE = 40

# NAL types
NAL_VPS = 32
NAL_SPS = 33
NAL_PPS = 34

# IDR/keyframe NAL types that need VPS/SPS/PPS prepended
IDR_NAL_TYPES = {16, 17, 18, 19, 20, 21}  # BLA, IDR, CRA


# ============================================================================
# Bit-level reader
# ============================================================================

class BitReader:
    """Read individual bits from a byte buffer."""
    __slots__ = ('data', 'pos', 'size')

    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.size = len(data) * 8

    def read(self, n):
        """Read n bits and return as integer."""
        val = 0
        for _ in range(n):
            if self.pos >= self.size:
                return val
            byte_idx = self.pos >> 3
            bit_idx = 7 - (self.pos & 7)
            val = (val << 1) | ((self.data[byte_idx] >> bit_idx) & 1)
            self.pos += 1
        return val

    def read_ue(self):
        """Read unsigned Exp-Golomb coded value."""
        leading = 0
        while self.pos < self.size and self.read(1) == 0:
            leading += 1
            if leading > 31:
                return 0
        return (1 << leading) - 1 + self.read(leading)

    def skip(self, n):
        """Skip n bits."""
        self.pos += n


# ============================================================================
# Bit-level writer
# ============================================================================

class BitWriter:
    """Write individual bits to a byte buffer."""
    __slots__ = ('buf', 'pos')

    def __init__(self):
        self.buf = bytearray()
        self.pos = 0  # bit position in current byte

    def write(self, val, n):
        """Write n bits from integer val."""
        for i in range(n - 1, -1, -1):
            bit = (val >> i) & 1
            byte_idx = len(self.buf) - 1 if self.pos > 0 else -1
            if self.pos == 0:
                self.buf.append(0)
                byte_idx = len(self.buf) - 1
            self.buf[byte_idx] |= bit << (7 - self.pos)
            self.pos = (self.pos + 1) & 7

    def write_ue(self, val):
        """Write unsigned Exp-Golomb coded value."""
        val1 = val + 1
        nbits = val1.bit_length()
        # Write (nbits-1) zero bits, then nbits bits of val1
        self.write(0, nbits - 1)
        self.write(val1, nbits)

    def write_se(self, val):
        """Write signed Exp-Golomb coded value."""
        if val > 0:
            self.write_ue(2 * val - 1)
        elif val < 0:
            self.write_ue(-2 * val)
        else:
            self.write_ue(0)

    def write_trailing_bits(self):
        """Write RBSP trailing bits (1 followed by zeros to byte boundary)."""
        self.write(1, 1)
        if self.pos != 0:
            self.write(0, 8 - self.pos)

    def add_emulation_prevention(self):
        """Add emulation prevention bytes to RBSP data."""
        result = bytearray()
        i = 0
        while i < len(self.buf):
            if (i + 2 < len(self.buf) and
                    self.buf[i] == 0 and self.buf[i + 1] == 0 and
                    self.buf[i + 2] in (0, 1, 2, 3)):
                result.append(0)
                result.append(0)
                result.append(3)  # emulation prevention byte
                i += 2
            else:
                result.append(self.buf[i])
                i += 1
        return bytes(result)

    def get_bytes(self):
        """Get the raw RBSP bytes (without emulation prevention)."""
        return bytes(self.buf)


# ============================================================================
# Remove emulation prevention for reading
# ============================================================================

def remove_ep3(data):
    """Remove emulation prevention bytes (0x00 0x00 0x03 -> 0x00 0x00)."""
    result = bytearray()
    i = 0
    while i < len(data):
        if (i + 2 < len(data) and
                data[i] == 0 and data[i + 1] == 0 and data[i + 2] == 3):
            result.append(0)
            result.append(0)
            i += 3
        else:
            result.append(data[i])
            i += 1
    return bytes(result)


# ============================================================================
# Parse original SPS to extract essential parameters
# ============================================================================

def parse_sps_params(nal_data, sc_len):
    """
    Parse EZVIZ SPS NAL to extract resolution, chroma format, and bit depth.

    We parse field-by-field, tolerating non-standard values in the
    profile-tier-level section, because we only need the fields that
    come AFTER the PTL: resolution, chroma format, bit depth.

    Returns dict with: width, height, chroma_format_idc, bit_depth_luma,
                       bit_depth_chroma, max_sub_layers_minus1
    Returns None if parsing fails.
    """
    rbsp_start = sc_len + 2  # skip start code + NAL header
    if len(nal_data) < rbsp_start + 4:
        return None

    rbsp = remove_ep3(nal_data[rbsp_start:])
    reader = BitReader(rbsp)

    try:
        # sps_video_parameter_set_id: u(4)
        vps_id = reader.read(4)

        # sps_max_sub_layers_minus1: u(3)
        max_sub = reader.read(3)

        # sps_temporal_id_nesting_flag: u(1)
        nesting = reader.read(1)

        # profile_tier_level(1, max_sub_layers_minus1)
        # General PTL: 2+1+5 + 32 + 1+1+1+1 + 44 + 8 = 96 bits
        reader.skip(2 + 1 + 5)       # profile_space, tier, profile_idc
        reader.skip(32)               # profile_compatibility_flags
        reader.skip(1 + 1 + 1 + 1)   # progressive, interlaced, non_packed, frame_only
        reader.skip(44)               # constraint flags
        reader.skip(8)                # general_level_idc

        # Sub-layer present flags
        sub_profile_present = []
        sub_level_present = []
        for i in range(max_sub):
            sub_profile_present.append(reader.read(1))
            sub_level_present.append(reader.read(1))

        # Padding to 8 sub-layers
        if max_sub > 0:
            for i in range(max_sub, 8):
                reader.skip(2)

        # Sub-layer PTL data
        for i in range(max_sub):
            if sub_profile_present[i]:
                reader.skip(2 + 1 + 5 + 32 + 1 + 1 + 1 + 1 + 44)
            if sub_level_present[i]:
                reader.skip(8)

        # NOW we're past the PTL section
        # sps_seq_parameter_set_id: ue(v)
        sps_id = reader.read_ue()

        # chroma_format_idc: ue(v)
        chroma = reader.read_ue()

        if chroma == 3:
            reader.read(1)  # separate_colour_plane_flag

        # pic_width_in_luma_samples: ue(v)
        width = reader.read_ue()
        # pic_height_in_luma_samples: ue(v)
        height = reader.read_ue()

        # conformance_window_flag
        conf_win = reader.read(1)
        conf_left = conf_right = conf_top = conf_bottom = 0
        if conf_win:
            conf_left = reader.read_ue()
            conf_right = reader.read_ue()
            conf_top = reader.read_ue()
            conf_bottom = reader.read_ue()

        # bit_depth_luma_minus8: ue(v)
        bd_luma = reader.read_ue()
        # bit_depth_chroma_minus8: ue(v)
        bd_chroma = reader.read_ue()

        # log2_max_pic_order_cnt_lsb_minus4: ue(v)
        log2_poc = reader.read_ue()

        # Sanity checks
        if width < 16 or width > 8192 or height < 16 or height > 8192:
            return None
        if chroma > 3 or bd_luma > 8 or bd_chroma > 8:
            return None
        if log2_poc > 12:
            return None

        params = {
            'width': width,
            'height': height,
            'chroma_format_idc': chroma,
            'bit_depth_luma': bd_luma + 8,
            'bit_depth_chroma': bd_chroma + 8,
            'max_sub_layers_minus1': max_sub,
            'log2_max_poc_lsb_minus4': log2_poc,
            'sps_id': sps_id,
            'conf_win': conf_win,
            'conf_left': conf_left,
            'conf_right': conf_right,
            'conf_top': conf_top,
            'conf_bottom': conf_bottom,
        }

        return params

    except Exception:
        return None


# ============================================================================
# Generate standard-compliant VPS, SPS, PPS
# ============================================================================

def build_vps():
    """
    Build a minimal, fully standard-compliant VPS NAL unit.

    VPS:
    - vps_video_parameter_set_id = 0
    - vps_max_layers_minus1 = 0
    - vps_max_sub_layers_minus1 = 0  (simplest possible)
    - vps_temporal_id_nesting_flag = 1
    - PTL: Main profile, Level 4.0
    - vps_sub_layer_ordering_info_present_flag = 1
    - max_dec_pic_buffering[0] = 5
    - max_num_reorder_pics[0] = 2
    - max_latency_increase[0] = 0
    """
    w = BitWriter()

    # vps_video_parameter_set_id: u(4)
    w.write(0, 4)
    # vps_base_layer_internal_flag: u(1)
    w.write(1, 1)
    # vps_base_layer_available_flag: u(1)
    w.write(1, 1)
    # vps_max_layers_minus1: u(6)
    w.write(0, 6)
    # vps_max_sub_layers_minus1: u(3) = 0
    w.write(0, 3)
    # vps_temporal_id_nesting_flag: u(1)
    w.write(1, 1)
    # vps_reserved_0xffff_16bits: u(16)
    w.write(0xFFFF, 16)

    # profile_tier_level(1, vps_max_sub_layers_minus1=0)
    # general_profile_space: u(2)
    w.write(0, 2)
    # general_tier_flag: u(1) = 0 (Main tier)
    w.write(0, 1)
    # general_profile_idc: u(5) = 1 (Main)
    w.write(1, 5)
    # general_profile_compatibility_flags: u(32) - bit 1 set for Main
    w.write(0x60000000, 32)
    # general_progressive_source_flag
    w.write(1, 1)
    # general_interlaced_source_flag
    w.write(0, 1)
    # general_non_packed_constraint_flag
    w.write(1, 1)
    # general_frame_only_constraint_flag
    w.write(1, 1)
    # 44 constraint bits = 0
    w.write(0, 44)
    # general_level_idc: u(8) = 120 (Level 4.0)
    w.write(120, 8)
    # No sub-layer flags (max_sub=0)

    # vps_sub_layer_ordering_info_present_flag
    w.write(1, 1)
    # For i=0 only (since max_sub=0):
    w.write_ue(4)  # max_dec_pic_buffering_minus1 = 4
    w.write_ue(2)  # max_num_reorder_pics = 2
    w.write_ue(0)  # max_latency_increase_plus1 = 0

    # vps_max_layer_id: u(6) = 0
    w.write(0, 6)
    # vps_num_layer_sets_minus1: ue(v) = 0
    w.write_ue(0)
    # vps_timing_info_present_flag
    w.write(0, 1)
    # vps_extension_flag
    w.write(0, 1)

    w.write_trailing_bits()

    rbsp = w.add_emulation_prevention()
    # Build complete NAL: start_code + nal_header + rbsp
    nal = bytearray(b'\x00\x00\x00\x01')  # 4-byte start code
    nal.append(0x40)  # NAL type 32 (VPS): (32 << 1) = 0x40
    nal.append(0x01)  # nuh_layer_id=0, nuh_temporal_id_plus1=1
    nal.extend(rbsp)
    return bytes(nal)


def build_sps(params):
    """
    Build a standard-compliant SPS NAL unit using parameters from the
    original EZVIZ SPS (width, height, chroma, bit depth).

    Key design decisions:
    - max_sub_layers_minus1 = 0 (simplest possible, matches our VPS)
    - Main profile, Level 4.0
    - No scaling lists, no PCM, no long-term ref pics
    - Minimal short-term RPS: one set with zero entries
    """
    w = BitWriter()

    width = params['width']
    height = params['height']
    chroma = params.get('chroma_format_idc', 1)
    bd_luma_m8 = params.get('bit_depth_luma', 8) - 8
    bd_chroma_m8 = params.get('bit_depth_chroma', 8) - 8
    log2_poc = params.get('log2_max_poc_lsb_minus4', 4)
    conf_win = params.get('conf_win', 0)

    # sps_video_parameter_set_id: u(4) = 0
    w.write(0, 4)
    # sps_max_sub_layers_minus1: u(3) = 0
    w.write(0, 3)
    # sps_temporal_id_nesting_flag: u(1) = 1
    w.write(1, 1)

    # profile_tier_level(1, max_sub=0)
    w.write(0, 2)   # profile_space
    w.write(0, 1)   # tier_flag (Main tier)
    w.write(1, 5)   # profile_idc = 1 (Main)
    w.write(0x60000000, 32)  # profile_compat: Main + Main10
    w.write(1, 1)   # progressive
    w.write(0, 1)   # interlaced
    w.write(1, 1)   # non_packed
    w.write(1, 1)   # frame_only
    w.write(0, 44)  # constraint flags
    w.write(120, 8) # level_idc = 120 (4.0)
    # No sub-layer data (max_sub=0)

    # sps_seq_parameter_set_id: ue(v) = 0
    w.write_ue(0)

    # chroma_format_idc
    w.write_ue(chroma)
    if chroma == 3:
        w.write(0, 1)  # separate_colour_plane_flag

    # pic_width_in_luma_samples
    w.write_ue(width)
    # pic_height_in_luma_samples
    w.write_ue(height)

    # conformance_window_flag
    w.write(conf_win, 1)
    if conf_win:
        w.write_ue(params.get('conf_left', 0))
        w.write_ue(params.get('conf_right', 0))
        w.write_ue(params.get('conf_top', 0))
        w.write_ue(params.get('conf_bottom', 0))

    # bit_depth_luma_minus8
    w.write_ue(bd_luma_m8)
    # bit_depth_chroma_minus8
    w.write_ue(bd_chroma_m8)

    # log2_max_pic_order_cnt_lsb_minus4
    w.write_ue(log2_poc)

    # sps_sub_layer_ordering_info_present_flag
    w.write(1, 1)
    # For i=0 only:
    w.write_ue(4)  # max_dec_pic_buffering_minus1
    w.write_ue(2)  # max_num_reorder_pics
    w.write_ue(0)  # max_latency_increase_plus1

    # log2_min_luma_coding_block_size_minus3
    w.write_ue(0)  # = 3 -> min CB size = 8
    # log2_diff_max_min_luma_coding_block_size
    w.write_ue(3)  # max CB size = 64
    # log2_min_luma_transform_block_size_minus2
    w.write_ue(0)  # min TB = 4
    # log2_diff_max_min_luma_transform_block_size
    w.write_ue(3)  # max TB = 32
    # max_transform_hierarchy_depth_inter
    w.write_ue(1)
    # max_transform_hierarchy_depth_intra
    w.write_ue(1)

    # scaling_list_enabled_flag
    w.write(0, 1)

    # amp_enabled_flag
    w.write(0, 1)
    # sample_adaptive_offset_enabled_flag
    w.write(1, 1)

    # pcm_enabled_flag
    w.write(0, 1)

    # num_short_term_ref_pic_sets: ue(v) = 0
    w.write_ue(0)

    # long_term_ref_pics_present_flag
    w.write(0, 1)

    # sps_temporal_mvp_enabled_flag
    w.write(1, 1)
    # strong_intra_smoothing_enabled_flag
    w.write(1, 1)

    # vui_parameters_present_flag
    w.write(0, 1)

    # sps_extension_present_flag
    w.write(0, 1)

    w.write_trailing_bits()

    rbsp = w.add_emulation_prevention()
    nal = bytearray(b'\x00\x00\x00\x01')
    nal.append(0x42)  # NAL type 33 (SPS): (33 << 1) = 0x42
    nal.append(0x01)  # nuh_layer_id=0, nuh_temporal_id_plus1=1
    nal.extend(rbsp)
    return bytes(nal)


def build_pps():
    """
    Build a minimal, standard-compliant PPS NAL unit.

    References SPS 0 and uses simple default values.
    """
    w = BitWriter()

    # pps_pic_parameter_set_id: ue(v) = 0
    w.write_ue(0)
    # pps_seq_parameter_set_id: ue(v) = 0
    w.write_ue(0)

    # dependent_slice_segments_enabled_flag
    w.write(0, 1)
    # output_flag_present_flag
    w.write(0, 1)
    # num_extra_slice_header_bits: u(3) = 0
    w.write(0, 3)
    # sign_data_hiding_enabled_flag
    w.write(0, 1)
    # cabac_init_present_flag
    w.write(1, 1)

    # num_ref_idx_l0_default_active_minus1
    w.write_ue(0)
    # num_ref_idx_l1_default_active_minus1
    w.write_ue(0)

    # init_qp_minus26: se(v) = 0
    w.write_se(0)

    # constrained_intra_pred_flag
    w.write(0, 1)
    # transform_skip_enabled_flag
    w.write(0, 1)

    # cu_qp_delta_enabled_flag
    w.write(0, 1)

    # No cb_qp_offset, cr_qp_offset needed since cu_qp_delta_enabled=0
    # pps_cb_qp_offset: se(v) = 0
    w.write_se(0)
    # pps_cr_qp_offset: se(v) = 0
    w.write_se(0)

    # pps_slice_chroma_qp_offsets_present_flag
    w.write(0, 1)
    # weighted_pred_flag
    w.write(0, 1)
    # weighted_bipred_flag
    w.write(0, 1)
    # transquant_bypass_enabled_flag
    w.write(0, 1)
    # tiles_enabled_flag
    w.write(0, 1)
    # entropy_coding_sync_enabled_flag
    w.write(0, 1)

    # No tiles or WPP → no more flags

    # loop_filter_across_slices_enabled_flag
    w.write(1, 1)

    # deblocking_filter_control_present_flag
    w.write(0, 1)

    # pps_scaling_list_data_present_flag
    w.write(0, 1)

    # lists_modification_present_flag
    w.write(0, 1)

    # log2_parallel_merge_level_minus2: ue(v) = 0
    w.write_ue(0)

    # slice_segment_header_extension_present_flag
    w.write(0, 1)

    # pps_extension_present_flag
    w.write(0, 1)

    w.write_trailing_bits()

    rbsp = w.add_emulation_prevention()
    nal = bytearray(b'\x00\x00\x00\x01')
    nal.append(0x44)  # NAL type 34 (PPS): (34 << 1) = 0x44
    nal.append(0x01)  # nuh_layer_id=0, nuh_temporal_id_plus1=1
    nal.extend(rbsp)
    return bytes(nal)


# ============================================================================
# NAL unit processing
# ============================================================================

def find_start_codes(data):
    """Find all HEVC start code positions in the data."""
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
    """Extract NAL unit type from first byte of NAL header."""
    return (nal_header_byte >> 1) & 0x3F


def ensure_4byte_sc(nal_with_sc, sc_len):
    """Ensure NAL unit uses 4-byte start code."""
    if sc_len == 4:
        return nal_with_sc
    return bytearray(b'\x00') + nal_with_sc


# ============================================================================
# Main filter
# ============================================================================

def filter_hevc_stream():
    """Read raw HEVC from stdin, filter NAL units, output to stdout."""
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    buf = bytearray()
    total_in = 0
    total_out = 0
    total_filtered = 0
    total_preamble = 0
    chunk_count = 0

    # Stream parameters extracted from original SPS
    stream_params = None
    params_logged = False

    # Generated replacement NAL units
    gen_vps = None
    gen_sps = None
    gen_pps = None

    # Cache for re-injection before IDR frames
    last_vps = None
    last_sps = None
    last_pps = None
    injections = 0

    CHUNK_SIZE = 32768  # 32KB
    MAX_BUFFER = 2 * 1024 * 1024  # 2MB

    while True:
        try:
            chunk = stdin.read(CHUNK_SIZE)
            if not chunk:
                break

            buf.extend(chunk)
            total_in += len(chunk)
            chunk_count += 1

            # Safety: prevent buffer from growing too large
            if len(buf) > MAX_BUFFER:
                positions = find_start_codes(buf)
                if positions:
                    buf = buf[positions[-1][0]:]
                else:
                    buf = buf[-4:]
                continue

            # Strip preamble before first start code
            positions = find_start_codes(buf)
            if not positions:
                if len(buf) > 65536:
                    buf = buf[-4:]
                continue

            first_pos = positions[0][0]
            if first_pos > 0:
                total_preamble += first_pos
                if chunk_count <= 20:
                    preview = buf[:min(16, first_pos)].hex()
                    print(
                        f"Stripped {first_pos} preamble bytes (hex: {preview})",
                        file=sys.stderr,
                    )
                buf = buf[first_pos:]
                positions = find_start_codes(buf)

            if len(positions) < 2:
                continue

            output = bytearray()

            for i in range(len(positions) - 1):
                pos, sc_len = positions[i]
                next_pos, _ = positions[i + 1]

                nal_with_sc = bytearray(buf[pos:next_pos])

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

                # Normalize to 4-byte start code
                normalized = ensure_4byte_sc(nal_with_sc, sc_len)

                # --- Handle VPS ---
                if nal_type == NAL_VPS:
                    if gen_vps is None:
                        # Generate replacement VPS (doesn't need stream params)
                        gen_vps = build_vps()
                        if chunk_count <= 10:
                            print(
                                f"Generated replacement VPS "
                                f"({len(gen_vps)} bytes)",
                                file=sys.stderr,
                            )
                    last_vps = bytearray(gen_vps)
                    output.extend(gen_vps)
                    continue  # skip original VPS

                # --- Handle SPS ---
                if nal_type == NAL_SPS:
                    # Parse original SPS to get resolution etc.
                    if stream_params is None:
                        stream_params = parse_sps_params(normalized, 4)
                        if stream_params and not params_logged:
                            print(
                                f"Parsed EZVIZ SPS: "
                                f"{stream_params['width']}x{stream_params['height']}, "
                                f"chroma={stream_params['chroma_format_idc']}, "
                                f"bd={stream_params['bit_depth_luma']}/"
                                f"{stream_params['bit_depth_chroma']}, "
                                f"poc={stream_params['log2_max_poc_lsb_minus4']+4}, "
                                f"sps_id={stream_params['sps_id']}",
                                file=sys.stderr,
                            )
                            params_logged = True

                    if stream_params is not None:
                        gen_sps = build_sps(stream_params)
                        if chunk_count <= 10:
                            print(
                                f"Generated replacement SPS "
                                f"({len(gen_sps)} bytes) for "
                                f"{stream_params['width']}x{stream_params['height']}",
                                file=sys.stderr,
                            )
                    else:
                        # Fallback: try common EZVIZ resolutions
                        fallback = {
                            'width': 1920, 'height': 1080,
                            'chroma_format_idc': 1,
                            'bit_depth_luma': 8, 'bit_depth_chroma': 8,
                            'log2_max_poc_lsb_minus4': 4,
                            'conf_win': 0,
                        }
                        gen_sps = build_sps(fallback)
                        print(
                            f"WARNING: Could not parse original SPS, "
                            f"using fallback 1920x1080",
                            file=sys.stderr,
                        )

                    last_sps = bytearray(gen_sps)
                    output.extend(gen_sps)
                    continue  # skip original SPS

                # --- Handle PPS ---
                if nal_type == NAL_PPS:
                    if gen_pps is None:
                        gen_pps = build_pps()
                        if chunk_count <= 10:
                            print(
                                f"Generated replacement PPS "
                                f"({len(gen_pps)} bytes)",
                                file=sys.stderr,
                            )
                    last_pps = bytearray(gen_pps)
                    output.extend(gen_pps)
                    continue  # skip original PPS

                # --- Before IDR frames, inject cached VPS/SPS/PPS ---
                if nal_type in IDR_NAL_TYPES:
                    if last_vps and last_sps and last_pps:
                        output.extend(last_vps)
                        output.extend(last_sps)
                        output.extend(last_pps)
                        injections += 1
                        if injections <= 3 or injections % 100 == 0:
                            print(
                                f"Injected VPS/SPS/PPS before IDR type "
                                f"{nal_type} [#{injections}]",
                                file=sys.stderr,
                            )

                # Pass through all other standard NAL units
                output.extend(normalized)

            if output:
                stdout.write(bytes(output))
                stdout.flush()
                total_out += len(output)

            # Keep from last start code onwards
            if positions:
                last_pos = positions[-1][0]
                buf = buf[last_pos:]
            else:
                if len(buf) > 4:
                    buf = buf[-4:]

            if chunk_count % 500 == 0:
                res = "unknown"
                if stream_params:
                    res = f"{stream_params['width']}x{stream_params['height']}"
                print(
                    f"HEVC filter stats: in={total_in // 1024}KB "
                    f"out={total_out // 1024}KB "
                    f"filtered={total_filtered} "
                    f"preamble={total_preamble}B "
                    f"injections={injections} "
                    f"res={res}",
                    file=sys.stderr,
                )

        except BrokenPipeError:
            break
        except Exception as e:
            print(f"HEVC filter error: {e}", file=sys.stderr)
            buf = bytearray()

    # Flush remaining
    if buf:
        try:
            stdout.write(bytes(buf))
            stdout.flush()
        except Exception:
            pass

    print(
        f"HEVC filter done: in={total_in // 1024}KB out={total_out // 1024}KB "
        f"filtered={total_filtered} preamble={total_preamble}B "
        f"injections={injections}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    filter_hevc_stream()
