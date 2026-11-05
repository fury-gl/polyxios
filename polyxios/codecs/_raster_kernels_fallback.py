"""Bit-level kernels for the PNG and JPEG codecs, in pure Python and NumPy.

This is the fallback for ``_raster_kernels``, the compiled twin. Both modules
expose the same five functions with the same results and the same errors;
``_png`` and ``_jpeg`` import whichever is present.

Huffman tables are ``(maxcode, valoffset, huffval)``: ``maxcode[l]`` is the
largest ``l``-bit code of length ``l`` (-1 when no code has that length),
``valoffset[l]`` makes ``huffval[code + valoffset[l]]`` the symbol, and
``huffval`` lists the symbols in code order. A table may be ``None`` when a
scan never reads it.

Decoding looks up the next 16 bits in a table built once per scan, so a
symbol costs one lookup rather than a loop over code lengths. Past a marker
or the end of the data the reader pads with zero bits and remembers how many
it padded; consuming any padding is an error.
"""

import numpy as np

from polyxios.exceptions import CodecError

_BAD_CODE: str = "JPEG scan holds an invalid Huffman code."
_ENDS_EARLY: str = "JPEG scan data ends early (truncated or corrupt)."
_OUT_OF_SEQ: str = "JPEG restart marker out of sequence."
_RANGE: str = "JPEG coefficient out of the 16-bit range (corrupt scan)."
_NO_CODE: str = "JPEG encoder has no code for a needed symbol."


# --------------------------------------------------------------------- PNG


def png_unfilter(data: bytes, height: int, row_bytes: int, bpp: int) -> bytearray:
    """Reverse the PNG per-row filters.

    Parameters
    ----------
    data
        ``height`` rows of one filter byte followed by ``row_bytes`` filtered
        bytes.
    height
        Number of rows.
    row_bytes
        Bytes per unfiltered row.
    bpp
        Bytes per complete pixel, at least 1; the filters reach back this far.

    Returns
    -------
    bytearray
        The ``height * row_bytes`` unfiltered bytes.

    Raises
    ------
    CodecError
        If the data is short, or a row uses a filter type above 4.
    """
    stride = row_bytes + 1
    if len(data) < height * stride:
        raise CodecError("PNG image data ends early.")
    out = bytearray(height * row_bytes)
    prev = bytes(row_bytes)
    for r in range(height):
        base = r * stride
        kind = data[base]
        if kind > 4:
            raise CodecError(f"PNG row {r} uses filter type {kind}; only 0-4 exist.")
        raw = data[base + 1 : base + stride]
        if kind == 0:
            cur = raw
        elif kind == 1:
            cur = _unfilter_sub(raw, bpp)
        elif kind == 2:
            cur = (
                np.frombuffer(raw, np.uint8) + np.frombuffer(prev, np.uint8)
            ).tobytes()
        elif kind == 3:
            cur = _unfilter_average(raw, prev, bpp)
        else:
            cur = _unfilter_paeth(raw, prev, bpp)
        start = r * row_bytes
        out[start : start + row_bytes] = cur
        prev = bytes(cur)
    return out


def _unfilter_sub(raw: bytes, bpp: int) -> bytes:
    """Sub filter: a running sum per byte lane, which NumPy's wrapping cumsum does."""
    arr = np.frombuffer(raw, np.uint8)
    pad = (-arr.size) % bpp
    if pad:
        arr = np.concatenate([arr, np.zeros(pad, np.uint8)])
    summed = np.cumsum(arr.reshape(-1, bpp), axis=0, dtype=np.uint8).reshape(-1)
    return summed[: len(raw)].tobytes()


def _unfilter_average(raw: bytes, prev: bytes, bpp: int) -> bytearray:
    cur = bytearray(len(raw))
    for i in range(len(raw)):
        left = cur[i - bpp] if i >= bpp else 0
        cur[i] = (raw[i] + ((left + prev[i]) >> 1)) & 0xFF
    return cur


def _unfilter_paeth(raw: bytes, prev: bytes, bpp: int) -> bytearray:
    cur = bytearray(len(raw))
    for i in range(len(raw)):
        a = cur[i - bpp] if i >= bpp else 0
        b = prev[i]
        c = prev[i - bpp] if i >= bpp else 0
        p = a + b - c
        pa = abs(p - a)
        pb = abs(p - b)
        pc = abs(p - c)
        if pa <= pb and pa <= pc:
            pred = a
        elif pb <= pc:
            pred = b
        else:
            pred = c
        cur[i] = (raw[i] + pred) & 0xFF
    return cur


# -------------------------------------------------------------- Huffman


def _lut(
    table: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[list[int], list[int]]:
    """Expand a table into 16-bit lookahead lists: code length (17 = none) and symbol."""
    maxcode, valoffset, huffval = table
    lut_len = np.full(1 << 16, 17, np.int64)
    lut_val = np.zeros(1 << 16, np.int64)
    cur = 0
    for length in range(1, 17):
        top = int(maxcode[length])
        if top < 0:
            cur <<= 1
            continue
        if top < cur or top >= (1 << length):
            raise CodecError(_BAD_CODE)
        shift = 16 - length
        start, stop = cur << shift, (top + 1) << shift
        codes = np.arange(start, stop) >> shift
        index = codes + int(valoffset[length])
        if index.min() < 0 or index.max() >= len(huffval):
            raise CodecError(_BAD_CODE)
        lut_len[start:stop] = length
        lut_val[start:stop] = np.asarray(huffval)[index]
        cur = (top + 1) << 1
    return lut_len.tolist(), lut_val.tolist()


class _Bits:
    """The entropy-coded bit reader: unstuffs ``FF 00``, stops at a marker."""

    __slots__ = ("data", "end", "pos", "buf", "cnt", "virt", "stuck")

    def __init__(self, data: bytes, pos: int) -> None:
        self.data = data
        self.end = len(data)
        self.pos = pos
        self.buf = 0
        self.cnt = 0
        self.virt = 0
        self.stuck = False

    def fill(self, need: int) -> None:
        """Pull whole bytes until ``cnt >= need``; past a marker, pad with zeros."""
        data, end = self.data, self.end
        while self.cnt < need:
            if self.stuck:
                b = 0
                self.virt += 8
            else:
                p = self.pos
                b = data[p] if p < end else -1
                if b == 0xFF:
                    if p + 1 < end and data[p + 1] == 0:
                        self.pos = p + 2
                    else:
                        b = -1
                elif b >= 0:
                    self.pos = p + 1
                if b < 0:
                    self.stuck = True
                    self.virt += 8
                    b = 0
            self.buf = (self.buf << 8) | b
            self.cnt += 8

    def decode(self, lut: tuple[list[int], list[int]]) -> int:
        """Read one Huffman symbol using a table from ``_lut``."""
        if self.cnt < 16:
            self.fill(16)
        index = (self.buf >> (self.cnt - 16)) & 0xFFFF
        length = lut[0][index]
        if length == 17:
            if self.cnt - self.virt < 16:
                raise CodecError(_ENDS_EARLY)
            raise CodecError(_BAD_CODE)
        self.drop(length)
        return lut[1][index]

    def drop(self, n: int) -> None:
        if n > self.cnt - self.virt:
            raise CodecError(_ENDS_EARLY)
        self.cnt -= n
        self.buf &= (1 << self.cnt) - 1

    def receive(self, n: int) -> int:
        """Read ``n`` raw bits as an unsigned number."""
        if n == 0:
            return 0
        if self.cnt < n:
            self.fill(n)
        if n > self.cnt - self.virt:
            raise CodecError(_ENDS_EARLY)
        self.cnt -= n
        value = self.buf >> self.cnt
        self.buf &= (1 << self.cnt) - 1
        return value


def _skip_to_marker(data: bytes, pos: int) -> int:
    """Offset of the next marker's ``0xFF`` at or after ``pos``, or ``len(data)``."""
    end = len(data)
    p = pos
    while p < end:
        if data[p] != 0xFF:
            p += 1
        elif p + 1 < end and data[p + 1] == 0:
            p += 2
        else:
            return p
    return end


def _extend(value: int, size: int) -> int:
    if size == 0:
        return 0
    return value if value >= (1 << (size - 1)) else value - (1 << size) + 1


def _store(blk: np.ndarray, k: int, value: int) -> None:
    if value < -32768 or value > 32767:
        raise CodecError(_RANGE)
    blk[k] = value


def _refine(bits: _Bits, blk: np.ndarray, k: int, p1: int, m1: int) -> None:
    """Correction bit for a coefficient that is already nonzero."""
    cur = int(blk[k])
    if bits.receive(1) and (cur & p1) == 0:
        _store(blk, k, cur + p1 if cur >= 0 else cur + m1)


# ------------------------------------------------------------ JPEG scans


def jpeg_decode_scan(
    data: bytes,
    pos: int,
    coefs: list[np.ndarray],
    h: list[int],
    v: list[int],
    blocks_x: list[int],
    blocks_y: list[int],
    dc_tables: list,
    ac_tables: list,
    mcus_x: int,
    mcus_y: int,
    ss: int,
    se: int,
    ah: int,
    al: int,
    restart_interval: int,
    progressive: bool,
) -> int:
    """Decode one entropy-coded JPEG scan into coefficient arrays.

    Parameters
    ----------
    data
        The whole file.
    pos
        Offset of the first entropy-coded byte, just past the SOS header.
    coefs
        One ``int16`` array of shape ``(rows, cols, 64)`` per scan component,
        in zig-zag order, filled in place.
    h, v
        Horizontal and vertical sampling factors per scan component.
    blocks_x, blocks_y
        Block counts per scan component, used when the scan has one component.
    dc_tables, ac_tables
        Huffman tables per scan component (``None`` where unused).
    mcus_x, mcus_y
        Size of the interleaved MCU grid.
    ss, se, ah, al
        Spectral selection and successive approximation of the scan.
    restart_interval
        MCUs between restart markers; 0 for none.
    progressive
        Whether the frame is progressive. Baseline scans are decoded as a
        DC-first pass plus a full AC pass with no EOB runs.

    Returns
    -------
    int
        Offset of the next marker's ``0xFF``.

    Raises
    ------
    CodecError
        On a bad code, truncated data, or an out-of-sequence restart marker.
    """
    n = len(coefs)
    dc_luts = [_lut(t) if t is not None else None for t in dc_tables]
    ac_luts = [_lut(t) if t is not None else None for t in ac_tables]
    if n == 1:
        groups = ([(0, r, c)] for r in range(blocks_y[0]) for c in range(blocks_x[0]))
    else:
        groups = _interleaved_groups(h, v, mcus_x, mcus_y)

    bits = _Bits(data, pos)
    preds = [0] * n
    eobrun = 0
    restart_idx = 0
    for m, group in enumerate(groups):
        if restart_interval and m and m % restart_interval == 0:
            p = _skip_to_marker(data, bits.pos)
            if (
                p + 1 >= len(data)
                or data[p] != 0xFF
                or data[p + 1] != 0xD0 + (restart_idx & 7)
            ):
                raise CodecError(_OUT_OF_SEQ)
            bits = _Bits(data, p + 2)
            preds = [0] * n
            eobrun = 0
            restart_idx += 1
        for c, r, cc in group:
            blk = coefs[c][r, cc]
            if ss == 0:
                if ah == 0:
                    preds[c] = _dc_first(bits, blk, dc_luts[c], preds[c], al)
                elif bits.receive(1):
                    _store(blk, 0, int(blk[0]) | (1 << al))
                if se > 0:
                    eobrun = _ac_part(bits, blk, ac_luts[c], 1, se, ah, al, eobrun)
            else:
                eobrun = _ac_part(bits, blk, ac_luts[c], ss, se, ah, al, eobrun)
    return _skip_to_marker(data, bits.pos)


def _interleaved_groups(h: list[int], v: list[int], mcus_x: int, mcus_y: int):
    for my in range(mcus_y):
        for mx in range(mcus_x):
            yield [
                (c, my * v[c] + iy, mx * h[c] + ix)
                for c in range(len(h))
                for iy in range(v[c])
                for ix in range(h[c])
            ]


def _dc_first(bits: _Bits, blk: np.ndarray, lut, pred: int, al: int) -> int:
    t = bits.decode(lut)
    if t > 16:
        raise CodecError(_BAD_CODE)
    pred += _extend(bits.receive(t), t)
    _store(blk, 0, pred << al)
    return pred


def _ac_part(bits, blk, lut, ss, se, ah, al, eobrun):
    if ah == 0:
        return _ac_first(bits, blk, lut, ss, se, al, eobrun)
    return _ac_refine(bits, blk, lut, ss, se, al, eobrun)


def _ac_first(
    bits: _Bits, blk: np.ndarray, lut, ss: int, se: int, al: int, eobrun: int
) -> int:
    """First AC pass of a spectral band; returns the EOB run still pending."""
    if eobrun > 0:
        return eobrun - 1
    k = ss
    while k <= se:
        rs = bits.decode(lut)
        r = rs >> 4
        s = rs & 15
        if s == 0:
            if r != 15:
                return (1 << r) + bits.receive(r) - 1
            k += 16
            continue
        k += r
        if k > se:
            raise CodecError(_BAD_CODE)
        _store(blk, k, _extend(bits.receive(s), s) << al)
        k += 1
    return 0


def _ac_refine(
    bits: _Bits, blk: np.ndarray, lut, ss: int, se: int, al: int, eobrun: int
) -> int:
    """Successive-approximation AC pass (T.81 G.1.2.3); returns the EOB run still pending."""
    p1 = 1 << al
    m1 = -1 << al
    k = ss
    if eobrun == 0:
        while k <= se:
            rs = bits.decode(lut)
            r = rs >> 4
            s = rs & 15
            if s:
                if s != 1:
                    raise CodecError(_BAD_CODE)
                s = p1 if bits.receive(1) else m1
            elif r != 15:
                eobrun = (1 << r) + bits.receive(r)
                break
            while k <= se:
                if int(blk[k]) != 0:
                    _refine(bits, blk, k, p1, m1)
                else:
                    if r == 0:
                        break
                    r -= 1
                k += 1
            if s:
                if k > se:
                    raise CodecError(_BAD_CODE)
                _store(blk, k, s)
            k += 1
    if eobrun > 0:
        while k <= se:
            if int(blk[k]) != 0:
                _refine(bits, blk, k, p1, m1)
            k += 1
        eobrun -= 1
    return eobrun


# ------------------------------------------------------------- lossless


def jpeg_lossless_decode_scan(
    data: bytes,
    pos: int,
    samples: list[np.ndarray],
    tables: list,
    predictor: int,
    point_transform: int,
    precision: int,
    restart_interval: int,
    width: int,
    height: int,
) -> int:
    """Decode one lossless (SOF3) scan into sample arrays (ITU T.81 Annex H).

    Parameters
    ----------
    data
        The whole file.
    pos
        Offset of the first entropy-coded byte.
    samples
        One ``int32`` array of shape ``(height, width)`` per scan component,
        filled in place. One MCU is one sample of each component, in scan
        order.
    tables
        One ``(maxcode, valoffset, huffval)`` table per scan component.
    predictor
        Scan predictor ``Ss``, 1-7.
    point_transform
        ``Al``; the caller shifts the output, so this only seeds the first sample.
    precision
        Sample precision ``P``.
    restart_interval
        MCUs between restart markers; 0 for none.
    width, height
        Samples per line and number of lines.

    Returns
    -------
    int
        Offset of the next marker's ``0xFF``.

    Raises
    ------
    CodecError
        On a bad predictor, bad code, truncated data, or an out-of-sequence
        restart marker.
    """
    if not 1 <= predictor <= 7:
        raise CodecError(f"JPEG lossless predictor {predictor} is not 1-7.")
    n = len(samples)
    luts = [_lut(t) for t in tables]
    flat = [[0] * (width * height) for _ in range(n)]
    start = 1 << (precision - point_transform - 1)
    bits = _Bits(data, pos)
    m = 0
    restart_idx = 0
    for y in range(height):
        for x in range(width):
            if restart_interval and m and m % restart_interval == 0:
                p = _skip_to_marker(data, bits.pos)
                if (
                    p + 1 >= len(data)
                    or data[p] != 0xFF
                    or data[p + 1] != 0xD0 + (restart_idx & 7)
                ):
                    raise CodecError(_OUT_OF_SEQ)
                bits = _Bits(data, p + 2)
                restart_idx += 1
            fresh = m == 0 or (restart_interval > 0 and m % restart_interval == 0)
            i = y * width + x
            for c in range(n):
                t = bits.decode(luts[c])
                if t > 16:
                    raise CodecError(_BAD_CODE)
                if t == 0:
                    diff = 0
                elif t == 16:
                    diff = 32768
                else:
                    diff = _extend(bits.receive(t), t)
                row = flat[c]
                if fresh:
                    pred = start
                elif y == 0:
                    pred = row[i - 1]
                elif x == 0:
                    pred = row[i - width]
                else:
                    ra = row[i - 1]
                    rb = row[i - width]
                    rc = row[i - width - 1]
                    if predictor == 1:
                        pred = ra
                    elif predictor == 2:
                        pred = rb
                    elif predictor == 3:
                        pred = rc
                    elif predictor == 4:
                        pred = ra + rb - rc
                    elif predictor == 5:
                        pred = ra + ((rb - rc) >> 1)
                    elif predictor == 6:
                        pred = rb + ((ra - rc) >> 1)
                    else:
                        pred = (ra + rb) >> 1
                row[i] = (pred + diff) & 0xFFFF
            m += 1
    for c in range(n):
        samples[c][:, :] = np.array(flat[c], dtype=np.int32).reshape(height, width)
    return _skip_to_marker(data, bits.pos)


class _BitWriter:
    """Packs bits MSB first, stuffs ``0xFF`` with ``0x00``, pads with 1-bits."""

    __slots__ = ("out", "acc", "nb")

    def __init__(self) -> None:
        self.out = bytearray()
        self.acc = 0
        self.nb = 0

    def put(self, value: int, n: int) -> None:
        if n == 0:
            return
        self.acc = (self.acc << n) | (value & ((1 << n) - 1))
        self.nb += n
        while self.nb >= 8:
            self.nb -= 8
            byte = (self.acc >> self.nb) & 0xFF
            self.out.append(byte)
            if byte == 0xFF:
                self.out.append(0)
        self.acc &= (1 << self.nb) - 1

    def finish(self) -> bytes:
        if self.nb:
            pad = 8 - self.nb
            byte = ((self.acc << pad) | ((1 << pad) - 1)) & 0xFF
            self.out.append(byte)
            if byte == 0xFF:
                self.out.append(0)
            self.acc = 0
            self.nb = 0
        return bytes(self.out)


def _category(value: int) -> tuple[int, int]:
    """Magnitude category ``SSSS`` of a difference and its extra bits."""
    size = abs(value).bit_length()
    if size == 0:
        return 0, 0
    return size, value if value > 0 else value + (1 << size) - 1


def jpeg_lossless_encode(
    diffs: np.ndarray, code: np.ndarray, size: np.ndarray
) -> bytes:
    """Entropy-code lossless differences (SSSS symbol, then extra bits).

    Parameters
    ----------
    diffs
        ``int32`` differences in scan order, each in -32767..32768.
    code, size
        ``uint16[17]`` codes and ``uint8[17]`` lengths, indexed by ``SSSS``.

    Returns
    -------
    bytes
        The stuffed, 1-padded entropy-coded segment.

    Raises
    ------
    CodecError
        If a difference is out of range or its symbol has no code.
    """
    codes = [int(x) for x in code]
    sizes = [int(x) for x in size]
    w = _BitWriter()
    for d in diffs.tolist():
        if d < -32767 or d > 32768:
            raise CodecError(f"JPEG lossless difference {d} is out of range.")
        if d == 32768:
            t, extra = 16, 0
        else:
            t, extra = _category(d)
        if sizes[t] == 0:
            raise CodecError(_NO_CODE)
        w.put(codes[t], sizes[t])
        w.put(extra, t if t < 16 else 0)
    return w.finish()


def jpeg_encode_scan(
    blocks: np.ndarray,
    comp: np.ndarray,
    table: np.ndarray,
    dc_code: np.ndarray,
    dc_size: np.ndarray,
    ac_code: np.ndarray,
    ac_size: np.ndarray,
    n_comps: int,
) -> bytes:
    """Entropy-code baseline blocks with the given Huffman tables.

    Parameters
    ----------
    blocks
        ``int32 (N, 64)`` zig-zag quantised coefficients, in emission order.
    comp
        ``uint8 (N,)`` DC predictor slot of each block.
    table
        ``uint8 (N,)`` Huffman table of each block: 0 luma, 1 chroma.
    dc_code, dc_size
        ``uint16 (2, 12)`` codes and ``uint8 (2, 12)`` lengths for DC categories.
    ac_code, ac_size
        ``uint16 (2, 256)`` codes and ``uint8 (2, 256)`` lengths for AC run/size.
    n_comps
        Number of DC predictor slots.

    Returns
    -------
    bytes
        The stuffed, 1-padded entropy-coded segment.

    Raises
    ------
    CodecError
        If a coefficient needs a category the baseline tables lack.
    """
    dcc = dc_code.tolist()
    dcs = dc_size.tolist()
    acc = ac_code.tolist()
    acs = ac_size.tolist()
    preds = [0] * n_comps
    w = _BitWriter()
    for row, c, tb in zip(blocks.tolist(), comp.tolist(), table.tolist()):
        dc = row[0]
        diff = dc - preds[c]
        preds[c] = dc
        t, extra = _category(diff)
        if t > 11:
            raise CodecError(_RANGE)
        w.put(dcc[tb][t], dcs[tb][t])
        w.put(extra, t)
        run = 0
        for k in range(1, 64):
            v = row[k]
            if v == 0:
                run += 1
                continue
            while run > 15:
                w.put(acc[tb][0xF0], acs[tb][0xF0])
                run -= 16
            s, extra = _category(v)
            if s > 10:
                raise CodecError(_RANGE)
            sym = (run << 4) | s
            w.put(acc[tb][sym], acs[tb][sym])
            w.put(extra, s)
            run = 0
        if run:
            w.put(acc[tb][0], acs[tb][0])
    return w.finish()
