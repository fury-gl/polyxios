# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Bit-level kernels for the PNG and JPEG codecs, compiled.

The compiled twin of ``_raster_kernels_fallback``: same functions, same
results byte for byte, same errors. ``_png`` and ``_jpeg`` import whichever
of the two is present. See the fallback module for the table layout and the
decoding rules; this file keeps the same structure with typed locals.
"""

import numpy as np

from polyxios.exceptions import CodecError

_BAD_CODE = "JPEG scan holds an invalid Huffman code."
_ENDS_EARLY = "JPEG scan data ends early (truncated or corrupt)."
_OUT_OF_SEQ = "JPEG restart marker out of sequence."
_RANGE = "JPEG coefficient out of the 16-bit range (corrupt scan)."
_NO_CODE = "JPEG encoder has no code for a needed symbol."


# --------------------------------------------------------------------- PNG


def png_unfilter(const unsigned char[::1] data, Py_ssize_t height, Py_ssize_t row_bytes, Py_ssize_t bpp):
    """Reverse the PNG per-row filters; see the fallback for the contract."""
    cdef Py_ssize_t stride = row_bytes + 1
    cdef Py_ssize_t r, i, base, start, up
    cdef int kind, raw, a, b, c, p, pa, pb, pc, pred
    if <Py_ssize_t>data.shape[0] < height * stride:
        raise CodecError("PNG image data ends early.")
    out = bytearray(height * row_bytes)
    cdef unsigned char[::1] o = out
    for r in range(height):
        base = r * stride
        kind = data[base]
        if kind > 4:
            raise CodecError(f"PNG row {r} uses filter type {kind}; only 0-4 exist.")
        start = r * row_bytes
        for i in range(row_bytes):
            raw = <int>data[base + 1 + i]
            up = <int>o[start - row_bytes + i] if r > 0 else 0
            if kind == 0:
                pred = 0
            elif kind == 1:
                pred = <int>o[start + i - bpp] if i >= bpp else 0
            elif kind == 2:
                pred = up
            elif kind == 3:
                a = <int>o[start + i - bpp] if i >= bpp else 0
                pred = (a + up) >> 1
            else:
                a = <int>o[start + i - bpp] if i >= bpp else 0
                c = <int>o[start - row_bytes + i - bpp] if (r > 0 and i >= bpp) else 0
                b = up
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
            o[start + i] = (raw + pred) & 0xFF
    return out


# -------------------------------------------------------------- Huffman


def _lut(table):
    """Expand a table into 16-bit lookahead arrays: code length (17 = none) and symbol."""
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
    return lut_len.astype(np.int32), lut_val.astype(np.int32)


cdef class _Bits:
    """The entropy-coded bit reader: unstuffs ``FF 00``, stops at a marker."""

    cdef const unsigned char[::1] data
    cdef Py_ssize_t end
    cdef Py_ssize_t pos
    cdef unsigned long long buf
    cdef int cnt
    cdef int virt
    cdef bint stuck

    def __init__(self, const unsigned char[::1] data, Py_ssize_t pos):
        self.data = data
        self.end = data.shape[0]
        self.pos = pos
        self.buf = 0
        self.cnt = 0
        self.virt = 0
        self.stuck = False

    cdef void fill(self, int need):
        cdef int b
        cdef Py_ssize_t p
        while self.cnt < need:
            if self.stuck:
                b = 0
                self.virt += 8
            else:
                p = self.pos
                b = self.data[p] if p < self.end else -1
                if b == 0xFF:
                    if p + 1 < self.end and self.data[p + 1] == 0:
                        self.pos = p + 2
                    else:
                        b = -1
                elif b >= 0:
                    self.pos = p + 1
                if b < 0:
                    self.stuck = True
                    self.virt += 8
                    b = 0
            self.buf = (self.buf << 8) | <unsigned long long>b
            self.cnt += 8

    cdef int decode(self, const int[::1] lut_len, const int[::1] lut_val) except -1:
        cdef int length
        cdef Py_ssize_t index
        if self.cnt < 16:
            self.fill(16)
        index = <Py_ssize_t>((self.buf >> (self.cnt - 16)) & 0xFFFF)
        length = lut_len[index]
        if length == 17:
            if self.cnt - self.virt < 16:
                raise CodecError(_ENDS_EARLY)
            raise CodecError(_BAD_CODE)
        self.drop(length)
        return lut_val[index]

    cdef void drop(self, int n) except *:
        if n > self.cnt - self.virt:
            raise CodecError(_ENDS_EARLY)
        self.cnt -= n
        self.buf &= (<unsigned long long>1 << self.cnt) - 1

    cdef int receive(self, int n) except? -1:
        cdef int value
        if n == 0:
            return 0
        if self.cnt < n:
            self.fill(n)
        if n > self.cnt - self.virt:
            raise CodecError(_ENDS_EARLY)
        self.cnt -= n
        value = <int>(self.buf >> self.cnt)
        self.buf &= (<unsigned long long>1 << self.cnt) - 1
        return value


cdef Py_ssize_t _skip_to_marker(const unsigned char[::1] data, Py_ssize_t pos):
    cdef Py_ssize_t end = data.shape[0]
    cdef Py_ssize_t p = pos
    while p < end:
        if data[p] != 0xFF:
            p += 1
        elif p + 1 < end and data[p + 1] == 0:
            p += 2
        else:
            return p
    return end


cdef int _extend(int value, int size):
    if size == 0:
        return 0
    return value if value >= (1 << (size - 1)) else value - (1 << size) + 1


cdef void _store(short[::1] blk, int k, long long value) except *:
    if value < -32768 or value > 32767:
        raise CodecError(_RANGE)
    blk[k] = <short>value


cdef void _refine(_Bits bits, short[::1] blk, int k, int p1, int m1) except *:
    cdef int cur = blk[k]
    if bits.receive(1) and (cur & p1) == 0:
        _store(blk, k, cur + p1 if cur >= 0 else cur + m1)


cdef int _dc_first(_Bits bits, short[::1] blk, const int[::1] ll, const int[::1] lv, int pred, int al) except? -1000000000:
    cdef int t = bits.decode(ll, lv)
    cdef long long value
    if t > 16:
        raise CodecError(_BAD_CODE)
    value = pred + _extend(bits.receive(t), t)
    _store(blk, 0, value << al)
    return <int>value


cdef int _ac_first(_Bits bits, short[::1] blk, const int[::1] ll, const int[::1] lv,
                   int ss, int se, int al, int eobrun) except? -1000000000:
    cdef int k, rs, r, s
    if eobrun > 0:
        return eobrun - 1
    k = ss
    while k <= se:
        rs = bits.decode(ll, lv)
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


cdef int _ac_refine(_Bits bits, short[::1] blk, const int[::1] ll, const int[::1] lv,
                    int ss, int se, int al, int eobrun) except? -1000000000:
    cdef int p1 = 1 << al
    cdef int m1 = -1 << al
    cdef int k = ss
    cdef int rs, r, s
    if eobrun == 0:
        while k <= se:
            rs = bits.decode(ll, lv)
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
                if blk[k] != 0:
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
            if blk[k] != 0:
                _refine(bits, blk, k, p1, m1)
            k += 1
        eobrun -= 1
    return eobrun




def jpeg_decode_scan(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    list coefs,
    list h,
    list v,
    list blocks_x,
    list blocks_y,
    list dc_tables,
    list ac_tables,
    Py_ssize_t mcus_x,
    Py_ssize_t mcus_y,
    int ss,
    int se,
    int ah,
    int al,
    Py_ssize_t restart_interval,
    bint progressive,
):
    """Decode one entropy-coded JPEG scan into coefficient arrays; see the fallback."""
    cdef Py_ssize_t n = len(coefs)
    cdef Py_ssize_t m, restart_idx = 0, p, r, cc, bx, c, my, mx, iy, ix
    cdef int eobrun = 0
    cdef list dc_luts = [_lut(t) if t is not None else None for t in dc_tables]
    cdef list ac_luts = [_lut(t) if t is not None else None for t in ac_tables]
    cdef list preds = [0] * n
    cdef _Bits bits = _Bits(data, pos)
    cdef short[:, :, ::1] arr
    cdef short[::1] blk
    cdef const int[::1] ll, lv
    cdef int total
    if n == 1:
        bx = blocks_x[0]
        total = blocks_y[0] * bx
    else:
        total = mcus_x * mcus_y
    for m in range(total):
        if restart_interval and m and m % restart_interval == 0:
            p = _skip_to_marker(data, bits.pos)
            if p + 1 >= <Py_ssize_t>data.shape[0] or data[p] != 0xFF or data[p + 1] != 0xD0 + (restart_idx & 7):
                raise CodecError(_OUT_OF_SEQ)
            bits = _Bits(data, p + 2)
            preds = [0] * n
            eobrun = 0
            restart_idx += 1
        if n == 1:
            arr = coefs[0]
            r = m // bx
            cc = m % bx
            blk = arr[r, cc]
            _decode_block(bits, blk, dc_luts, ac_luts, preds, 0, ss, se, ah, al, &eobrun)
        else:
            my = m // mcus_x
            mx = m % mcus_x
            for c in range(n):
                arr = coefs[c]
                for iy in range(v[c]):
                    for ix in range(h[c]):
                        blk = arr[my * v[c] + iy, mx * h[c] + ix]
                        _decode_block(bits, blk, dc_luts, ac_luts, preds, c, ss, se, ah, al, &eobrun)
    return _skip_to_marker(data, bits.pos)


cdef void _decode_block(_Bits bits, short[::1] blk, list dc_luts, list ac_luts, list preds,
                        Py_ssize_t c, int ss, int se, int ah, int al, int* eob) except *:
    cdef const int[::1] ll, lv
    cdef int pred
    cdef tuple lut
    if ss == 0:
        if ah == 0:
            lut = dc_luts[c]
            ll = lut[0]
            lv = lut[1]
            pred = _dc_first(bits, blk, ll, lv, preds[c], al)
            preds[c] = pred
        elif bits.receive(1):
            _store(blk, 0, <long long>blk[0] | (1 << al))
        if se > 0:
            lut = ac_luts[c]
            ll = lut[0]
            lv = lut[1]
            eob[0] = _ac_part(bits, blk, ll, lv, 1, se, ah, al, eob[0])
    else:
        lut = ac_luts[c]
        ll = lut[0]
        lv = lut[1]
        eob[0] = _ac_part(bits, blk, ll, lv, ss, se, ah, al, eob[0])


cdef int _ac_part(_Bits bits, short[::1] blk, const int[::1] ll, const int[::1] lv,
                  int ss, int se, int ah, int al, int eobrun) except? -1000000000:
    if ah == 0:
        return _ac_first(bits, blk, ll, lv, ss, se, al, eobrun)
    return _ac_refine(bits, blk, ll, lv, ss, se, al, eobrun)


def jpeg_lossless_decode_scan(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    list samples,
    list tables,
    int predictor,
    int point_transform,
    int precision,
    Py_ssize_t restart_interval,
    Py_ssize_t width,
    Py_ssize_t height,
):
    """Decode one lossless (SOF3) scan into sample arrays; see the fallback."""
    if not 1 <= predictor <= 7:
        raise CodecError(f"JPEG lossless predictor {predictor} is not 1-7.")
    cdef Py_ssize_t n = len(samples)
    cdef list luts = [_lut(table) for table in tables]
    cdef list flats = [np.zeros(width * height, np.int32) for _ in range(n)]
    cdef list views = [f for f in flats]
    cdef long long start = 1 << (precision - point_transform - 1)
    cdef _Bits bits = _Bits(data, pos)
    cdef Py_ssize_t m = 0, restart_idx = 0, y, x, i, c, p
    cdef bint fresh
    cdef int t
    cdef long long diff, pred, ra, rb, rc
    cdef const int[::1] ll, lv
    cdef int[::1] row
    cdef tuple lut
    for y in range(height):
        for x in range(width):
            if restart_interval and m and m % restart_interval == 0:
                p = _skip_to_marker(data, bits.pos)
                if p + 1 >= <Py_ssize_t>data.shape[0] or data[p] != 0xFF or data[p + 1] != 0xD0 + (restart_idx & 7):
                    raise CodecError(_OUT_OF_SEQ)
                bits = _Bits(data, p + 2)
                restart_idx += 1
            fresh = m == 0 or (restart_interval > 0 and m % restart_interval == 0)
            i = y * width + x
            for c in range(n):
                lut = luts[c]
                ll = lut[0]
                lv = lut[1]
                t = bits.decode(ll, lv)
                if t > 16:
                    raise CodecError(_BAD_CODE)
                if t == 0:
                    diff = 0
                elif t == 16:
                    diff = 32768
                else:
                    diff = _extend(bits.receive(t), t)
                row = views[c]
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
                row[i] = <int>((pred + diff) & 0xFFFF)
            m += 1
    for c in range(n):
        samples[c][:, :] = flats[c].reshape(height, width)
    return _skip_to_marker(data, bits.pos)


# ------------------------------------------------------------- encoders


cdef class _BitWriter:
    """Packs bits MSB first, stuffs ``0xFF`` with ``0x00``, pads with 1-bits."""

    cdef object out
    cdef unsigned long long acc
    cdef int nb

    def __init__(self):
        self.out = bytearray()
        self.acc = 0
        self.nb = 0

    cdef void put(self, unsigned long long value, int n):
        cdef int byte
        if n == 0:
            return
        self.acc = (self.acc << n) | (value & ((<unsigned long long>1 << n) - 1))
        self.nb += n
        while self.nb >= 8:
            self.nb -= 8
            byte = <int>((self.acc >> self.nb) & 0xFF)
            self.out.append(byte)
            if byte == 0xFF:
                self.out.append(0)
        self.acc &= (<unsigned long long>1 << self.nb) - 1

    cdef bytes finish(self):
        cdef int pad, byte
        if self.nb:
            pad = 8 - self.nb
            byte = <int>(((self.acc << pad) | ((<unsigned long long>1 << pad) - 1)) & 0xFF)
            self.out.append(byte)
            if byte == 0xFF:
                self.out.append(0)
            self.acc = 0
            self.nb = 0
        return bytes(self.out)


cdef int _bit_length(long long a):
    cdef int size = 0
    while a:
        size += 1
        a >>= 1
    return size


cdef tuple _category(long long value):
    cdef int size = _bit_length(value if value >= 0 else -value)
    if size == 0:
        return 0, 0
    return size, (value if value > 0 else value + (<long long>1 << size) - 1)


def jpeg_lossless_encode(diffs, code, size):
    """Entropy-code lossless differences; see the fallback."""
    cdef list codes = [int(x) for x in code]
    cdef list sizes = [int(x) for x in size]
    cdef _BitWriter w = _BitWriter()
    cdef long long d
    cdef int t
    cdef long long extra
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


def jpeg_encode_scan(blocks, comp, table, dc_code, dc_size, ac_code, ac_size, Py_ssize_t n_comps):
    """Entropy-code baseline blocks with the given Huffman tables; see the fallback."""
    cdef list dcc = dc_code.tolist()
    cdef list dcs = dc_size.tolist()
    cdef list acc = ac_code.tolist()
    cdef list acs = ac_size.tolist()
    cdef list preds = [0] * n_comps
    cdef _BitWriter w = _BitWriter()
    cdef list row
    cdef long long dc, diff, v
    cdef int c, tb, t, k, run, s, sym
    cdef long long extra
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
