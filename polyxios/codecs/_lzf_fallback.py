"""LZF block compression in pure Python, the fallback for ``_lzf``.

LZF is the codec PCL uses for ``DATA binary_compressed``: a stream of
literal runs and back-references, no header, no checksum. A control byte
below 32 starts a literal run of ``ctrl + 1`` bytes; any other starts a
back-reference whose length is its top three bits plus two, with a length of
seven meaning "read one more length byte", and whose offset is its low five
bits shifted by eight plus the next byte, plus one, counted back from the
output end. The compressor here is a greedy hash-table one with liblzf's
8 KiB window and 264-byte match limit, so any LZF decoder reads its output
and this one reads liblzf's; the streams themselves differ, since the hash
picks different matches. The compiled module and this one agree byte for
byte.
"""

from __future__ import annotations

from polyxios.exceptions import CodecError

_MAX_LIT: int = 32
_MAX_OFF: int = 1 << 13
_MAX_REF: int = (1 << 8) + (1 << 3)
_HLOG: int = 16
_HSIZE: int = 1 << _HLOG


def decompress(data: bytes, out_len: int) -> bytes:
    """Expand one LZF block.

    Parameters
    ----------
    data
        The compressed bytes.
    out_len
        The exact length the block expands to; a PCD header spells it.

    Returns
    -------
    bytes
        The expanded bytes, ``out_len`` long.

    Raises
    ------
    CodecError
        When the stream ends mid-instruction, a back-reference reaches
        before the start of the output, or the block does not expand to
        ``out_len`` bytes exactly.
    """
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        ctrl = data[i]
        i += 1
        if ctrl < _MAX_LIT:
            run = ctrl + 1
            if i + run > n:
                raise CodecError("LZF stream ends inside a literal run.")
            if len(out) + run > out_len:
                raise CodecError(
                    f"LZF block expands past the {out_len} bytes the header promised."
                )
            out += data[i : i + run]
            i += run
            continue
        length = ctrl >> 5
        if length == 7:
            if i >= n:
                raise CodecError("LZF stream ends inside a back-reference.")
            length += data[i]
            i += 1
        if i >= n:
            raise CodecError("LZF stream ends inside a back-reference.")
        offset = ((ctrl & 0x1F) << 8) + data[i] + 1
        i += 1
        length += 2
        start = len(out) - offset
        if start < 0:
            raise CodecError("LZF back-reference reaches before the output start.")
        if len(out) + length > out_len:
            raise CodecError(
                f"LZF block expands past the {out_len} bytes the header promised."
            )
        if offset >= length:
            out += out[start : start + length]
        else:
            # An overlapping reference repeats what it has just emitted, so
            # the bytes are copied one at a time in order.
            for _ in range(length):
                out.append(out[start])
                start += 1
    if len(out) != out_len:
        raise CodecError(
            f"LZF block expands to {len(out)} bytes, the header promised {out_len}."
        )
    return bytes(out)


def compress(data: bytes) -> bytes:
    """Compress one block into a stream any LZF decoder reads.

    Parameters
    ----------
    data
        The bytes to compress.

    Returns
    -------
    bytes
        The LZF stream. Incompressible input grows by one byte per 32.
    """
    n = len(data)
    out = bytearray()
    if n == 0:
        return bytes(out)
    htab = [-1] * _HSIZE
    lit: list[int] = []

    def flush_lit() -> None:
        while lit:
            chunk = lit[:_MAX_LIT]
            del lit[:_MAX_LIT]
            out.append(len(chunk) - 1)
            out.extend(chunk)

    i = 0
    while i < n:
        if i + 2 < n:
            h = ((data[i] << 16) | (data[i + 1] << 8) | data[i + 2]) & 0xFFFFFF
            h = ((h * 2654435761) >> (24 - _HLOG + 8)) & (_HSIZE - 1)
            ref = htab[h]
            htab[h] = i
            off = i - ref - 1
            if (
                ref >= 0
                and off < _MAX_OFF
                and data[ref] == data[i]
                and data[ref + 1] == data[i + 1]
                and data[ref + 2] == data[i + 2]
            ):
                # The three hashed bytes are known equal, so the match starts
                # at 3: a shorter one would encode as a control byte below
                # 32, which every decoder reads as a literal run.
                length = 3
                maxlen = min(n - i - 2, _MAX_REF)
                while length < maxlen and data[ref + length] == data[i + length]:
                    length += 1
                length -= 2
                flush_lit()
                if length < 7:
                    out.append((off >> 8) + (length << 5))
                else:
                    out.append((off >> 8) + (7 << 5))
                    out.append(length - 7)
                out.append(off & 0xFF)
                i += length + 2
                # liblzf re-enters the table at the match's last byte, so a
                # run continuing past it is found again from there.
                if 0 < i - 1 and i + 1 < n:
                    j = i - 1
                    h = (data[j] << 16) | (data[j + 1] << 8) | data[j + 2]
                    h = ((h * 2654435761) >> (24 - _HLOG + 8)) & (_HSIZE - 1)
                    htab[h] = j
                continue
        lit.append(data[i])
        i += 1
    flush_lit()
    return bytes(out)
