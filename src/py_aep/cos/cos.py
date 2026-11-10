from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, BinaryIO, NoReturn

"""
source: https://gitlab.com/mattbas/python-lottie/-/blob/master/lib/lottie/parsers/aep/cos.py
documentation: https://github.com/hunger-zh/lottie-docs/blob/main/docs/aep.md#list-btdk
"""


def cos_get(data: Any, *keys: str | int) -> Any:
    """Safely traverse parsed COS data by successive keys.

    Each key is tried as a dict string key first and then as a list
    index. Returns `None` when any step along the path is missing.

    Example::

        cos_get(data, "1", "1", 0, "0", "0")  # -> text string
    """
    cur: Any = data
    for key in keys:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(str(key))
        elif isinstance(cur, list):
            try:
                cur = cur[int(key)]
            except (IndexError, ValueError, TypeError):
                return None
        else:
            return None
    return cur


def run_spans(
    doc: dict[str, Any], doc_key: str, style_key: str | None
) -> list[tuple[int, int, dict[str, Any]]]:
    """Decode a text document's COS run array into `(start, end, payload)`.

    Run arrays (`doc["0"]["5"/"6"/"8"]["0"]`) store `{"0": payload,
    "1": count}` entries whose counts are UTF-16 units over the raw
    text. `style_key` picks the style sub-dict for paragraph ("5") and
    character ("6") runs; kerning runs ("8") use their payload directly.
    Runs with a missing or negative count are skipped. Shared by the
    range models (models/text/ranges.py) and the composed-line resolver
    (resolvers/text_composition.py).
    """
    runs = cos_get(doc, "0", doc_key, "0")
    spans: list[tuple[int, int, dict[str, Any]]] = []
    if not isinstance(runs, list):
        return spans
    pos = 0
    for run in runs:
        length = run.get("1") if isinstance(run, dict) else None
        if not isinstance(length, int) or length < 0:
            continue
        if style_key is None:
            payload = cos_get(run, "0")
        else:
            payload = cos_get(run, "0", "0", style_key)
        spans.append((pos, pos + length, payload if isinstance(payload, dict) else {}))
        pos += length
    return spans


class TokenType(Enum):
    Identifier = auto()  # /foo
    Number = auto()  # 123
    String = auto()  # (foo)
    HexString = auto()  # <f000>
    Boolean = auto()  # true
    ObjectStart = auto()  # <<
    ObjectEnd = auto()  # >>
    ArrayStart = auto()  # [
    ArrayEnd = auto()  # ]
    Null = auto()  # null
    Eof = auto()  # end of file
    IndirectObjectStart = auto()  # obj
    IndirectObjectEnd = auto()  # endobj
    IndirectReference = auto()  # R
    Stream = auto()  # stream...endstream


@dataclass
class Token:
    type: TokenType
    value: Any = None


@dataclass
class IndirectObject:
    object_number: int
    generation_number: int
    data: Any


@dataclass
class IndirectReference:
    object_number: int
    generation_number: int


class CosString(str):
    """A COS string literal that remembers its on-disk encoding.

    After Effects writes some string literals as UTF-16 BE (with a BOM,
    e.g. visible text) and others as raw ASCII / UTF-8 (e.g. bit-flag
    strings like `00001000...`). The encoding cannot be inferred from
    the decoded content, so it is preserved here for byte-faithful
    re-serialization. Behaves exactly like `str` otherwise. New strings
    created without this wrapper default to UTF-16 on write.
    """

    cos_utf16: bool
    #: The literal's own bytes, kept only when no byte-order mark said how
    #: to read them and the lexer fell back to UTF-8. A PDF *text* string
    #: with no BOM is PDFDocEncoding instead (PDF 32000-1 7.9.2.2), which
    #: only the consumer knows to apply - see `decode_pdf_text_string`.
    #: `None` whenever a BOM settled the encoding.
    cos_raw: bytes | None

    def __new__(
        cls, value: str, *, utf16: bool = True, raw: bytes | None = None
    ) -> CosString:
        obj = super().__new__(cls, value)
        obj.cos_utf16 = utf16
        obj.cos_raw = raw
        return obj


@dataclass
class CosName:
    """A COS name object used as a value (e.g. `/CoolTypeFont`).

    Distinct from a dict key identifier: when serialized, a CosName
    writes `/name` while a plain `str` writes `(string)`.
    """

    value: str

    def __str__(self) -> str:
        return self.value


@dataclass
class Stream:
    dictionary: dict[str, Any]
    data: bytes


class CosParser:
    """Recursive-descent parser for the COS syntax of text documents and PDF
    objects.

    The lexer works on the data in memory: the bytes left in `file` (up to
    `max_pos` of them) are read once, and tokens are scanned with regular
    expressions rather than one `read(1)` per byte.
    """

    def __init__(self, file: BinaryIO, max_pos: int | None = None) -> None:
        self.file = file
        data = file.read() if max_pos is None else file.read(max(max_pos, 0))
        self._data: bytes = bytes(data)
        self._pos = 0
        self._end = len(self._data)
        self._limited = max_pos is not None
        self.lookahead: Token = Token(TokenType.Eof)  # Will be set by lex()

    @property
    def max_pos(self) -> int | None:
        """Bytes left to read, or `None` when the data was not limited."""
        return self._end - self._pos if self._limited else None

    def parse(self) -> dict[str, Any] | list[Any] | Any:
        self.lex()
        if self.lookahead.type == TokenType.Identifier:
            return self.parse_dict_content()

        val = self.parse_value()
        if self.lookahead.type == TokenType.Eof:
            return val

        return [val] + self.parse_array_content()

    def parse_value(self) -> Any:
        if self.lookahead.type == TokenType.Identifier:
            val = CosName(self.lookahead.value)
            self.lex()
            return val

        if (
            self.lookahead.type == TokenType.String
            or self.lookahead.type == TokenType.HexString
            or self.lookahead.type == TokenType.Null
            or self.lookahead.type == TokenType.Boolean
            or self.lookahead.type == TokenType.Stream
        ):
            val = self.lookahead.value
            self.lex()
            return val

        if self.lookahead.type == TokenType.Number:
            val = self.lookahead.value
            self.lex()
            if self.lookahead.type == TokenType.Number:
                gen = self.lookahead.value
                state = self.save_state()
                self.lex()
                if self.lookahead.type == TokenType.IndirectObjectStart:
                    self.lex()
                    data = self.parse_value()
                    self.expect(TokenType.IndirectObjectEnd)
                    self.lex()
                    return IndirectObject(val, gen, data)
                elif self.lookahead.type == TokenType.IndirectReference:
                    self.lex()
                    return IndirectReference(val, gen)
                else:
                    self.restore_state(state)
            return val

        if self.lookahead.type == TokenType.ObjectStart:
            self.lex()
            val = self.parse_dict_content()
            self.expect(TokenType.ObjectEnd)
            self.lex()
            if self.lookahead.type == TokenType.Stream:
                val = Stream(val, self.lookahead.value)
                self.lex()
            return val

        if self.lookahead.type == TokenType.ArrayStart:
            self.lex()
            val = self.parse_array_content()
            self.expect(TokenType.ArrayEnd)
            self.lex()
            return val

        raise SyntaxError(f"Expected COS value, got {self.lookahead}")

    def save_state(self) -> tuple[int, Token]:
        return (self._pos, self.lookahead)

    def restore_state(self, state: tuple[int, Token]) -> None:
        self._pos, self.lookahead = state

    def parse_dict_content(self) -> dict[str, Any]:
        value = {}
        while True:
            if (
                self.lookahead.type == TokenType.Eof
                or self.lookahead.type == TokenType.ObjectEnd
            ):
                break

            self.expect(TokenType.Identifier)
            key = self.lookahead.value
            self.lex()
            val = self.parse_value()
            value[key] = val

        return value

    def parse_array_content(self) -> list[Any]:
        value: list[Any] = []
        while True:
            if (
                self.lookahead.type == TokenType.Eof
                or self.lookahead.type == TokenType.ArrayEnd
            ):
                break
            value.append(self.parse_value())
        return value

    def expect(self, token_type: TokenType) -> None:
        if self.lookahead.type != token_type:
            raise SyntaxError(f"Expected {token_type}, got {self.lookahead}")

    def lex(self) -> None:
        self.lookahead = self.lex_token()

    def lex_token(self) -> Token:
        data = self._data
        end = self._end
        # One match skips whitespace and comments and reads the common
        # tokens whole; anything else (strings, hex strings, `#` escapes,
        # malformed input) goes on from `pos` byte by byte below.
        match = _TOKEN_RE.match(data, self._pos, end)
        assert match is not None
        kind = match.lastindex
        if kind != _OTHER:
            self._pos = match.end()
            if kind == _IDENTIFIER:
                return Token(TokenType.Identifier, match.group(kind).decode("latin-1"))
            if kind == _INTEGER:
                return Token(TokenType.Number, int(match.group(kind)))
            if kind == _FLOAT:
                return Token(TokenType.Number, float(match.group(kind)))
            if kind == _KEYWORD:
                return self._keyword_token(match.group(kind))
            return _PUNCTUATION_TOKENS[kind]
        pos = match.end()
        if pos >= end:
            self._pos = end
            return Token(TokenType.Eof)
        c = data[pos]
        self._pos = pos + 1

        # <<
        if c == 0x3C:  # <
            next_char = self.get_char()
            if next_char == b"<":
                return Token(TokenType.ObjectStart)
            elif next_char is None:
                self.raise_lex(b"<")
            elif self.is_hex(next_char):
                return self.lex_hex_string(next_char)
            else:
                self.raise_lex(b"<" + next_char)

        # >>
        if c == 0x3E:  # >
            self.expect_char(b">", b">")
            return Token(TokenType.ObjectEnd)

        if c == 0x5B:  # [
            return Token(TokenType.ArrayStart)

        if c == 0x5D:  # ]
            return Token(TokenType.ArrayEnd)

        # /foo
        if c == 0x2F:  # /
            return self.lex_identifier()

        # (foo)
        if c == 0x28:  # (
            return self.lex_string()

        char = data[pos : pos + 1]
        # Keyword
        if char.isalpha():
            return self.lex_keyword(char)

        # Number
        if char.isdigit() or char in b".+-":
            return self.lex_number(char)

        self.raise_lex(char)

    def expect_char(self, exp: bytes, head: bytes) -> None:
        char = self.get_char()
        if char is None or char != exp:
            self.raise_lex(head + (char or b""), head + exp)

    def raise_lex(self, token: bytes, exp: bytes | None = None) -> NoReturn:
        msg = f"Unknown COS token {token!r}"
        if exp is not None:
            msg += f", expected {exp!r}"
        raise SyntaxError(msg)

    def get_char(self) -> bytes | None:
        """The next byte, or `None` at the end of the (limited) data."""
        pos = self._pos
        if pos >= self._end:
            return None
        self._pos = pos + 1
        return self._data[pos : pos + 1]

    def unget(self) -> None:
        self._pos -= 1

    # Skip until newline
    def lex_comment(self) -> None:
        while True:
            char = self.get_char()
            if char == b"\n":
                break
            elif char is None:
                break

    def lex_number(self, char: bytes) -> Token:
        """A number whose first byte, `char`, was just read: digits with an
        optional fraction, after an optional sign. A lone sign or dot raises
        `ValueError`, as `int` / `float` do."""
        if char == b".":
            match = _DIGITS_RE.match(self._data, self._pos, self._end)
            assert match is not None
            self._pos = match.end()
            return Token(TokenType.Number, float(char + match.group()))
        match = _NUMBER_REST_RE.match(self._data, self._pos, self._end)
        assert match is not None
        self._pos = match.end()
        rest = match.group()
        if b"." in rest:
            return Token(TokenType.Number, float(char + rest))
        return Token(TokenType.Number, int(char + rest))

    def lex_keyword(self, char: bytes) -> Token:
        match = _ALPHA_RE.match(self._data, self._pos, self._end)
        assert match is not None
        self._pos = match.end()
        return self._keyword_token(char + match.group())

    def _keyword_token(self, kw: bytes) -> Token:
        if kw == b"true":
            return Token(TokenType.Boolean, True)
        elif kw == b"false":
            return Token(TokenType.Boolean, False)
        elif kw == b"null":
            return Token(TokenType.Null, None)
        elif kw == b"obj":
            return Token(TokenType.IndirectObjectStart)
        elif kw == b"endobj":
            return Token(TokenType.IndirectObjectEnd)
        elif kw == b"R":
            return Token(TokenType.IndirectReference)
        elif kw == b"stream":
            return self.lex_stream()
        elif kw == b"xref":
            return Token(TokenType.Eof)
        else:
            raise SyntaxError(f"Unknown keyword {kw!r}")

    def lex_stream(self) -> Token:
        char = self.get_char()
        if char == b"\r":
            if self.get_char() != b"\n":
                raise SyntaxError("Invalid newline")
        elif char != b"\n":
            raise SyntaxError("Expected newline after `stream`")

        # The data runs up to the first `endstream`.
        start = self._pos
        end = self._data.find(b"endstream", start, self._end)
        if end < 0:
            self._pos = self._end
            raise SyntaxError("Unterminated stream")
        self._pos = end + len(b"endstream")
        return Token(TokenType.Stream, self._data[start:end])

    def lex_string(self) -> Token:
        # Read the entire string content first: the runs between escapes are
        # sliced whole, up to the first unescaped `)`.
        data = self._data
        end = self._end
        pos = self._pos
        parts: list[bytes] = []
        while True:
            close = data.find(b")", pos, end)
            backslash = data.find(b"\\", pos, end if close < 0 else close)
            if backslash < 0:
                if close < 0:
                    self._pos = end
                    raise SyntaxError("Unterminated string")
                parts.append(data[pos:close])
                self._pos = close + 1
                break
            parts.append(data[pos:backslash])
            self._pos = backslash + 1
            parts.append(self.lex_string_escape())
            pos = self._pos
        string = b"".join(parts)

        # Default to UTF-8 encoding
        encoding = "utf-8"
        no_bom = False

        # Check for BOM at the start of the string and remove it
        if string.startswith(b"\xef\xbb\xbf"):
            # UTF-8 BOM - remove it (3 bytes)
            string = string[3:]
        elif string.startswith(b"\xfe\xff"):
            # UTF-16 BE BOM - remove it (2 bytes)
            string = string[2:]
            encoding = "utf-16-be"
        elif string.startswith(b"\xff\xfe"):
            # UTF-16 LE BOM - remove it (2 bytes)
            string = string[2:]
            encoding = "utf-16-le"
        else:
            no_bom = True

        try:
            decoded = string.decode(encoding)
        except UnicodeDecodeError:
            return Token(TokenType.String, string)
        return Token(
            TokenType.String,
            CosString(
                decoded,
                utf16=encoding == "utf-16-be",
                raw=string if no_bom else None,
            ),
        )

    def lex_string_char(self) -> bytes | None:
        char = self.get_char()
        if char is None:
            raise SyntaxError("Unterminated string")
        elif char == b")":
            return None
        elif char == b"\\":
            return self.lex_string_escape()
        else:
            # AE's COS strings hold binary UTF-16 data, so a raw 0x0d / 0x0a
            # byte is part of a character (e.g. U+300D), not a line ending.
            # Preserve it verbatim - normalizing CR/CRLF to LF (per the COS
            # spec for text) would corrupt the data and break round-trip.
            return char

    def lex_string_escape(self) -> bytes:
        char = self.get_char()
        if char is None:
            raise SyntaxError("Unterminated string")

        if char == b"n":
            return b"\n"
        elif char == b"r":
            return b"\r"
        elif char == b"b":
            return b"\b"
        elif char == b"f":
            return b"\f"
        elif char == b"(":
            return b"("
        elif char == b")":
            return b")"
        elif char == b"\\":
            return b"\\"
        elif self.is_octal(char):
            octal = char[0] - ord(b"0")
            for _ in range(2):
                next_char = self.get_char()
                if next_char is None:
                    break
                elif not self.is_octal(next_char):
                    self.unget()
                    break
                octal = octal * 8 + next_char[0] - ord(b"0")
            return octal.to_bytes(1, "big")

        raise SyntaxError("Invalid escape sequence")

    def is_octal(self, char: bytes) -> bool:
        return b"0" <= char <= b"7"

    def is_hex(self, char: bytes) -> bool:
        return char.isdigit() or b"a" <= char <= b"f" or b"A" <= char <= b"F"

    def lex_hex_string(self, hstr: bytes) -> Token:
        while True:
            char = self.get_char()
            if char is None:
                raise SyntaxError("Unterminated hex string")
            elif self.is_hex(char):
                hstr += char
            elif char == b">":
                break
            elif not char.isspace():
                raise SyntaxError(f"Invalid character in hex string: {char!r}")

        if len(hstr) % 2:
            hstr += b"0"

        data = b""
        for i in range(0, len(hstr), 2):
            data += int(hstr[i : i + 2], 16).to_bytes(1, "big")

        return Token(TokenType.HexString, data)

    def lex_identifier(self) -> Token:
        # Printable ASCII up to a delimiter or whitespace; `#xx` is the byte
        # with that hex code. A `#` the match stops at lacks its two digits.
        match = _IDENT_RE.match(self._data, self._pos, self._end)
        assert match is not None
        self._pos = match.end()
        if self._pos < self._end and self._data[self._pos] == 0x23:  # #
            raise SyntaxError("Invalid identifier")
        raw = match.group()
        if b"#" in raw:
            raw = _HEX_ESCAPE_RE.sub(lambda m: bytes([int(m.group(1), 16)]), raw)
        return Token(TokenType.Identifier, raw.decode("latin-1"))


_TOKEN_RE = re.compile(
    rb"""
    (?:[ \t\n\r\x0b\x0c]+|%[^\n]*\n?)*        # whitespace and comments
    (?:
        (<<)                                    # 1
      | (>>)                                    # 2
      | (\[)                                    # 3
      | (\])                                    # 4
      | /([^\x00-\x20\x7f-\xff()\[\]<>{}/%\#]*)       # 5 name, no escapes,
        (?![^\x00-\x20\x7f-\xff()\[\]<>{}/%])         #   whole (no backtracking)
      | ([+-]?[0-9]+)(?![.0-9])                 # 6 integer
      | ([+-]?[0-9]*\.[0-9]*)                    # 7 real (a lone "." raises)
      | ([A-Za-z]+)                             # 8 keyword
      | ()                                      # 9 anything else
    )
    """,
    re.VERBOSE,
)
"""The tokens `CosParser.lex_token` reads in one match, after the
whitespace and `%` comments before them; group 9 matches the empty string
before any other token, which is then lexed byte by byte."""

_IDENTIFIER, _INTEGER, _FLOAT, _KEYWORD, _OTHER = 5, 6, 7, 8, 9

_PUNCTUATION_TOKENS = {
    1: Token(TokenType.ObjectStart),
    2: Token(TokenType.ObjectEnd),
    3: Token(TokenType.ArrayStart),
    4: Token(TokenType.ArrayEnd),
}
"""Shared tokens for value-less punctuation (tokens are never mutated)."""

_SKIP_RE = re.compile(rb"(?:[ \t\n\r\x0b\x0c]+|%[^\n]*\n?)*")
"""Whitespace (as `bytes.isspace` has it) and `%` comments."""

_DIGITS_RE = re.compile(rb"[0-9]*")
_NUMBER_REST_RE = re.compile(rb"[0-9]*(?:\.[0-9]*)?")
_ALPHA_RE = re.compile(rb"[A-Za-z]*")
_IDENT_RE = re.compile(rb"(?:[^\x00-\x20\x7f-\xff()\[\]<>{}/%#]+|#[0-9A-Fa-f]{2})*")
_HEX_ESCAPE_RE = re.compile(rb"#([0-9A-Fa-f]{2})")
