import base64
import binascii
import io
import os
import struct
import tempfile

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


_MAGIC = b'SECMED01'
_CHUNK_SIZE = 1024 * 1024
_TAG_SIZE = 16
_HEADER_FIELDS = struct.Struct('>8sQI16s8s')
_HEADER_SIZE = _HEADER_FIELDS.size + _TAG_SIZE
_HEADER_NONCE_INDEX = 0xFFFFFFFF


def _derive_file_key(master_key, salt):
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=b'SECMEDIA file encryption v1',
    ).derive(master_key)


class MediaStorageError(Exception):
    """Raised when an encrypted media operation cannot be completed."""


class MediaEncryptionKeyError(MediaStorageError):
    """Raised when encrypted writes are requested without a valid key."""


class EncryptedMediaReader:
    """Seekable reader that authenticates encrypted chunks as they are read."""

    def __init__(self, path, key):
        self._file = open(path, 'rb')
        self._key = key
        header = self._file.read(_HEADER_SIZE)
        if len(header) != _HEADER_SIZE:
            self._file.close()
            raise MediaStorageError(f'Encrypted media header is truncated: {path}')

        (
            magic,
            self._size,
            self._chunk_size,
            salt,
            self._nonce_prefix,
        ) = _HEADER_FIELDS.unpack(header[:_HEADER_FIELDS.size])
        if magic != _MAGIC or self._chunk_size <= 0:
            self._file.close()
            raise MediaStorageError(f'Encrypted media header is invalid: {path}')

        self._header_fields = header[:_HEADER_FIELDS.size]
        self._file_key = _derive_file_key(self._key, salt)
        expected_chunks = (
            (self._size + self._chunk_size - 1) // self._chunk_size
        )
        expected_file_size = _HEADER_SIZE + self._size + expected_chunks * _TAG_SIZE
        if os.fstat(self._file.fileno()).st_size != expected_file_size:
            self._file.close()
            raise MediaStorageError(f'Encrypted media length is invalid: {path}')
        header_nonce = self._nonce_prefix + struct.pack('>I', _HEADER_NONCE_INDEX)
        try:
            AESGCM(self._file_key).decrypt(
                header_nonce,
                header[_HEADER_FIELDS.size:],
                self._header_fields,
            )
        except InvalidTag as exc:
            self._file.close()
            raise MediaStorageError(f'Encrypted media header authentication failed: {path}') from exc

        self._position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._position

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self._position + offset
        elif whence == os.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError(f'Invalid seek mode: {whence}')
        if position < 0:
            raise ValueError('Negative seek position')
        self._position = min(position, self._size)
        return self._position

    def read(self, size=-1):
        if size is None or size < 0:
            size = self._size - self._position
        size = min(size, self._size - self._position)
        if size <= 0:
            return b''

        output = bytearray()
        while size:
            chunk_index, chunk_offset = divmod(self._position, self._chunk_size)
            chunk_start = chunk_index * self._chunk_size
            plaintext_length = min(self._chunk_size, self._size - chunk_start)
            disk_offset = _HEADER_SIZE + chunk_index * (self._chunk_size + _TAG_SIZE)
            self._file.seek(disk_offset)
            encrypted_chunk = self._file.read(plaintext_length + _TAG_SIZE)
            if len(encrypted_chunk) != plaintext_length + _TAG_SIZE:
                raise MediaStorageError('Encrypted media chunk is truncated.')

            nonce = self._nonce_prefix + struct.pack('>I', chunk_index)
            aad = self._header_fields + struct.pack('>I', chunk_index)
            try:
                plaintext = AESGCM(self._file_key).decrypt(nonce, encrypted_chunk, aad)
            except InvalidTag as exc:
                raise MediaStorageError(
                    f'Encrypted media chunk {chunk_index} failed authentication.'
                ) from exc

            take = min(size, len(plaintext) - chunk_offset)
            output.extend(plaintext[chunk_offset:chunk_offset + take])
            self._position += take
            size -= take
        return bytes(output)

    def close(self):
        self._file.close()

    @property
    def closed(self):
        return self._file.closed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class MediaStorage:
    """Application media I/O with encrypted writes and legacy plaintext reads."""

    def __init__(self, encoded_key=None, chunk_size=_CHUNK_SIZE):
        self._key = self._decode_key(encoded_key)
        self.chunk_size = chunk_size
        if not 0 < chunk_size <= 0xFFFFFFFF:
            raise ValueError('Media encryption chunk size must be a positive 32-bit integer.')

    @staticmethod
    def _decode_key(encoded_key):
        if not encoded_key:
            return None
        try:
            key = base64.b64decode(
                encoded_key.encode('ascii'),
                altchars=b'-_',
                validate=True,
            )
        except (binascii.Error, ValueError, UnicodeEncodeError) as exc:
            raise MediaEncryptionKeyError(
                'MEDIA_ENCRYPTION_KEY must be a URL-safe base64-encoded 32-byte key.'
            ) from exc
        if len(key) != 32:
            raise MediaEncryptionKeyError(
                'MEDIA_ENCRYPTION_KEY must decode to exactly 32 bytes.'
            )
        return key

    @classmethod
    def from_environment(cls):
        return cls(os.environ.get('MEDIA_ENCRYPTION_KEY'))

    @staticmethod
    def encrypted_path(path):
        return f'{path}.enc'

    @staticmethod
    def logical_path(path):
        return path[:-4] if path.lower().endswith('.enc') else path

    def _require_key(self):
        if self._key is None:
            raise MediaEncryptionKeyError(
                'Set MEDIA_ENCRYPTION_KEY before saving media. Generate a stable '
                '32-byte key and keep it outside the media volume.'
            )

    def ensure_writable(self):
        self._require_key()

    def resolve(self, path):
        encrypted = self.encrypted_path(path)
        if os.path.isfile(encrypted):
            return encrypted
        if os.path.isfile(path):
            return path
        return None

    def exists(self, path):
        return self.resolve(path) is not None

    def open(self, path):
        stored_path = self.resolve(path)
        if stored_path is None:
            raise FileNotFoundError(path)
        with open(stored_path, 'rb') as probe:
            magic = probe.read(len(_MAGIC))
        if magic == _MAGIC:
            if self._key is None:
                raise MediaEncryptionKeyError(
                    'MEDIA_ENCRYPTION_KEY is required to read encrypted media.'
                )
            return EncryptedMediaReader(stored_path, self._key)
        return open(stored_path, 'rb')

    def write_stream(self, source, path):
        self._require_key()
        destination = self.encrypted_path(path)
        source_name = getattr(source, 'name', None)
        source_is_destination = (
            isinstance(source_name, (str, bytes, os.PathLike))
            and os.path.normcase(os.path.abspath(source_name))
            == os.path.normcase(os.path.abspath(path))
        )
        directory = os.path.dirname(os.path.abspath(destination))
        os.makedirs(directory, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='wb',
                dir=directory,
                prefix='.media-',
                suffix='.tmp',
                delete=False,
            ) as output:
                temporary_path = output.name
                size = self._stream_size(source)
                if size > self.chunk_size * _HEADER_NONCE_INDEX:
                    raise MediaStorageError('Media source exceeds the encrypted format size limit.')
                salt = os.urandom(16)
                nonce_prefix = os.urandom(8)
                file_key = _derive_file_key(self._key, salt)
                header_fields = _HEADER_FIELDS.pack(
                    _MAGIC,
                    size,
                    self.chunk_size,
                    salt,
                    nonce_prefix,
                )
                header_nonce = nonce_prefix + struct.pack('>I', _HEADER_NONCE_INDEX)
                header_tag = AESGCM(file_key).encrypt(header_nonce, b'', header_fields)
                output.write(header_fields)
                output.write(header_tag)

                source.seek(0)
                chunk_index = 0
                remaining = size
                while remaining:
                    chunk_length = min(self.chunk_size, remaining)
                    chunk = bytearray()
                    while len(chunk) < chunk_length:
                        data = source.read(chunk_length - len(chunk))
                        if not data:
                            raise MediaStorageError(
                                'Source media ended before its reported size.'
                            )
                        chunk.extend(data)
                    nonce = nonce_prefix + struct.pack('>I', chunk_index)
                    aad = header_fields + struct.pack('>I', chunk_index)
                    output.write(AESGCM(file_key).encrypt(nonce, bytes(chunk), aad))
                    remaining -= chunk_length
                    chunk_index += 1
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, destination)
            if not source_is_destination:
                self._remove_plaintext(path)
            return destination
        except Exception:
            if temporary_path and os.path.exists(temporary_path):
                os.remove(temporary_path)
            raise

    @staticmethod
    def _stream_size(source):
        try:
            original_position = source.tell()
            source.seek(0, os.SEEK_END)
            size = source.tell()
            source.seek(original_position)
        except (AttributeError, OSError, ValueError) as exc:
            raise MediaStorageError(
                'Media source must be seekable for encrypted storage.'
            ) from exc
        if size < 0 or size > 0xFFFFFFFFFFFFFFFF:
            raise MediaStorageError('Media source size is not supported.')
        return size

    def write_bytes(self, data, path):
        return self.write_stream(io.BytesIO(data), path)

    def save_file(self, source_path, path):
        with open(source_path, 'rb') as source:
            destination = self.write_stream(source, path)
        if os.path.normcase(os.path.abspath(source_path)) == os.path.normcase(os.path.abspath(path)):
            self._remove_plaintext(path)
        return destination

    @staticmethod
    def _remove_plaintext(path):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise MediaStorageError(
                f'Encrypted media was saved, but legacy plaintext could not be removed: {path}'
            ) from exc

    def read_bytes(self, path):
        with self.open(path) as source:
            return source.read()

    def size(self, path):
        with self.open(path) as source:
            source.seek(0, os.SEEK_END)
            return source.tell()

    def delete(self, path):
        deleted = False
        for candidate in (self.encrypted_path(path), path):
            try:
                os.remove(candidate)
                deleted = True
            except FileNotFoundError:
                continue
        if not deleted:
            raise FileNotFoundError(path)
        return deleted

    def listdir(self, directory):
        names = {}
        try:
            entries = os.listdir(directory)
        except FileNotFoundError:
            return []
        for name in entries:
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            logical_name = self.logical_path(name)
            if logical_name not in names or name.endswith('.enc'):
                names[logical_name] = name
        return sorted(names)
