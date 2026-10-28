import base64
import os
import tempfile
import unittest

from flask import Flask, request, send_file

from media_storage import MediaEncryptionKeyError, MediaStorage, MediaStorageError


def _encoded_test_key():
    return base64.urlsafe_b64encode(bytes(range(32))).decode('ascii')


class MediaStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.storage = MediaStorage(_encoded_test_key(), chunk_size=8)
        self.logical_path = os.path.join(self.temporary_directory.name, 'photo.jpg')

    def test_writes_encrypted_file_and_reads_it_with_seeking(self):
        payload = b'legacy compatibility and chunked encryption'
        encrypted_path = self.storage.write_bytes(payload, self.logical_path)

        self.assertEqual(encrypted_path, self.logical_path + '.enc')
        with open(encrypted_path, 'rb') as stored_file:
            self.assertNotEqual(stored_file.read(), payload)
        self.assertFalse(os.path.exists(self.logical_path))
        self.assertTrue(self.storage.exists(self.logical_path))
        self.assertEqual(self.storage.read_bytes(self.logical_path), payload)

        with self.storage.open(self.logical_path) as media_file:
            media_file.seek(7)
            self.assertEqual(media_file.read(15), payload[7:22])
            media_file.seek(-5, os.SEEK_END)
            self.assertEqual(media_file.read(), payload[-5:])

    def test_reads_legacy_plaintext_and_prefers_encrypted_copy(self):
        with open(self.logical_path, 'wb') as legacy_file:
            legacy_file.write(b'legacy media')
        self.assertEqual(self.storage.read_bytes(self.logical_path), b'legacy media')

        self.storage.write_bytes(b'new encrypted media', self.logical_path)
        self.assertFalse(os.path.exists(self.logical_path))
        self.assertEqual(self.storage.read_bytes(self.logical_path), b'new encrypted media')

    def test_save_file_can_encrypt_a_legacy_file_in_place(self):
        payload = b'legacy file to encrypt'
        with open(self.logical_path, 'wb') as legacy_file:
            legacy_file.write(payload)

        encrypted_path = self.storage.save_file(self.logical_path, self.logical_path)

        self.assertEqual(encrypted_path, self.logical_path + '.enc')
        self.assertFalse(os.path.exists(self.logical_path))
        self.assertEqual(self.storage.read_bytes(self.logical_path), payload)

    def test_rejects_modified_encrypted_chunks(self):
        encrypted_path = self.storage.write_bytes(b'protected media content', self.logical_path)
        with open(encrypted_path, 'r+b') as stored_file:
            stored_file.seek(48)
            current = stored_file.read(1)
            stored_file.seek(48)
            stored_file.write(bytes([current[0] ^ 1]))

        with self.assertRaises(MediaStorageError):
            self.storage.read_bytes(self.logical_path)

    def test_empty_file_is_authenticated_and_readable(self):
        self.storage.write_bytes(b'', self.logical_path)
        self.assertEqual(self.storage.read_bytes(self.logical_path), b'')

    def test_encrypted_writes_require_a_configured_key(self):
        storage = MediaStorage()
        with self.assertRaises(MediaEncryptionKeyError):
            storage.write_bytes(b'media', self.logical_path)

    def test_delete_removes_encrypted_and_legacy_files(self):
        self.storage.write_bytes(b'encrypted', self.logical_path)
        with open(self.logical_path, 'wb') as legacy_file:
            legacy_file.write(b'legacy')

        self.storage.delete(self.logical_path)

        self.assertFalse(os.path.exists(self.logical_path))
        self.assertFalse(os.path.exists(self.logical_path + '.enc'))

    def test_listdir_exposes_logical_names_for_both_formats(self):
        self.storage.write_bytes(b'encrypted', self.logical_path)
        with open(os.path.join(self.temporary_directory.name, 'legacy.png'), 'wb') as legacy_file:
            legacy_file.write(b'legacy')

        self.assertEqual(
            self.storage.listdir(self.temporary_directory.name),
            ['legacy.png', 'photo.jpg'],
        )

    def test_seekable_encrypted_reader_supports_http_range_responses(self):
        payload = b'video content with byte-range playback'
        self.storage.write_bytes(payload, self.logical_path)
        app = Flask(__name__)

        @app.route('/media')
        def serve_media():
            response = send_file(
                self.storage.open(self.logical_path),
                download_name='video.webm',
                conditional=False,
                etag=False,
            )
            response.make_conditional(
                request.environ,
                accept_ranges=True,
                complete_length=self.storage.size(self.logical_path),
            )
            return response

        response = app.test_client().get('/media', headers={'Range': 'bytes=6-12'})

        try:
            self.assertEqual(response.status_code, 206)
            self.assertEqual(response.data, payload[6:13])
            self.assertEqual(response.headers['Content-Range'], f'bytes 6-12/{len(payload)}')
        finally:
            response.close()


if __name__ == '__main__':
    unittest.main()
