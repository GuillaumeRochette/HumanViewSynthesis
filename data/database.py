from typing import Union
from pathlib import Path
import io
import lmdb
import pickle

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Database(object):
    def __init__(
        self,
        path: Union[str, Path],
        readahead: bool = False,
        pre_open: bool = False,
    ):
        """
        Base class for LMDB-backed databases.

        :param path: Path to the database.
        :param readahead: Enables the filesystem readahead mechanism.
        :param pre_open: If set to True, the first iterations will be faster, but it will raise error when doing multi-gpu training. If set to False, the database will open when you will retrieve the first item.
        :param protocol: Defines the pickle protocol used when dumping keys, e.g. in _convert_key(). This is useful when reading a database that was made with an older python version.
        """
        if not isinstance(path, str):
            path = str(path)

        self.path = path
        self.readahead = readahead

        self.db = None
        self._open()
        self.protocol = self._protocol()
        self.keys = self._keys()
        if not pre_open:
            self._close()

    def _open(self):
        if self.db is None:
            self.db = lmdb.open(
                path=self.path,
                readonly=True,
                readahead=self.readahead,
                max_spare_txns=256,
                lock=False,
            )

    def _close(self):
        if self.db is not None:
            self.db.close()
        self.db = None

    def _protocol(self):
        """
        Read the pickle protocol contained in the database.

        :return: The set of available keys.
        """
        with self.db.begin() as txn:
            key = "protocol".encode("ascii")
            value = txn.get(key=key)
            value = pickle.loads(value)
        return value

    def _keys(self):
        """
        Read the keys contained in the database.

        :return: The set of available keys.
        """
        with self.db.begin() as txn:
            key = pickle.dumps("keys", protocol=self.protocol)
            value = txn.get(key=key)
            value = pickle.loads(value)
        return sorted(value)

    def __len__(self):
        """
        Returns the number of keys available in the database.

        :return: The number of keys.
        """
        return len(self.keys)

    def __getitem__(self, item):
        """
        Retrieves a value or a list of values from the database depending on whether item is a key or a list of keys.

        :param item: A key or a list of keys.
        :return: A value or a list of values.
        """
        self._open()
        if not isinstance(item, list):
            return self._get(key=item)
        else:
            return self._gets(keys=item)

    def _get(self, key):
        """
        Retrieve a value of a single key.

        :param key: A key.
        :return: A value.
        """
        with self.db.begin() as txn:
            key = self._convert_key(key=key)
            value = txn.get(key=key)
            value = self._convert_value(value=value)
        return value

    def _gets(self, keys):
        """
        Retrieve a list of values given a list of keys.

        :param keys: A list of keys.
        :return: A list of values.
        """
        with self.db.begin() as txn:
            values = []
            for key in keys:
                key = self._convert_key(key=key)
                value = txn.get(key=key)
                value = self._convert_value(value=value)
                values.append(value)
        return values

    def _convert_key(self, key):
        """
        Converts a key into a byte key.

        :param key: A key.
        :return: A byte key.
        """
        return pickle.dumps(key, protocol=self.protocol)

    def _convert_value(self, value):
        """
        Converts a byte value back into a value.

        :param value: A byte value.
        :return: A value
        """
        return pickle.loads(value)

    def __iter__(self):
        """
        Provides an iterator over the keys when iterating over the database.

        :return: An iterator on the keys.
        """
        return iter(self.keys)

    def __del__(self):
        """
        Closes the database properly.
        """
        self._close()


class ImageDatabase(Database):
    def __init__(
        self,
        path: Union[str, Path],
        readahead: bool = False,
        pre_open: bool = False,
    ):
        """
        Class for image databases (usually doesn't fit in RAM hence the readahead=False).

        :param path: Path to the database.
        """
        super(ImageDatabase, self).__init__(
            path=path,
            readahead=readahead,
            pre_open=pre_open,
        )

    def _convert_value(self, value):
        """
        Converts a byte image back into a PIL Image.

        :param value: A byte image.
        :return: A PIL Image image.
        """
        return Image.open(io.BytesIO(value))


class MaskDatabase(ImageDatabase):
    def _convert_value(self, value):
        """
        Converts a byte image back into a PIL Image.

        :param value: A byte image.
        :return: A PIL Image image.
        """
        return Image.open(io.BytesIO(value)).convert("1")


class LabelDatabase(Database):
    def __init__(
        self,
        path: Union[str, Path],
        readahead: bool = True,
        pre_open: bool = False,
    ):
        """
        Class for label databases (usually fits in RAM hence the readahead=True).

        :param path: Path to the database.
        """
        super(LabelDatabase, self).__init__(
            path=path,
            readahead=readahead,
            pre_open=pre_open,
        )
