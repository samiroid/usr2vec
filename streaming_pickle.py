"""Streaming pickle implementation for efficiently serializing and
de-serializing an iterable (e.g., list)

Created on 2010-06-19 by Philip Guo
  
http://code.google.com/p/streaming-pickle/

Modified by Brian Thorne 2013 to add base64 encoding to support
python3 bytearray and the like.
"""
import base64
from cPickle import dumps, loads
import unittest
import tempfile

def s_dump(iterable_to_pickle, file_obj):
    """dump contents of an iterable iterable_to_pickle to file_obj, a file
    opened in write mode"""
    for elt in iterable_to_pickle:
        s_dump_elt(elt, file_obj)


def s_dump_elt(elt_to_pickle, file_obj):
    """dumps one element to file_obj, a file opened in write mode"""
    pickled_elt = dumps(elt_to_pickle,-1)
    encoded = base64.b64encode(pickled_elt)
    file_obj.write(encoded)

    # record separator is a blank line
    # (since pickled_elt as base64 encoded cannot contain its own newlines)
    file_obj.write(b'\n\n')


def s_load(file_obj):
    """load contents from file_obj, returning a generator that yields one
    element at a time"""
    cur_elt = []
    for line in file_obj:

        if line == b'\n':
            encoded_elt = b''.join(cur_elt)
            try:
                pickled_elt = base64.b64decode(encoded_elt)
                elt = loads(pickled_elt)
            except EOFError:
                print("EOF found while unpickling data")
                print(pickled_elt)
                raise StopIteration
            cur_elt = []
            yield elt
        else:
            cur_elt.append(line)


class TestStreamingPickle(unittest.TestCase):
    def setUp(self):
        pass

    def testSimpleList(self):
        # data = [1, 2, 3, 4, None,  b'test', '\n', '\x00', 3, b'\n\n\n\n', 5, 7, 9, 11, "hello", bytearray([2, 4, 4])]
        data = [1,[1,2,3,4],[8,9,29]]
        with tempfile.TemporaryFile() as f:
            s_dump(data, f)
            # reset the temporary file
            f.seek(0)
            i = 0
            for i, element in enumerate(s_load(f)):
                self.assertEqual(data[i], element)
                # print(i, element)
            self.assertEqual(i, len(data)-1)

if __name__ == "__main__":
    unittest.main()
