# Adapted from
# http://nbviewer.ipython.org/github/Carreau/posts/blob/master/07-the-sound-of-hydrogen.ipynb

import struct
from io import BytesIO
import sys
import base64
import numpy as np


DEFAULT_RATE = 44100


def get_html5_wave_player(data, rate=DEFAULT_RATE):
    bio = BytesIO()
    bio.write(b'RIFF')
    bio.write(b'\x00\x00\x00\x00')
    bio.write(b'WAVE')

    # clip
    data = np.minimum(1, data)
    data = np.maximum(-1, data)

    data = data * np.iinfo(np.int16).max
    data = data.astype(np.int16)

    bio.write(b'fmt ')
    if data.ndim == 1:
        noc = 1
    else:
        noc = data.shape[1]

    bits = data.dtype.itemsize * 8
    sbytes = rate*(bits // 8)*noc
    ba = noc * (bits // 8)
    bio.write(struct.pack('<ihHIIHH', 16, 1, noc, rate, sbytes, ba, bits))

    # data chunk
    bio.write(b'data')
    bio.write(struct.pack('<i', data.nbytes))

    if (data.dtype.byteorder == '>'
            or (data.dtype.byteorder == '=' and sys.byteorder == 'big')):
        data = data.byteswap()

    bio.write(data.tostring())

    # Determine file size and place it at the correct position at start of the
    # file.

    size = bio.tell()
    bio.seek(4)
    bio.write(struct.pack('<i', size-8))

    val = bio.getvalue()

    src = """
        <audio controls="controls" style="width:600px" >
          <source
            controls src="data:audio/wav;base64,{base64}" type="audio/wav" />
          Your browser does not support the audio element.
        </audio>
        """.format(base64=base64.encodebytes(val).decode())

    from IPython.core.display import HTML
    return HTML(src)
