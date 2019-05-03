#!/usr/bin/python

import sys
import struct
from itertools import islice

with open('/mnt/vivek-data2.csv', 'r') as inf:
    with open('/mnt/uie-sparse.dat', 'wb') as ouf:
        try:
            counter = 0
            while True:
                data =  islice(inf, 0, 3)
                le = int(next(data))
                idx = [int(i) for i in next(data)]
                data= [float(i) for i in next(data)]
                ouf.write(struct.pack('i', le))
                ouf.write(struct.pack('%si' % le , *idx ))
                ouf.write(struct.pack('%sf' % le,*data ))
                counter+=3
                if counter % 10000 == 0:
                    sys.stdout.write('%d points processed...\n' % counter)
        except StopIteration:
            pass