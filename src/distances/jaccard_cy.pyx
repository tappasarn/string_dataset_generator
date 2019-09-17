cpdef float jaccard_seq(list x, list y):
    cdef int len_x, len_y
    cdef list fst, snd
    cdef float num_intersect
    
    len_x = len(x)
    len_y = len(y)

    fst, snd = (x, y) if len_x < len_y else (y, x)
    num_intersect = len(set(fst).intersection(snd))
    
    return 1 - (num_intersect / (len_x + len_y - num_intersect))
