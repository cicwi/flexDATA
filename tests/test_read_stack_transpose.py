from flexdata import data
import numpy as np


def test_read_stack_transpose():
    path = '/ufs/ciacc/flexbox/skull/'
    proj = data.read_stack(path, 'scan_', skip=4, sample=4, updown=False,
                           transpose=[0, 1, 2])

    proj_transpose = data.read_stack(path, 'scan_', skip=4, sample=4,
                                     updown=False, transpose=[1, 0, 2])
    assert np.array_equal(proj_transpose.transpose([1, 0, 2]), proj)

    proj_transpose = data.read_stack(path, 'scan_', skip=4, sample=4,
                                     updown=False, transpose=[0, 2, 1])
    assert np.array_equal(proj_transpose.transpose([0, 2, 1]), proj)

    proj_transpose = data.read_stack(path, 'scan_', skip=4, sample=4,
                                     updown=False, transpose=[2, 1, 0])
    assert np.array_equal(proj_transpose.transpose([2, 1, 0]), proj)
