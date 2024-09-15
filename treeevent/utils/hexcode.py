import math
import collections

import numpy as np

Point = collections.namedtuple("Point", ["x", "y"])

_Hex = collections.namedtuple("Hex", ["q", "r", "s"])


def Hex(q, r, s):
    assert not all(round(q + r + s) != 0), "q + r + s must be 0"
    return _Hex(q, r, s)


Orientation = collections.namedtuple(
    "Orientation", ["f0", "f1", "f2", "f3", "b0", "b1", "b2", "b3", "start_angle"]
)

Layout = collections.namedtuple("Layout", ["orientation", "size", "origin"])

layout_pointy = Orientation(
    math.sqrt(3.0),
    math.sqrt(3.0) / 2.0,
    0.0,
    3.0 / 2.0,
    math.sqrt(3.0) / 3.0,
    -1.0 / 3.0,
    0.0,
    2.0 / 3.0,
    0.5,
)
layout_flat = Orientation(
    3.0 / 2.0,
    0.0,
    math.sqrt(3.0) / 2.0,
    math.sqrt(3.0),
    2.0 / 3.0,
    0.0,
    -1.0 / 3.0,
    math.sqrt(3.0) / 3.0,
    0.0,
)


def hex_round(h):
    qi = (round(h.q)).astype(int)
    ri = (round(h.r)).astype(int)
    si = (round(h.s)).astype(int)
    q_diff = abs(qi - h.q)
    r_diff = abs(ri - h.r)
    s_diff = abs(si - h.s)

    cond1 = (q_diff > r_diff) & (q_diff > s_diff)
    cond2 = (~cond1) & (r_diff > s_diff)
    cond3 = ~(cond1 & (r_diff > s_diff))

    qi[cond1] = -ri[cond1] - si[cond1]
    ri[cond2] = -qi[cond2] - si[cond2]
    si[cond3] = -qi[cond3] - ri[cond3]

    return Hex(qi, ri, si)


def hex_to_pixel(layout, h):
    M = layout.orientation
    size = layout.size
    origin = layout.origin
    x = (M.f0 * h.q + M.f1 * h.r) * size.x
    y = (M.f2 * h.q + M.f3 * h.r) * size.y
    return Point(x + origin.x, y + origin.y)


def pixel_to_hex(layout, p):
    M = layout.orientation
    size = layout.size
    origin = layout.origin
    pt = Point((p.x - origin.x) / size.x, (p.y - origin.y) / size.y)
    q = M.b0 * pt.x + M.b1 * pt.y
    r = M.b2 * pt.x + M.b3 * pt.y
    return Hex(q, r, -q - r)


def equal_hex(a, b):
    if not (a.q == b.q and a.s == b.s and a.r == b.r):
        raise ValueError("Hex values are not equal")


def hexcodes(coords, sz_cell):
    coords = coords / sz_cell

    mn_X, mn_Y = np.min(coords, axis=0)
    mx_X, mx_Y = np.max(coords, axis=0)

    cX = int(mn_X + (mx_X - mn_X) / 2.0)
    cY = int(mn_Y + (mx_Y - mn_Y) / 2.0)

    cX = mn_X
    cY = mn_Y

    nX = int(mx_X - mn_X) + 1
    nY = int(mx_Y - mn_Y) + 1

    flat = Layout(layout_flat, Point(1, 1), Point(cX, cY))

    h = hex_round(pixel_to_hex(flat, Point(coords.x, coords.y)))

    return (h, (h.q + nX) * nY + (h.r + nY))
