/* This file is part of MyPaint.
 * Copyright (C) 2018 by the MyPaint Development Team.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#include "floodfill.hpp"

#include "../blending.hpp"
#include <cmath>
#include <vector>
#include <exception>

SpectralColor::SpectralColor(PyObject* targ_col)
{
    Py_ssize_t num_chans;
    bool valid_targ_col =
        PySequence_Check(targ_col) &&
        (num_chans = PySequence_Size(targ_col)) == MYPAINT_NUM_CHANS;
    if (!valid_targ_col) throw std::invalid_argument("Invalid target color");

    for (Py_ssize_t i = 0; i < num_chans; ++i) {
        // Borrowed ref, don't decrement refcount
        PyObject* chan_val_ob = PySequence_Fast_GET_ITEM(targ_col, i);
        float chan_val = static_cast<float>(PyFloat_AsDouble(chan_val_ob));
        color_representation.channels[i] = chan_val;
    }
}

SpectralColor::SpectralColor(double r, double g, double b)
{
    rgb_to_spectral(r, g, b, color_representation.channels);
    color_representation.channels[MYPAINT_NUM_CHANS - 1] = 1.0;
}

Color SpectralColor::representation()
{
    return color_representation;
}

/*
  Returns a list [(start, end)] of segments corresponding
  to contiguous occurrences of "true" in the given boolean array.
  This is the representation that is passed between the Python
  and C++ parts of the fill algorithm, representing, along with a
  direction marker, sequences of fill seeds to be handled.
*/
static inline PyObject*
to_seeds(const bool edge[N])
{
    PyObject* seg_list = PyList_New(0);

    bool contiguous = false;
    int seg_start = 0;
    int seg_end = 0;

    for (int c = 0; c < N; ++c) {
        if (edge[c]) {
            if (!contiguous) { // a new segment begins
                seg_start = c;
                seg_end = c;
                contiguous = true;
            } else { // segment continues
                seg_end++;
            }
        } else if (contiguous) { // segment ends
            PyObject* segment = Py_BuildValue("ii", seg_start, seg_end);
            PyList_Append(seg_list, segment);
            Py_DECREF(segment);
#ifdef HEAVY_DEBUG
            assert(segment->ob_refcnt == 1);
#endif
            contiguous = false;
        }
    }
    if (contiguous) {
        PyObject* segment = Py_BuildValue("ii", seg_start, seg_end);
        PyList_Append(seg_list, segment);
        Py_DECREF(segment);
#ifdef HEAVY_DEBUG
        assert(segment->ob_refcnt == 1);
#endif
    }

    return seg_list;
}

/*
Distance metric for two colors w. spectral representations.
Call with straightened (linear + unassociated) color values
*/
static inline float
distance(Color& a, Color& b)
{
    float a1 = a.alpha();
    float a2 = b.alpha();
    float alpha_diff = fabs(a1 - a2);
    float alpha_avg = (a1 + a2) / 2.0f;
    float accum_spectral_diff = 0.0;
    for (int i = 0; i < MYPAINT_NUM_CHANS-1; ++i) {
        float chan_diff = fabs(a.channels[i] - b.channels[i]);
        accum_spectral_diff += chan_diff;
    }
    accum_spectral_diff /= SPECTRAL_WEIGHTS_SUM;
    float color_factor = alpha_avg * (1.0f - alpha_diff);
    float alpha_factor = 1.0f - color_factor;

    return color_factor * accum_spectral_diff + alpha_factor * alpha_diff;
}

Filler::Filler(SpectralColor& targ_col, double tol)
    : target_color(targ_col.representation()),
      target_color_straight(targ_col.representation().straightened()),
      tolerance(tol)
{
}

/*
  Evaluates the source pixel based on the target color and
  tolerance value, returning the alpha of the fill as it
  should be set for the corresponding destination pixel.

  A non-zero result indicates that the destination pixel
  should be filled.
*/
chan_t
Filler::pixel_fill_alpha(const Color& px)
{
    float dist;

    if (target_color.alpha() == 0.0 && px.alpha() == 0.0)
        return fix15_one;
    else if (tolerance == 0) {
        return fix15_one * (target_color == px);
    }

    if (target_color.alpha() == 0)
        dist = px.alpha();
    else {
        Color src_straight = px.straightened();
        dist = distance(target_color_straight, src_straight);
    }

    // Compare with adjustable tolerance of mismatches.
    dist = dist / tolerance;
    if (dist > 1.0) {
        return 0;
    } else {
        float aa = 1.0 - dist;
        if (aa < 0.5)
            return fix15_short_clamp(fix15_one * (aa * 2));
        else
            return fix15_one;
    }
}

/*
  Helper function for the fill algorithm, enqueues the source pixel (at (x,y))
  if the corresponding destination pixel is not already filled, the source
  pixel color is within the fill tolerance and the pixel is not part of
  a contiguous sequence (on the same row) already represented in the queue.

  Return value indicates:
  "sequence is no longer contiguous, must enqueue next eligible pixel"
*/
bool
Filler::check_enqueue(
    const int x, const int y, bool check, const Color& src_pixel,
    const chan_t& dst_pixel)
{
    if (dst_pixel != 0) return true;
    bool match = pixel_fill_alpha(src_pixel) > 0;
    if (match && check) {
        seed_queue.push(coord(x, y));
        return false;
    }
    return !match;
}

/*
  If the input tile is:

  Not uniform - returns Py_None
  If uniform, returns its alpha, if filled, as a Py_Int
*/
PyObject*
Filler::tile_uniformity(bool empty_tile, PyObject* src_arr)
{
    if (empty_tile) {
        chan_t alpha = pixel_fill_alpha(Color());
        return Py_BuildValue("i", alpha);
    }

    PixelBuffer<Color> src = PixelBuffer<Color>(src_arr);

    if (src.is_uniform()) {
        chan_t alpha = pixel_fill_alpha(src(0, 0));
        return Py_BuildValue("i", alpha);
    }

    Py_INCREF(Py_None);
    return Py_None;
}

void
Filler::queue_seeds(
    PyObject* seeds, PixelBuffer<Color>& src, PixelBuffer<chan_t> dst)
{
    Py_ssize_t num_seeds = PySequence_Size(seeds);
    for (Py_ssize_t i = 0; i < num_seeds; ++i) {
        PyObject* seed_tuple = PySequence_GetItem(seeds, i);
        int x;
        int y;
        PyArg_ParseTuple(seed_tuple, "ii", &x, &y);
        Py_DECREF(seed_tuple);
        if (!dst(x, y) && pixel_fill_alpha(src(x, y)) > 0) {
            seed_queue.push(coord(x, y));
        }
    }
}

/*
  Generate and queue pixel coordinates based on a list of ranges, a direction
  of origin indicating the edge the ranges apply to, and source and destination
  tiles to determine if the coordinates can be, or have already been, filled.

  The input coordinates are marked in a boolean array, used to (potentially)
  exclude input seeds from output seeds.
*/
void
Filler::queue_ranges(
    edge origin, PyObject* seeds, bool input_marks[N], PixelBuffer<Color>& src,
    PixelBuffer<chan_t>& dst)
{
#ifdef HEAVY_DEBUG
    assert(PySequence_Check(seeds));
#endif
    // The tile edge and direction determined by where the seed segments
    // originates, left-to-right or top-to-bottom
    int x_base = (origin == edges::east) * (N - 1);
    int y_base = (origin == edges::south) * (N - 1);
    int x_offs = (origin + 1) % 2;
    int y_offs = origin % 2;

    for (int i = 0; i < PySequence_Size(seeds); ++i) {

        int seg_start;
        int seg_end;
        PyObject* segment = PySequence_GetItem(seeds, i);

#ifdef HEAVY_DEBUG
        assert(PyTuple_CheckExact(segment));
        assert(PySequence_Size(segment) == 2);
#endif
        if (!PyArg_ParseTuple(segment, "ii", &seg_start, &seg_end)) {
            Py_DECREF(segment);
            continue;
        }
        Py_DECREF(segment);
#ifdef HEAVY_DEBUG
        assert(segment->ob_refcnt == 1);
#endif

        // Check all pixels in the segment, adding the first
        // of every contiguous section to the queue

        bool contiguous = false;

        for (int n = seg_start, x = x_base + x_offs * n,
                 y = y_base + y_offs * n;
             n <= seg_end; ++n, x += x_offs, y += y_offs) {

            input_marks[n] = true; // mark incoming segment to avoid reseeding

            if (!dst(x, y) && pixel_fill_alpha(src(x, y)) > 0) {
                if (!contiguous) {
                    seed_queue.push(coord(x, y));
                    contiguous = true;
                }
            } else {
                contiguous = false;
            }
        }
    }
}

/*
  Four-way fill algorithm using segments to represent
  sequences of input and output seeds.

  Parameters src_o and dst_o should be N x N numpy arrays
  of types rgba (chan_t[4]) and chan_t respectively.
  The fill produces alpha values with reference to src_o and
  writes those alpha values to dst_o.

  The seed coordinates are derived from a list of tuples denoting
  ranges, which along with the seed origin parameter are used to
  determine the in-tile pixel coordinates.

  The bounds defined by the min/max,x/y parameters limit the fill
  within the tile, if they are more constrained than (0, 0, N-1, N-1)
*/
PyObject*
Filler::fill(
    PyObject* src_o, PyObject* dst_o, PyObject* seeds, edge seed_origin,
    int min_x, int min_y, int max_x, int max_y)
{
    // Sanity checks (not really necessary)
    if (min_x > max_x || min_y > max_y) return Py_BuildValue("[()()()()]");
    if (min_x < 0) min_x = 0;
    if (min_y < 0) min_y = 0;
    if (max_x > (N - 1)) max_x = (N - 1);
    if (max_y > (N - 1)) max_y = (N - 1);

    PixelBuffer<Color> src(src_o);
    PixelBuffer<chan_t> dst(dst_o);

#ifdef HEAVY_DEBUG
    assert(PySequence_Check(seeds));
#endif

    // Store input seed positions to filter them out
    // prior to constructing the output seed segment lists
    bool input_seeds[N] = {0,};

    if (seed_origin == edges::none) { // Initial seeds, a list of coordinates
        queue_seeds(seeds, src, dst);
    } else {
        queue_ranges(seed_origin, seeds, input_seeds, src, dst);
    } // Seed queue populated

    // 0-initialized arrays used to mark points reached on
    // the tile boundaries to potentially create new seed segments.
    // Ordered left-to-right / top-to-bottom, for n/s, e/w respectively
    bool _n[N] = {0,}, _e[N] = {0,}, _s[N] = {0,}, _w[N] = {0,};
    bool* edge_marks[] = {_n, _e, _s, _w};

    // Fill loop
    while (!seed_queue.empty()) {

        int x0 = seed_queue.front().x;
        int y = seed_queue.front().y;

        seed_queue.pop();

        // skip if we're outside the bbox range
        if (y < min_y || y > max_y) continue;

        for (int i = 0; i < 2; ++i) {
            bool look_above = true;
            bool look_below = true;

            const int x_start =
                x0 + i; // include starting coordinate when moving left
            const int x_delta = i * 2 - 1; // first scan left, then right

            PixelRef<Color> src_px = src.get_pixel(x_start, y);
            PixelRef<chan_t> dst_px = dst.get_pixel(x_start, y);

            for (int x = x_start; x >= min_x && x <= max_x;
                 x += x_delta, src_px.move_x(x_delta), dst_px.move_x(x_delta)) {
                if (dst_px.read()) {
                    break;
                } // Pixel is already filled

                chan_t alpha = pixel_fill_alpha(src_px.read());

                if (alpha <= 0) // Colors are too different
                {
                    break;
                }

                dst_px.write(alpha); // Fill the pixel

                if (y > 0) {
                    look_above = check_enqueue( //check/enqueue above
                        x, y-1, look_above, src_px.above(), dst_px.above());
                } else {
                    _n[x] = true; // On northern edge
                }
                if (y < (N - 1)) {
                    look_below = check_enqueue( // check/enqueue below
                        x, y+1, look_below, src_px.below(), dst_px.below());
                } else {
                    _s[x] = true; // On southern edge
                }

                if (x == 0) {
                    _w[y] = true; // On western edge
                } else if (x == (N - 1)) {
                    _e[y] = true; // On eastern edge
                }
            }
        }
    }

    if (seed_origin != edges::none) {
        // Remove incoming seeds from outgoing seeds
        bool* edge = edge_marks[seed_origin];
        for (int n = 0; n < N; ++n) {
            edge[n] = edge[n] && !input_seeds[n];
        }
    }

    return Py_BuildValue(
        "[NNNN]", to_seeds(_n), to_seeds(_e), to_seeds(_s), to_seeds(_w));
}

void
Filler::flood(PyObject* src_arr, PyObject* dst_arr)
{
    PixelRef<Color> src_px = PixelBuffer<Color>(src_arr).get_pixel(0, 0);
    PixelRef<chan_t> dst_px = PixelBuffer<chan_t>(dst_arr).get_pixel(0, 0);
    for (int i = 0; i < N * N; ++i, src_px.move_x(1), dst_px.move_x(1)) {
        dst_px.write(pixel_fill_alpha(src_px.read()));
    }
}

PyObject*
rgba_tile_from_alpha_tile(
    PyObject* src, SpectralColor& color,
    int min_x, int min_y, int max_x, int max_y)
{
    npy_intp dims[] = {N, N, MYPAINT_NUM_CHANS};
    // A zeroed array is used instead of an empty, since
    // less than the entire output tile may be written to.
    PyObject* dst_arr = PyArray_ZEROS(3, dims, NPY_FLOAT, 0);
    PixelBuffer<Color> dst_buf(dst_arr);
    PixelBuffer<chan_t> src_buf(src);
    for (int y = min_y; y <= max_y; ++y) {
        int x = min_x;
        PixelRef<chan_t> src_px = src_buf.get_pixel(x, y);
        PixelRef<Color> dst_px = dst_buf.get_pixel(x, y);
        for (; x <= max_x; ++x, src_px.move_x(1), dst_px.move_x(1)) {
            fix15_t alpha = src_px.read();
            if (alpha != 0){
                float alpha_f = static_cast<float>(src_px.read()) / fix15_one;
                // Copy-construction
                Color out_col = color.representation();
                out_col.set_alpha(alpha_f);
                if (alpha < fix15_one)
                    out_col.premultiply();
                dst_px.write(out_col);
            }
        }
    }
    return dst_arr;
}
