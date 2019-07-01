/* This file is part of MyPaint.
 * Copyright (C) 2014 by Andrew Chadwick <a.t.chadwick@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

// Template functions for compositing buffers

#ifndef __HAVE_COMPOSITING
#define __HAVE_COMPOSITING

#include "fix15.hpp"
#include <mypaint-tiled-surface.h>

#include <glib.h>

// Abstract interface for TileDataCombine<> blend mode functors
//
// Blend functors are low-level pixel operations. Derived classes' operator()
// implementations should be declared inline and in the class body.
//
// These functors that apply a source colour to qa destination, with no
// metadata. The R, G and B values are not premultiplied by alpha during the
// blending phase.

class BlendFunc
{
  public:
    virtual void operator() (const float * const src,
                              float * dst,
                              const float * const opts) const = 0;
};


// Abstract interface for TileDataCombine<> compositing op functors
//
// Compositing functors are low-level pixel operations. Derived classes'
// operator() implementations should be declared inline and in the class body.
//
// These are primarily stateless functors which apply a source colour and pixel
// alpha to a destination. At this phase in the rendering workflow, the input
// R, G, and B values are not muliplied by their corresponding A, but the
// output pixel's R, G and B values are multiplied by alpha, and must be
// written as such.
//
// Implementations must also supply details which allow C++ pixel-level
// operations and Python tile-level operations to optimize away blank data or
// skip the dst_has_alpha speedup when necessary.

class CompositeFunc
{
  public:
    virtual void operator() (float * const src,
                            float * dst) const = 0;
    static const bool zero_alpha_has_effect = true;
    static const bool can_decrease_alpha = true;
    static const bool zero_alpha_clears_backdrop = true;
};


// Composable blend+composite functor for buffers
//
// The template parameters define whether the destination's alpha is used,
// and supply the BlendFunc and CompositeFunc functor classes to use.  The
// size of the buffers to be processed must also be specified.
//
// This is templated at the class level so that more optimal partial template
// specializations can be written for more common code paths. The C++ spec
// does not permit plain functions to be partially specialized.
//
// Ref: http://www.w3.org/TR/compositing-1/#generalformula

template <bool DSTALPHA,
          unsigned int BUFSIZE,
          class BLENDFUNC,
          class COMPOSITEFUNC>
class BufferCombineFunc
{
  private:
    BLENDFUNC blendfunc;
    COMPOSITEFUNC compositefunc;

  public:
    inline void operator() (const float * const src,
                            float * const dst,
                            const float src_opacity,
                            const float * const opts) const
    {
#ifndef HEAVY_DEBUG
        // Skip tile if it can't affect the backdrop
        const bool skip_empty_src = ! compositefunc.zero_alpha_has_effect;
        if (skip_empty_src && src_opacity == 0) {
            return;
        }
#endif

        // Pixel loop
        //float Rs,Gs,Bs,as, Rb,Gb,Bb,ab, one_minus_ab;
        float Psrc[MYPAINT_NUM_CHANS];
        float Pdst[MYPAINT_NUM_CHANS];
        float one_minus_ab;
//#pragma omp parallel for private(Psrc, Pdst, one_minus_ab)
        for (unsigned int i = 0; i < BUFSIZE; i += MYPAINT_NUM_CHANS)
        {
            // Calculate unpremultiplied source RGB values
            Psrc[MYPAINT_NUM_CHANS-1] = src[i+MYPAINT_NUM_CHANS-1];
            if (Psrc[MYPAINT_NUM_CHANS-1] == 0) {
#ifndef HEAVY_DEBUG
                // Skip pixel if it can't affect the backdrop pixel
                if (skip_empty_src) {
                    continue;
                }
#endif
            }
            else {
                for (int p=0; i<MYPAINT_NUM_CHANS-1; p++) {
                    Psrc[p] = (float_div(src[i+p], Psrc[MYPAINT_NUM_CHANS-1]));
                }
            }

            // Calculate unpremultiplied backdrop RGB values
            if (DSTALPHA) {
                Pdst[MYPAINT_NUM_CHANS-1] = dst[i+MYPAINT_NUM_CHANS-1];
                if (Pdst[MYPAINT_NUM_CHANS-1] == 0) {
                    //Rb = Gb = Bb = 0;
                }
                else {
                    for (int p=0; i<MYPAINT_NUM_CHANS-1; p++) {
                      Pdst[p] = (float_div(dst[i+p], Pdst[MYPAINT_NUM_CHANS-1]));
                    }
                }
            }
            else {
                Pdst[MYPAINT_NUM_CHANS-1] = 1.0;
                for (int p=0; i<MYPAINT_NUM_CHANS-1; p++) {
                  Pdst[p] = dst[i+p];
                }
            }

            // Apply the colour blend functor
            blendfunc(Psrc, Pdst, opts);

            // Apply results of the blend in place
            if (DSTALPHA) {
                one_minus_ab = 1.0 - Pdst[MYPAINT_NUM_CHANS-1];
                for (int p=0; i<MYPAINT_NUM_CHANS-1; p++) {
                  Pdst[p] = float_sumprods(one_minus_ab, Psrc[p], Pdst[MYPAINT_NUM_CHANS-1], Pdst[p]);
                }
            }
            // Use the blend result as a source, and composite directly into
            // the destination buffer as premultiplied RGB.
            Pdst[MYPAINT_NUM_CHANS-1] = float_mul(Psrc[MYPAINT_NUM_CHANS-1], src_opacity);
            compositefunc(Pdst, dst);
        }
    }
};


// Abstract interface for tile-sized BufferCombineFunc<>s
//
// This is the interface the Python-facing code uses, one per supported
// tiledsurface (layer) combine mode. Implementations are intended to be
// templated things exposing their CompositeFunc's flags via the
// abstract methods defined in this interface.

class TileDataCombineOp
{
  public:
    virtual void combine_data (const float *src_p,
                               float *dst_p,
                               const bool dst_has_alpha,
                               const float src_opacity,
                               const float *opts) const = 0;
    virtual const char* get_name() const = 0;
    virtual bool zero_alpha_has_effect() const = 0;
    virtual bool can_decrease_alpha() const = 0;
    virtual bool zero_alpha_clears_backdrop() const = 0;
};


// Source Over: place the source over the destination. This implements the
// conventional "basic alpha blending" compositing mode.
// http://www.w3.org/TR/compositing-1/#porterduffcompositingoperators_srcover

class CompositeSourceOver : public CompositeFunc
{
  public:
    inline void operator() (float * const src,
                            float * dst) const
    {
        const float j = 1.0 - src[MYPAINT_NUM_CHANS-1];
        const float k = float_mul(dst[MYPAINT_NUM_CHANS-1], j);
        for (int i=0; i<MYPAINT_NUM_CHANS-1; i++) {
            dst[i] = (float_sumprods(src[MYPAINT_NUM_CHANS-1], src[i], j, dst[i]));
        }
        dst[MYPAINT_NUM_CHANS-1] = (src[MYPAINT_NUM_CHANS-1] + k);
    }

    static const bool zero_alpha_has_effect = false;
    static const bool can_decrease_alpha = false;
    static const bool zero_alpha_clears_backdrop = false;
};

// Source Over Spectral WGM:  place the source over the destination
// Similar to paint.  Use weighted geometric mean, upsample to 10 channels
// must use un-premultiplied color and alpha ratios normalized to sum to 1.0

class CompositeSpectralWGM : public CompositeFunc
{
  public:
    inline void operator() (float * const src,
                            float * dst) const
    {
        // pseudo code example:
        // ratio = as / as + (1 - as) * ab;
        // rgb = pow(rgb, ratio) * pow(rgb, (1-ratio));
        // ab = (as + k);
        // rgb = rgb * ab;
    }

    static const bool zero_alpha_has_effect = false;
    static const bool can_decrease_alpha = false;
    static const bool zero_alpha_clears_backdrop = false;
};

class CompositeBumpMap : public CompositeFunc
{
  public:
    inline void operator() (float * const src,
                            float * dst) const
    {
//        const float j = 1.0 - as;
//        const float k = float_mul(ab, j);

//        rb = (float_sumprods(as, Rs, j, rb));
//        gb = (float_sumprods(as, Gs, j, gb));
//        bb = (float_sumprods(as, Bs, j, bb));
//        ab = (as + k);
    }

    static const bool zero_alpha_has_effect = true;
    static const bool can_decrease_alpha = false;
    static const bool zero_alpha_clears_backdrop = false;
};

class CompositeBumpMapDst : public CompositeFunc
{
  public:
    inline void operator() (float * const src,
                            float * dst) const
    {
//        const float j = 1.0 - as;
//        const float k = float_mul(ab, j);

//        rb = (float_sumprods(as, Rs, j, rb));
//        gb = (float_sumprods(as, Gs, j, gb));
//        bb = (float_sumprods(as, Bs, j, bb));
//        ab = (as + k);
    }

    static const bool zero_alpha_has_effect = true;
    static const bool can_decrease_alpha = false;
    static const bool zero_alpha_clears_backdrop = false;
};


// Destination-In: the painted areas make stencil voids. The backdrop shows
// through only within the painted areas of the source.
// http://www.w3.org/TR/compositing-1/#compositingoperators_dstin

class CompositeDestinationIn : public CompositeFunc
{
  public:
    inline void operator() (float * const src,
                            float * dst) const
    {
//        rb = (float_mul(rb, as));
//        gb = (float_mul(gb, as));
//        bb = (float_mul(bb, as));
//        ab = (float_mul(ab, as));
    }

    static const bool zero_alpha_has_effect = true;
    static const bool can_decrease_alpha = true;
    static const bool zero_alpha_clears_backdrop = true;
};


// Destination-Out: the painted areas work a little like masking fluid or tape,
// or wax resist. The backdrop shows through only outside painted source areas.
// http://www.w3.org/TR/compositing-1/#porterduffcompositingoperators_dstout

class CompositeDestinationOut : public CompositeFunc
{
  public:
    inline void operator() (float * const src,
                            float * dst) const
    {
//        const float j = 1.0 - as;
//        rb = (float_mul(rb, j));
//        gb = (float_mul(gb, j));
//        bb = (float_mul(bb, j));
//        ab = (float_mul(ab, j));
    }

    static const bool zero_alpha_has_effect = false;
    static const bool can_decrease_alpha = true;
    static const bool zero_alpha_clears_backdrop = false;
};


// Source-Atop: Source which overlaps the destination, replaces the destination.
// Destination is placed elsewhere.
// http://www.w3.org/TR/compositing-1/#porterduffcompositingoperators_srcatop

class CompositeSourceAtop : public CompositeFunc
{
  public:
    inline void operator() (float * const src,
                            float * dst) const
    {
        // W3C spec:
        //   co = as*Cs*ab + ab*Cb*(1-as)
        // where
        //   Cs ∈ {Rs, Gs, Bs}         -- input is non-premultiplied
        //   cb ∈ {rb gb, bb} = ab*Cb  -- output is premultiplied by alpha
//        const float one_minus_as = 1.0 - as;
//        const float ab_mul_as = float_mul(as, ab);
//        rb = (float_sumprods(ab_mul_as, Rs, one_minus_as, rb));
//        gb = (float_sumprods(ab_mul_as, Gs, one_minus_as, gb));
//        bb = (float_sumprods(ab_mul_as, Bs, one_minus_as, bb));
        // W3C spec:
        //   ao = as*ab + ab*(1-as)
        //   ao = ab
        // (leave output alpha unchanged)
    }

    static const bool zero_alpha_has_effect = false;
    static const bool can_decrease_alpha = false;
    static const bool zero_alpha_clears_backdrop = false;
};


// Destination-Atop: Destination which overlaps the source replaces the source.
// Source is placed elsewhere.
// http://www.w3.org/TR/compositing-1/#porterduffcompositingoperators_dstatop

class CompositeDestinationAtop : public CompositeFunc
{
  public:
    inline void operator() (float * const src,
                            float * dst) const
    {
        // W3C spec:
        //   co = as*Cs*(1-ab) + ab*Cb*as
        // where
        //   Cs ∈ {Rs, Gs, Bs}         -- input is non-premultiplied
        //   cb ∈ {rb gb, bb} = ab*Cb  -- output is premultiplied by alpha
//        const float one_minus_ab = 1.0 - ab;
//        const float as_mul_one_minus_ab = float_mul(as, one_minus_ab);
//        rb = (float_sumprods(as_mul_one_minus_ab, Rs, as, rb));
//        gb = (float_sumprods(as_mul_one_minus_ab, Gs, as, gb));
//        bb = (float_sumprods(as_mul_one_minus_ab, Bs, as, bb));
//        // W3C spec:
//        //   ao = as*(1-ab) + ab*as
//        //   ao = as
//        ab = as;
    }

    static const bool zero_alpha_has_effect = true;
    static const bool can_decrease_alpha = true;
    static const bool zero_alpha_clears_backdrop = true;
};


// W3C "Lighter", a.k.a. Porter-Duff "plus", a.k.a. "svg:plus". This just adds
// together corresponding channels of the destination and source.
// Ref: http://www.w3.org/TR/compositing-1/#porterduffcompositingoperators_plus

class CompositeLighter : public CompositeFunc
{
  public:
    inline void operator() (float * const src,
                            float * dst) const
    {
//        rb = (float_mul(Rs, as) + rb);
//        gb = (float_mul(Gs, as) + gb);
//        bb = (float_mul(Bs, as) + bb);
//        ab = (ab + as);
    }

    static const bool zero_alpha_has_effect = false;
    static const bool can_decrease_alpha = false;
    static const bool zero_alpha_clears_backdrop = false;
};


#endif //__HAVE_COMPOSITING
