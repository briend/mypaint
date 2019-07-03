/* This file is part of MyPaint.
 * Copyright (C) 2012 by Andrew Chadwick <a.t.chadwick@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

// Generic blend mode functors, with partially specialized buffer compositors
// for some optimized cases.

#ifndef __HAVE_BLENDING
#define __HAVE_BLENDING
#define WGM_EPSILON 0.0001
#define NUM_WAVES 7

#include "fix15.hpp"
#include <mypaint-tiled-surface.h>
#include "fastapprox/fastpow.h"
#include "fastapprox/fasttrig.h"
#include "compositing.hpp"
#include <math.h>

static const float T_MATRIX_SMALL[3][NUM_WAVES] = {{0.004727862039458, 0.082644899379487, -0.322515894576622, -0.064320292139570,
1.064746457514018, 0.288869101686002, 0.010454417702711},
{-0.004081870492374, -0.101308479809214, 0.320514309815141, 0.720325047228787,
0.066431970334792, -0.028358642287937, -0.001135818542699},
{0.028683360043884, 1.054907349924059, 0.116111201474362, -0.084435897516297,
-0.029621508810678, -0.002318568718824, -0.000070180490104}};

static const float spectral_r_small[NUM_WAVES] = {.014976989831103, 0.015163469993149, 0.024828861915840, 0.055372724024590,
0.311175941451513, 2.261540004074889, 2.451861959778458};

static const float spectral_g_small[NUM_WAVES] = {0.060871084436057, 0.063645032450431, 0.344088900200936, 1.235198096662594,
0.145221682434442, 0.101106655125270, 0.099848117829856};

static const float spectral_b_small[NUM_WAVES] = {0.777465337464873, 0.899749264722067, 0.258544195013949, 0.015623896354842,
0.004846585772726, 0.003989003708280, 0.003962407615164};


void
rgb_to_spectral (float r, float g, float b, float *spectral_) {
  float offset = 1.0 - WGM_EPSILON;
  r = r * offset + WGM_EPSILON;
  g = g * offset + WGM_EPSILON;
  b = b * offset + WGM_EPSILON;
  //upsample rgb to spectral primaries
  float spec_r[NUM_WAVES] = {0};
  for (int i=0; i < NUM_WAVES; i++) {
    spec_r[i] = spectral_r_small[i] * r;
  }
  float spec_g[NUM_WAVES] = {0};
  for (int i=0; i < NUM_WAVES; i++) {
    spec_g[i] = spectral_g_small[i] * g;
  }
  float spec_b[NUM_WAVES] = {0};
  for (int i=0; i < NUM_WAVES; i++) {
    spec_b[i] = spectral_b_small[i] * b;
  }
  //collapse into one spd
  for (int i=0; i<NUM_WAVES; i++) {
    spectral_[i] += log2f(spec_r[i] + spec_g[i] + spec_b[i]);
  }

}

void
spectral_to_rgb (float *spectral, float *rgb_) {
  float offset = 1.0 - WGM_EPSILON;
  for (int i=0; i<NUM_WAVES; i++) {
    rgb_[0] += T_MATRIX_SMALL[0][i] * exp2f(spectral[i]);
    rgb_[1] += T_MATRIX_SMALL[1][i] * exp2f(spectral[i]);
    rgb_[2] += T_MATRIX_SMALL[2][i] * exp2f(spectral[i]);
  }
  for (int i=0; i<3; i++) {
    rgb_[i] = CLAMP((rgb_[i] - WGM_EPSILON) / offset, 0.0f, (1.0));
  }
}


// Normal: http://www.w3.org/TR/compositing/#blendingnormal

class BlendNormal : public BlendFunc
{
  public:
    inline void operator() (const float * const src,
                            float * dst,
                            const float * const opts) const
    {
//        dst_r = src_r;
//        dst_g = src_g;
//        dst_b = src_b;
    }
};

template <bool DSTALPHA, unsigned int BUFSIZE>
class BufferCombineFunc <DSTALPHA, BUFSIZE, BlendNormal, CompositeSourceOver>
{
    // Partial specialization for normal painting layers (svg:src-over),
    // working in premultiplied alpha for speed.
  public:
    inline void operator() (const float * const src,
                            float * dst,
                            const float opac,
                            const float * const opts) const
    {
        for (unsigned int i=0; i<BUFSIZE; i+=MYPAINT_NUM_CHANS) {
            const float Sa = float_mul(src[i+MYPAINT_NUM_CHANS-1], opac);
            const float one_minus_Sa = 1.0 - Sa;
            for (int p=0; p<MYPAINT_NUM_CHANS-1; p++) {
                dst[i+p] = float_sumprods(src[i+p], opac, one_minus_Sa, dst[i+p]);
            }
            if (DSTALPHA) {
                dst[i+MYPAINT_NUM_CHANS-1] = (Sa + float_mul(dst[i+MYPAINT_NUM_CHANS-1], one_minus_Sa));
            }
        }
    }
};



template <bool DSTALPHA, unsigned int BUFSIZE>
class BufferCombineFunc <DSTALPHA, BUFSIZE, BlendNormal, CompositeBumpMap>
{
    // Apply bump map to SRC using itself.
  public:
    inline void operator() (const float * const src,
                            float * dst,
                            const float opac,
                            const float * const opts) const
    {
        const float Oren_rough = opts[0];
        const float Oren_A = 1.0 - 0.5 * (Oren_rough / (Oren_rough + 0.33));
        const float Oren_B = 0.45 * (Oren_rough / (Oren_rough + 0.09));
        const float Oren_exposure = 1.0 / Oren_A;
        const unsigned int stride = MYPAINT_TILE_SIZE * MYPAINT_NUM_CHANS;
        for (unsigned int i=0; i<BUFSIZE; i+=MYPAINT_NUM_CHANS) {
            // Calcuate bump map 
            // Use alpha as  height-map
            float slope = 0.0;
            const int reach = 1;
            float center = 0.0;
            for (int c=0; c<MYPAINT_NUM_CHANS; c++) {
              center += src[i+c];
            }
            for (int p=1; p<=reach; p++) {
                // North
                if (i >= stride * p) {
                    int o = i - stride * p;
                    float _slope = 0.0;
                    for (int c=0; c<MYPAINT_NUM_CHANS; c++) {
                      _slope += src[o+c];
                    }
                    slope += abs(_slope - center);
                } else {
                    int o = i + stride * p;
                    float _slope = 0.0;
                    for (int c=0; c<MYPAINT_NUM_CHANS; c++) {
                      _slope += src[o+c];
                    }
                    slope += abs(_slope - center);
                }
                // East
                if (i % stride < stride - MYPAINT_NUM_CHANS * p) {
                    int o = i + MYPAINT_NUM_CHANS * p;
                    float _slope = 0.0;
                    for (int c=0; c<MYPAINT_NUM_CHANS; c++) {
                      _slope += src[o+c];
                    }
                    slope += abs(_slope - center);
                } else {
                    int o = i - MYPAINT_NUM_CHANS * p;
                    float _slope = 0.0;
                    for (int c=0; c<MYPAINT_NUM_CHANS; c++) {
                      _slope += src[o+c];
                    }
                    slope += abs(_slope - center);
                }
                // West
                if (i % stride >= MYPAINT_NUM_CHANS * p) {
                    int o = i - MYPAINT_NUM_CHANS * p;
                    float _slope = 0.0;
                    for (int c=0; c<MYPAINT_NUM_CHANS; c++) {
                      _slope += src[o+c];
                    }
                    slope += abs(_slope - center);
                } else {
                    int o = i + MYPAINT_NUM_CHANS * p;
                    float _slope = 0.0;
                    for (int c=0; c<MYPAINT_NUM_CHANS; c++) {
                      _slope += src[o+c];
                    }
                    slope += abs(_slope - center);
                }
                // South
                if (i < BUFSIZE - stride * p) {
                    int o = i + stride * p;
                    float _slope = 0.0;
                    for (int c=0; c<MYPAINT_NUM_CHANS; c++) {
                      _slope += src[o+c];
                    }
                    slope += abs(_slope - center);
                } else {
                    int o = i - stride * p;
                    float _slope = 0.0;
                    for (int c=0; c<MYPAINT_NUM_CHANS; c++) {
                      _slope += src[o+c];
                    }
                    slope += abs(_slope - center);
                }
            }
            
            // amplify slope with options array
            slope = slope / fasterpow(2, opts[1]);
            float degrees = atan(slope);
            float lambert = (fastcos(degrees) * (Oren_A + (Oren_B * fastsin(degrees) * fasttan(degrees)))) * Oren_exposure;
            
            for (int c=0; c<MYPAINT_NUM_CHANS-1; c++) {
                dst[i+c] = (float_mul(dst[i+c], lambert));
            }
        }
    }
};

template <bool DSTALPHA, unsigned int BUFSIZE>
class BufferCombineFunc <DSTALPHA, BUFSIZE, BlendNormal, CompositeBumpMapDst>
{
    // apply SRC as bump map to DST.
  public:
    inline void operator() (const float * const src,
                            float * dst,
                            const float opac,
                            const float * const opts) const
    {
        const float Oren_rough = opts[0];
        const float Oren_A = 1.0 - 0.5 * (Oren_rough / (Oren_rough + 0.33));
        const float Oren_B = 0.45 * (Oren_rough / (Oren_rough + 0.09));
        const float Oren_exposure = 1.0 / Oren_A;
        const unsigned int stride = MYPAINT_TILE_SIZE * MYPAINT_NUM_CHANS;
        for (unsigned int i=0; i<BUFSIZE; i+=MYPAINT_NUM_CHANS) {
            // Calcuate bump map 
            // Use alpha as  height-map
            float slope = 0.0;
            const int reach = 1;
            int o = 0;
            float center = 0.0;
            for (int c=0; c<MYPAINT_NUM_CHANS; c++) {
              center += src[i+c];
            }
            for (int p=1; p<=reach; p++) {
                // North
                o = (i - stride * p) % BUFSIZE;
                float _slope = 0.0;
                for (int c=0; c<MYPAINT_NUM_CHANS; c++) {
                  _slope += src[o+c];
                }
                slope += abs(_slope - center);
                // East
                o = (i + MYPAINT_NUM_CHANS * p) % stride;
                _slope = 0.0;
                for (int c=0; c<MYPAINT_NUM_CHANS; c++) {
                  _slope += src[o+c];
                }
                slope += abs(_slope - center);
                // West
                o = (i - MYPAINT_NUM_CHANS * p) % stride;
                _slope = 0.0;
                for (int c=0; c<MYPAINT_NUM_CHANS; c++) {
                  _slope += src[o+c];
                }
                slope += abs(_slope - center);
                // South
                o = (i + stride * p) % BUFSIZE;
                _slope = 0.0;
                for (int c=0; c<MYPAINT_NUM_CHANS; c++) {
                  _slope += src[o+c];
                }
                slope += abs(_slope - center);
            }

            // amplify slope with options array

            slope = slope / fasterpow(2, opts[1]);

            // reduce slope when dst alpha is very high, like thick paint hiding texture
            slope *= (1.0 - fastpow((float)dst[i+MYPAINT_NUM_CHANS-1] / (1.0), 16));

            float degrees = atan(slope);
            float lambert = (fastcos(degrees) * (Oren_A + (Oren_B * fastsin(degrees) * fasttan(degrees)))) * Oren_exposure;

            for (int c=0; c<MYPAINT_NUM_CHANS-1; c++) {
                dst[i+c] = (float_mul(dst[i+c], lambert));
            }
        }
    }
};



template <bool DSTALPHA, unsigned int BUFSIZE>
class BufferCombineFunc <DSTALPHA, BUFSIZE, BlendNormal, CompositeSpectralWGM>
{
    // Spectral Upsampled Weighted Geometric Mean Pigment/Paint Emulation
    // Based on work by Scott Allen Burns, Meng, and others.
  public:
    inline void operator() (const float * const src,
                            float * dst,
                            const float opac,
                            const float * const opts) const
    {
        for (unsigned int i=0; i<BUFSIZE; i+=MYPAINT_NUM_CHANS) {
            const float Sa = float_mul(src[i+MYPAINT_NUM_CHANS-1], opac);
            const float one_minus_Sa = 1.0 - Sa;
            for (int p=0; p<MYPAINT_NUM_CHANS-1; p++) {
                dst[i+p] = float_sumprods(src[i+p], opac, one_minus_Sa, dst[i+p]);
            }
            if (DSTALPHA) {
                dst[i+MYPAINT_NUM_CHANS-1] = (Sa + float_mul(dst[i+MYPAINT_NUM_CHANS-1], one_minus_Sa));
            }
        }
    }
};

template <bool DSTALPHA, unsigned int BUFSIZE>
class BufferCombineFunc <DSTALPHA, BUFSIZE, BlendNormal, CompositeDestinationIn>
{
    // Partial specialization for svg:dst-in layers,
    // working in premultiplied alpha for speed.
  public:
    inline void operator() (const float * const src,
                            float * dst,
                            const float opac,
                            const float * const opts) const
    {
//        for (unsigned int i=0; i<BUFSIZE; i+=4) {
//            const float Sa = float_mul(src[i+3], opac);
//            dst[i+0] = float_mul(dst[i+0], Sa);
//            dst[i+1] = float_mul(dst[i+1], Sa);
//            dst[i+2] = float_mul(dst[i+2], Sa);
//            if (DSTALPHA) {
//                dst[i+3] = float_mul(Sa, dst[i+3]);
//            }
//        }
    }
};

template <bool DSTALPHA, unsigned int BUFSIZE>
class BufferCombineFunc <DSTALPHA, BUFSIZE, BlendNormal, CompositeDestinationOut>
{
    // Partial specialization for svg:dst-out layers,
    // working in premultiplied alpha for speed.
  public:
    inline void operator() (const float * const src,
                            float * dst,
                            const float opac,
                            const float * const opts) const
    {
//        for (unsigned int i=0; i<BUFSIZE; i+=4) {
//            const float one_minus_Sa = 1.0-float_mul(src[i+3], opac);
//            dst[i+0] = float_mul(dst[i+0], one_minus_Sa);
//            dst[i+1] = float_mul(dst[i+1], one_minus_Sa);
//            dst[i+2] = float_mul(dst[i+2], one_minus_Sa);
//            if (DSTALPHA) {
//                dst[i+3] = float_mul(one_minus_Sa, dst[i+3]);
//            }
//        }
    }
};


template <bool DSTALPHA, unsigned int BUFSIZE>
class BufferCombineFunc <DSTALPHA, BUFSIZE, BlendNormal, CompositeSourceAtop>
{
    // Partial specialization for svg:src-atop layers,
    // working in premultiplied alpha for speed.
  public:
    inline void operator() (const float * const src,
                            float * dst,
                            const float opac,
                            const float * const opts) const
    {
//        for (unsigned int i=0; i<BUFSIZE; i+=4) {
//            const float as = float_mul(src[i+3], opac);
//            const float ab = dst[i+3];
//            const float one_minus_as = 1.0 - as;
//            // W3C spec:
//            //   co = as*Cs*ab + ab*Cb*(1-as)
//            // where
//            //   src[n] = as*Cs    -- premultiplied
//            //   dst[n] = ab*Cb    -- premultiplied
//            dst[i+0] = float_sumprods(float_mul(src[i+0], opac), ab,
//                                      float_mul(dst[i+0], ab), one_minus_as);
//            dst[i+1] = float_sumprods(float_mul(src[i+1], opac), ab,
//                                      float_mul(dst[i+1], ab), one_minus_as);
//            dst[i+2] = float_sumprods(float_mul(src[i+2], opac), ab,
//                                      float_mul(dst[i+2], ab), one_minus_as);
////            printf("%i, %i, %i\n", dst[i+0], dst[i+3], as);
//            if (DSTALPHA) {
//                float alpha = float_sumprods(as, ab, ab, one_minus_as);
//            }
//            // W3C spec:
//            //   ao = as*ab + ab*(1-as)
//            //   ao = ab
//            // (leave output alpha unchanged)
//        }
    }
};

template <bool DSTALPHA, unsigned int BUFSIZE>
class BufferCombineFunc <DSTALPHA, BUFSIZE, BlendNormal, CompositeDestinationAtop>
{
    // Partial specialization for svg:dst-atop layers,
    // working in premultiplied alpha for speed.
  public:
    inline void operator() (const float * const src,
                            float * dst,
                            const float opac,
                            const float * const opts) const
    {
//        for (unsigned int i=0; i<BUFSIZE; i+=4) {
//            const float as = float_mul(src[i+3], opac);
//            const float ab = dst[i+3];
//            const float one_minus_ab = 1.0 - ab;
//            // W3C Spec:
//            //   co = as*Cs*(1-ab) + ab*Cb*as
//            // where
//            //   src[n] = as*Cs    -- premultiplied
//            //   dst[n] = ab*Cb    -- premultiplied
//            dst[i+0] = float_sumprods(float_mul(src[i+0], opac), one_minus_ab,
//                                      float_mul(dst[i+0], dst[i+3]), as);
//            dst[i+1] = float_sumprods(float_mul(src[i+1], opac), one_minus_ab,
//                                      float_mul(dst[i+1], dst[i+3]), as);
//            dst[i+2] = float_sumprods(float_mul(src[i+2], opac), one_minus_ab,
//                                      float_mul(dst[i+2], dst[i+3]), as);
//            // W3C spec:
//            //   ao = as*(1-ab) + ab*as
//            //   ao = as
//            if (DSTALPHA) {
//                dst[i+3] = as;
//            }
//        }
    }
};


// Multiply: http://www.w3.org/TR/compositing/#blendingmultiply

class BlendMultiply : public BlendFunc
{
  public:
    inline void operator() (const float * const src,
                            float * dst,
                            const float * const opts) const
    {
//        dst_r = float_mul(src_r, dst_r);
//        dst_g = float_mul(src_g, dst_g);
//        dst_b = float_mul(src_b, dst_b);
    }
};




// Screen: http://www.w3.org/TR/compositing/#blendingscreen

class BlendScreen : public BlendFunc
{
  public:
    inline void operator() (const float * const src,
                            float * dst,
                            const float * const opts) const
    {
//        dst_r = dst_r + src_r - float_mul(dst_r, src_r);
//        dst_g = dst_g + src_g - float_mul(dst_g, src_g);
//        dst_b = dst_b + src_b - float_mul(dst_b, src_b);
    }
};



// Overlay: http://www.w3.org/TR/compositing/#blendingoverlay

class BlendOverlay : public BlendFunc
{
  private:
    static inline void process_channel(const float Cs, float &Cb)
    {
        const float two_Cb = float(Cb);
        if (two_Cb <= 1.0) {
            Cb = float_mul(Cs, two_Cb);
        }
        else {
            const float tmp = two_Cb - 1.0;
            Cb = Cs + tmp - float_mul(Cs, tmp);
        }
    }

  public:
    inline void operator() (const float * const src,
                            float * dst,
                            const float * const opts) const
    {
//        process_channel(src_r, dst_r);
//        process_channel(src_g, dst_g);
//        process_channel(src_b, dst_b);
    }
};


// Darken: http://www.w3.org/TR/compositing/#blendingdarken

class BlendDarken : public BlendFunc
{
  public:
    inline void operator() (const float * const src,
                            float * dst,
                            const float * const opts) const
    {
//        if (src_r < dst_r) dst_r = src_r;
//        if (src_g < dst_g) dst_g = src_g;
//        if (src_b < dst_b) dst_b = src_b;
    }
};


// Lighten: http://www.w3.org/TR/compositing/#blendinglighten

class BlendLighten : public BlendFunc
{
  public:
    inline void operator() (const float * const src,
                            float * dst,
                            const float * const opts) const
    {
//        if (src_r > dst_r) dst_r = src_r;
//        if (src_g > dst_g) dst_g = src_g;
//        if (src_b > dst_b) dst_b = src_b;
    }
};



// Hard Light: http://www.w3.org/TR/compositing/#blendinghardlight

class BlendHardLight : public BlendFunc
{
  private:
    static inline void process_channel(const float Cs, float &Cb)
    {
        const float two_Cs = float(Cs);
        if (two_Cs <= 1.0) {
            Cb = float_mul(Cb, two_Cs);
        }
        else {
            const float tmp = two_Cs - 1.0;
            Cb = Cb + tmp - float_mul(Cb, tmp);
        }
    }

  public:
    inline void operator() (const float * const src,
                            float * dst,
                            const float * const opts) const
    {
//        process_channel(src_r, dst_r);
//        process_channel(src_g, dst_g);
//        process_channel(src_b, dst_b);
    }
};


// Color-dodge: http://www.w3.org/TR/compositing/#blendingcolordodge

class BlendColorDodge : public BlendFunc
{
  private:
    static inline void process_channel(const float Cs, float &Cb)
    {
        if (Cs < 1.0) {
            const float tmp = float_div(Cb, 1.0 - Cs);
            if (tmp < 1.0) {
                Cb = tmp;
                return;
            }
        }
        Cb = 1.0;
    }

  public:
    inline void operator() (const float * const src,
                            float * dst,
                            const float * const opts) const
    {
//        process_channel(src_r, dst_r);
//        process_channel(src_g, dst_g);
//        process_channel(src_b, dst_b);
    }
};


// Color-burn: http://www.w3.org/TR/compositing/#blendingcolorburn

class BlendColorBurn : public BlendFunc
{
  private:
    static inline void process_channel(const float Cs, float &Cb)
    {
        if (Cs > 0) {
            const float tmp = float_div(1.0 - Cb, Cs);
            if (tmp < 1.0) {
                Cb = 1.0 - tmp;
                return;
            }
        }
        Cb = 0;
    }

  public:
    inline void operator() (const float * const src,
                            float * dst,
                            const float * const opts) const
    {
//        process_channel(src_r, dst_r);
//        process_channel(src_g, dst_g);
//        process_channel(src_b, dst_b);
    }
};


// Soft-light: http://www.w3.org/TR/compositing/#blendingsoftlight

class BlendSoftLight : public BlendFunc
{
  private:
    static inline void process_channel(const float Cs, float &Cb)
    {
        const float two_Cs = float(Cs);
        float B = 0;
        if (two_Cs <= 1.0) {
            B = 1.0 - float_mul(1.0 - two_Cs,
                                      1.0 - Cb);
            B = float_mul(B, Cb);
        }
        else {
            float D = 0;
            const float four_Cb = Cb * 4;
            if (four_Cb <= 1.0) {
                const float Cb_squared = float_mul(Cb, Cb);
                D = four_Cb; /* which is always greater than... */
                D += 16 * float_mul(Cb_squared, Cb);
                D -= 12 * Cb_squared;
                /* ... in the range 0 <= C_b <= 0.25 */
            }
            else {
                D = float_sqrt(Cb);
            }
#ifdef HEAVY_DEBUG
            /* Guard against underflows */
            assert(two_Cs > 1.0);
            assert(D >= Cb);
#endif
            B = Cb + float_mul(2*Cs - 1.0 /* 2*Cs > 1 */,
                               D - Cb           /* D >= Cb */  );
        }
        Cb = B;
    }

  public:
    inline void operator() (const float * const src,
                            float * dst,
                            const float * const opts) const
    {
//        process_channel(src_r, dst_r);
//        process_channel(src_g, dst_g);
//        process_channel(src_b, dst_b);
    }
};


// Difference: http://www.w3.org/TR/compositing/#blendingdifference

class BlendDifference : public BlendFunc
{
  private:
    static inline void process_channel(const float Cs, float &Cb)
    {
        if (Cs >= Cb)
            Cb = Cs - Cb;
        else
            Cb = Cb - Cs;
    }

  public:
    inline void operator() (const float * const src,
                            float * dst,
                            const float * const opts) const
    {
//        process_channel(src_r, dst_r);
//        process_channel(src_g, dst_g);
//        process_channel(src_b, dst_b);
    }
};


// Exclusion: http://www.w3.org/TR/compositing/#blendingexclusion

class BlendExclusion : public BlendFunc
{
  private:
    static inline void process_channel(const float Cs, float &Cb)
    {
        Cb = Cb + Cs - float(float_mul(Cb, Cs));
    }

  public:
    inline void operator() (const float * const src,
                            float * dst,
                            const float * const opts) const
    {
//        process_channel(src_r, dst_r);
//        process_channel(src_g, dst_g);
//        process_channel(src_b, dst_b);
    }
};



//
// Non-separable modes
// http://www.w3.org/TR/compositing/#blendingnonseparable
//

// Auxiliary functions


static const uint16_t BLENDING_LUM_R_COEFF = 0.2126  * 1.0;
static const uint16_t BLENDING_LUM_G_COEFF = 0.7152 * 1.0;
static const uint16_t BLENDING_LUM_B_COEFF = 0.0722 * 1.0;


static inline const float
blending_nonsep_lum (const float r,
                     const float g,
                     const float b)
{
    return (  (r) * BLENDING_LUM_R_COEFF
            + (g) * BLENDING_LUM_G_COEFF
            + (b) * BLENDING_LUM_B_COEFF) / 1.0;
}


static inline void
blending_nonsel_clipcolor (float &r,
                           float &g,
                           float &b)
{
    const float lum = blending_nonsep_lum(r, g, b);
    const float cmin = (r < g) ? MIN(r, b) : MIN(g, b);
    const float cmax = (r > g) ? MAX(r, b) : MAX(g, b);
    if (cmin < 0) {
        const float lum_minus_cmin = lum - cmin;
        r = lum + (((r - lum) * lum) / lum_minus_cmin);
        g = lum + (((g - lum) * lum) / lum_minus_cmin);
        b = lum + (((b - lum) * lum) / lum_minus_cmin);
    }
    if (cmax > (float)1.0) {
        const float one_minus_lum = 1.0 - lum;
        const float cmax_minus_lum = cmax - lum;
        r = lum + (((r - lum) * one_minus_lum) / cmax_minus_lum);
        g = lum + (((g - lum) * one_minus_lum) / cmax_minus_lum);
        b = lum + (((b - lum) * one_minus_lum) / cmax_minus_lum);
    }
}


static inline void
blending_nonsep_setlum (float &r,
                        float &g,
                        float &b,
                        const float lum)
{
    const float diff = lum - blending_nonsep_lum(r, g, b);
    r += diff;
    g += diff;
    b += diff;
    blending_nonsel_clipcolor(r, g, b);
}


static inline const float
blending_nonsep_sat (const float r,
                     const float g,
                     const float b)
{
    const float cmax = (r > g) ? MAX(r, b) : MAX(g, b);
    const float cmin = (r < g) ? MIN(r, b) : MIN(g, b);
    return cmax - cmin;
}


static inline void
blending_nonsep_setsat (float &r,
                        float &g,
                        float &b,
                        const float s)
{
    float *top_c = &b;
    float *mid_c = &g;
    float *bot_c = &r;
    float *tmp = NULL;
    if (*top_c < *mid_c) { tmp = top_c; top_c = mid_c; mid_c = tmp; }
    if (*top_c < *bot_c) { tmp = top_c; top_c = bot_c; bot_c = tmp; }
    if (*mid_c < *bot_c) { tmp = mid_c; mid_c = bot_c; bot_c = tmp; }
#ifdef HEAVY_DEBUG
    assert(top_c != mid_c);
    assert(mid_c != bot_c);
    assert(bot_c != top_c);
    assert(*top_c >= *mid_c);
    assert(*mid_c >= *bot_c);
    assert(*top_c >= *bot_c);
#endif
    if (*top_c > *bot_c) {
        *mid_c = (*mid_c - *bot_c) * s;  // up to fix30
        *mid_c /= *top_c - *bot_c;       // back down to fix15
        *top_c = s;
    }
    else {
        *top_c = *mid_c = 0;
    }
    *bot_c = 0;
}


// Hue: http://www.w3.org/TR/compositing/#blendinghue

class BlendHue : public BlendFunc
{
  public:
    inline void operator() (const float * const src,
                            float * dst,
                            const float * const opts) const
    {
//        const float dst_lum = blending_nonsep_lum(dst_r, dst_g, dst_b);
//        const float dst_sat = blending_nonsep_sat(dst_r, dst_g, dst_b);
//        float r = src_r;
//        float g = src_g;
//        float b = src_b;
//        blending_nonsep_setsat(r, g, b, dst_sat);
//        blending_nonsep_setlum(r, g, b, dst_lum);
//#ifdef HEAVY_DEBUG
//        assert(r <= (float)1.0);
//        assert(g <= (float)1.0);
//        assert(b <= (float)1.0);
//        assert(r >= 0);
//        assert(g >= 0);
//        assert(b >= 0);
//#endif
//        dst_r = r;
//        dst_g = g;
//        dst_b = b;
    }
};


// Saturation: http://www.w3.org/TR/compositing/#blendingsaturation

class BlendSaturation : public BlendFunc
{
  public:
    inline void operator() (const float * const src,
                            float * dst,
                            const float * const opts) const
    {
//        const float dst_lum = blending_nonsep_lum(dst_r, dst_g, dst_b);
//        const float src_sat = blending_nonsep_sat(src_r, src_g, src_b);
//        float r = dst_r;
//        float g = dst_g;
//        float b = dst_b;
//        blending_nonsep_setsat(r, g, b, src_sat);
//        blending_nonsep_setlum(r, g, b, dst_lum);
//#ifdef HEAVY_DEBUG
//        assert(r <= (float)1.0);
//        assert(g <= (float)1.0);
//        assert(b <= (float)1.0);
//        assert(r >= 0);
//        assert(g >= 0);
//        assert(b >= 0);
//#endif
//        dst_r = r;
//        dst_g = g;
//        dst_b = b;
    }
};


// Color: http://www.w3.org/TR/compositing/#blendingcolor

class BlendColor : public BlendFunc
{
  public:
    inline void operator() (const float * const src,
                            float * dst,
                            const float * const opts) const
    {
//        float r = src_r;
//        float g = src_g;
//        float b = src_b;
//        blending_nonsep_setlum(r, g, b,
//          blending_nonsep_lum(dst_r, dst_g, dst_b));
//#ifdef HEAVY_DEBUG
//        assert(r <= (float)1.0);
//        assert(g <= (float)1.0);
//        assert(b <= (float)1.0);
//        assert(r >= 0);
//        assert(g >= 0);
//        assert(b >= 0);
//#endif
//        dst_r = r;
//        dst_g = g;
//        dst_b = b;
    }
};


// Luminosity http://www.w3.org/TR/compositing/#blendingluminosity

class BlendLuminosity : public BlendFunc
{
  public:
    inline void operator() (const float * const src,
                            float * dst,
                            const float * const opts) const
    {
//        float r = dst_r;
//        float g = dst_g;
//        float b = dst_b;
//        blending_nonsep_setlum(r, g, b,
//          blending_nonsep_lum(src_r, src_g, src_b));
//#ifdef HEAVY_DEBUG
//        assert(r <= (float)1.0);
//        assert(g <= (float)1.0);
//        assert(b <= (float)1.0);
//        assert(r >= 0);
//        assert(g >= 0);
//        assert(b >= 0);
//#endif
//        dst_r = r;
//        dst_g = g;
//        dst_b = b;
    }
};



#endif //__HAVE_BLENDING
