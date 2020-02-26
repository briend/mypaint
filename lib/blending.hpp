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
#include <cstdio>


extern float T_MATRIX[3][NUM_WAVES];

extern float spectral_r[NUM_WAVES];

extern float spectral_g[NUM_WAVES];

extern float spectral_b[NUM_WAVES];

/*
  The sum of all channel coefficients - this _should be_ a compile-time
  constant, but that is tricky to implement nicely for floats without C++14
*/
extern float SPECTRAL_WEIGHTS_SUM;

inline void
rgb_to_spectral (float r, float g, float b, float *spectral_) {
  float offset = 1.0 - WGM_EPSILON;
  r = r * offset + WGM_EPSILON;
  g = g * offset + WGM_EPSILON;
  b = b * offset + WGM_EPSILON;
  //upsample rgb to spectral primaries
  float spec_r[NUM_WAVES] = {0};
  for (int i=0; i < NUM_WAVES; i++) {
    spec_r[i] = spectral_r[i] * r;
  }
  float spec_g[NUM_WAVES] = {0};
  for (int i=0; i < NUM_WAVES; i++) {
    spec_g[i] = spectral_g[i] * g;
  }
  float spec_b[NUM_WAVES] = {0};
  for (int i=0; i < NUM_WAVES; i++) {
    spec_b[i] = spectral_b[i] * b;
  }
  //collapse into one spd
  for (int i=0; i<NUM_WAVES; i++) {
    spectral_[i] += log2f(spec_r[i] + spec_g[i] + spec_b[i]);
  }

}

inline void
spectral_to_rgb (float *spectral, float *rgb_) {
  float offset = 1.0 - WGM_EPSILON;
  for (int i=0; i<NUM_WAVES; i++) {
    rgb_[0] += T_MATRIX[0][i] * exp2f(spectral[i]);
    rgb_[1] += T_MATRIX[1][i] * exp2f(spectral[i]);
    rgb_[2] += T_MATRIX[2][i] * exp2f(spectral[i]);
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
        for (int i=0; i<MYPAINT_NUM_CHANS-1; i++) {
            dst[i] = src[i];
        }
    printf("%f", dst[0]);
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
            const float Sa = src[i+MYPAINT_NUM_CHANS-1] * opac;
            if (Sa <= 0.0) continue;
            const float one_minus_Sa = 1.0 - Sa;
            const float alpha = CLAMP(Sa + one_minus_Sa * dst[i+MYPAINT_NUM_CHANS-1], 0.0, 1.0);
            for (int p=0; p<MYPAINT_NUM_CHANS-1; p++) {
                float dstp = dst[i+p];
                float srcp = src[i+p];
                if (dst[i+MYPAINT_NUM_CHANS-1] > 0.0) dstp /= dst[i+MYPAINT_NUM_CHANS-1];
                if (src[i+MYPAINT_NUM_CHANS-1] > 0.0) srcp /= src[i+MYPAINT_NUM_CHANS-1];
                float res = exp2f(srcp) * Sa + exp2f(dstp) * dst[i+MYPAINT_NUM_CHANS-1] * one_minus_Sa;
                if (alpha > 0.0) res /= alpha;
                dst[i+p] = log2f(res) * alpha;
                
            }
            if (DSTALPHA) {
                dst[i+MYPAINT_NUM_CHANS-1] = alpha;
            }
        }
    }
};


template <bool DSTALPHA, unsigned int BUFSIZE>
class BufferCombineFunc <DSTALPHA, BUFSIZE, BlendNormal, CompositeLighter>
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
            const float Sa = src[i+MYPAINT_NUM_CHANS-1] * opac;
            if (Sa <= 0.0) continue;
            const float one_minus_Sa = 1.0 - Sa;
            const float alpha = CLAMP(Sa + dst[i+MYPAINT_NUM_CHANS-1], 0.0, 1.0);
            for (int p=0; p<MYPAINT_NUM_CHANS-1; p++) {
                float dstp = dst[i+p];
                float srcp = src[i+p];
                if (dst[i+MYPAINT_NUM_CHANS-1] > 0.0) dstp /= dst[i+MYPAINT_NUM_CHANS-1];
                if (src[i+MYPAINT_NUM_CHANS-1] > 0.0) srcp /= src[i+MYPAINT_NUM_CHANS-1];
                float res = exp2f(srcp) * Sa + exp2f(dstp) * dst[i+MYPAINT_NUM_CHANS-1];
                if (alpha > 0.0) res /= alpha;
                dst[i+p] = log2f(res) * alpha;
                
            }
            if (DSTALPHA) {
                dst[i+MYPAINT_NUM_CHANS-1] = alpha;
            }
        }
    }
};

template <bool DSTALPHA, unsigned int BUFSIZE>
class BufferCombineFunc <DSTALPHA, BUFSIZE, BlendMultiply, CompositeSourceOver>
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
                dst[i+p] += src[i+p] * opac;
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
    // Apply bump map from SRC to DST.
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
        float Slopes_Array[MYPAINT_TILE_SIZE * MYPAINT_TILE_SIZE] = {0};


        for (unsigned int i=0; i<BUFSIZE; i+=MYPAINT_NUM_CHANS) {
            // Calcuate bump map 
            float slopes[2] = {0.0};
            float dir = 0.0;
            const int reach = 1;
            int o = 0;
            float center = 0.0;
            for (int c=0; c<MYPAINT_NUM_CHANS-1; c++) {
              center += src[i+c];
            }
            for (int p=1; p<=reach; p++) {
              // North
              if (i >= stride * p) {
                  int o = i - stride * p;
                  float _slope = 0.0;
                  float _slope_a = src[o+MYPAINT_NUM_CHANS-1];
                  if (_slope_a > 0.0) {
                    for (int c=0; c<MYPAINT_NUM_CHANS-1; c++) {
                      _slope += src[o+c];
                    }

                      slopes[1] += 2.0 * _slope;

                  }
              } else {
                  int o = i + (BUFSIZE * 4) - (stride * p);
                  float _slope = 0.0;
                  if (src[o+MYPAINT_NUM_CHANS-1] > 0.0) {
                    for (int c=0; c<MYPAINT_NUM_CHANS-1; c++) {
                      _slope += src[o+c];
                    }

                      slopes[1] += 2.0 * _slope;

                  }
              }
              // East
              if (i % stride < stride - MYPAINT_NUM_CHANS * p) {
                  int o = i + MYPAINT_NUM_CHANS * p;
                  float _slope = 0.0;
                  if (src[o+MYPAINT_NUM_CHANS-1] > 0.0) {
                    for (int c=0; c<MYPAINT_NUM_CHANS-1; c++) {
                      _slope += src[o+c];
                    }

                      slopes[0] += 2.0 * _slope;

                  }
              } else {
                  int o = i + (BUFSIZE - (stride - MYPAINT_NUM_CHANS * p));
                  float _slope = 0.0;
                  if (src[o+MYPAINT_NUM_CHANS-1] > 0.0) {
                    for (int c=0; c<MYPAINT_NUM_CHANS-1; c++) {
                      _slope += src[o+c];
                    }

                      slopes[0] += 2.0 * _slope;

                  }
              }
              // West
              if (i % stride >= MYPAINT_NUM_CHANS * p) {
                  int o = i - MYPAINT_NUM_CHANS * p;
                  float _slope = 0.0;
                  if (src[o+MYPAINT_NUM_CHANS-1] > 0.0) {
                    for (int c=0; c<MYPAINT_NUM_CHANS-1; c++) {
                      _slope += src[o+c];
                    }

                      slopes[0] += -2.0 * _slope;

                  }
              } else {
                  int o = i  + (BUFSIZE * 2) + stride - (MYPAINT_NUM_CHANS * p);
                  float _slope = 0.0;
                  if (src[o+MYPAINT_NUM_CHANS-1] > 0.0) {
                    for (int c=0; c<MYPAINT_NUM_CHANS-1; c++) {
                      _slope += src[o+c];
                    }

                      slopes[0] += -2.0 * _slope;

                  }
              }
              // South
              if (i < BUFSIZE - stride * p) {
                  int o = i + stride * p;
                  float _slope = 0.0;
                  if (src[o+MYPAINT_NUM_CHANS-1] > 0.0) {
                    for (int c=0; c<MYPAINT_NUM_CHANS-1; c++) {
                      _slope += src[o+c];
                    }

                      slopes[1] += -2.0 * _slope;

                  }
              } else {
                  int o =  i + (BUFSIZE * 3) + (stride * p);
                  float _slope = 0.0;
                  if (src[o+MYPAINT_NUM_CHANS-1] > 0.0) {
                    for (int c=0; c<MYPAINT_NUM_CHANS-1; c++) {
                      _slope += src[o+c];
                    }

                    slopes[1] += -2.0 * _slope;

                }
              }
            }
            
            // amplify slope with options array
            float slope = abs(slopes[0]) + abs(slopes[1]);
            slope /= fastpow(MYPAINT_NUM_CHANS-1, opts[1]);
            if (slopes[1] != 0.0) {
              dir = atan(abs(slopes[0]) / abs(slopes[1]));
            }
            float degrees = atan(slope * dir);
            float lambert = (fastcos(degrees) * (Oren_A + (Oren_B * fastsin(degrees) * fasttan(degrees)))) * Oren_exposure;
            Slopes_Array[i / MYPAINT_NUM_CHANS] = lambert;
            
        }
            
        for (unsigned int i=0; i<BUFSIZE; i+=MYPAINT_NUM_CHANS) {
          float lambert = Slopes_Array[i / MYPAINT_NUM_CHANS];
          if (lambert != 0.0) {
            for (int c=0; c<MYPAINT_NUM_CHANS-1; c++) {
              dst[i+c] /= lambert;
            }
          }
        }
    }
};


template <bool DSTALPHA, unsigned int BUFSIZE>
class BufferCombineFunc <DSTALPHA, BUFSIZE, BlendNormal, CompositeBumpMapDst>
{
    // apply SRC as bump map to DST.
    // optimize for Background tiles as SRC
    // read pixels from opposite side of tile for edges
    // introduces artifacts if BG texture is not TILE_SIZE
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
        float Slopes_Array[MYPAINT_TILE_SIZE * MYPAINT_TILE_SIZE] = {0};


        for (unsigned int i=0; i<BUFSIZE; i+=MYPAINT_NUM_CHANS) {
            // Calcuate bump map 
            float slopes[2] = {0.0};
            float dir = 0.0;
            const int reach = 1;
            int o = 0;
            float center = 0.0;
            for (int c=0; c<MYPAINT_NUM_CHANS-1; c++) {
              center += src[i+c];
            }
            for (int p=1; p<=reach; p++) {
              // North
              if (i >= stride * p) {
                  int o = i - stride * p;
                  float _slope = 0.0;
                  float _slope_a = src[o+MYPAINT_NUM_CHANS-1];
                  if (_slope_a > 0.0) {
                    for (int c=0; c<MYPAINT_NUM_CHANS-1; c++) {
                      _slope += src[o+c];
                    }

                      slopes[1] += 2.0 * _slope;

                  }
              } else {
                  int o = i + (BUFSIZE * 4) - (stride * p);
                  float _slope = 0.0;
                  if (src[o+MYPAINT_NUM_CHANS-1] > 0.0) {
                    for (int c=0; c<MYPAINT_NUM_CHANS-1; c++) {
                      _slope += src[o+c];
                    }

                      slopes[1] += 2.0 * _slope;

                  }
              }
              // East
              if (i % stride < stride - MYPAINT_NUM_CHANS * p) {
                  int o = i + MYPAINT_NUM_CHANS * p;
                  float _slope = 0.0;
                  if (src[o+MYPAINT_NUM_CHANS-1] > 0.0) {
                    for (int c=0; c<MYPAINT_NUM_CHANS-1; c++) {
                      _slope += src[o+c];
                    }

                      slopes[0] += 2.0 * _slope;

                  }
              } else {
                  int o = i + (BUFSIZE - (stride - MYPAINT_NUM_CHANS * p));
                  float _slope = 0.0;
                  if (src[o+MYPAINT_NUM_CHANS-1] > 0.0) {
                    for (int c=0; c<MYPAINT_NUM_CHANS-1; c++) {
                      _slope += src[o+c];
                    }

                      slopes[0] += 2.0 * _slope;

                  }
              }
              // West
              if (i % stride >= MYPAINT_NUM_CHANS * p) {
                  int o = i - MYPAINT_NUM_CHANS * p;
                  float _slope = 0.0;
                  if (src[o+MYPAINT_NUM_CHANS-1] > 0.0) {
                    for (int c=0; c<MYPAINT_NUM_CHANS-1; c++) {
                      _slope += src[o+c];
                    }

                      slopes[0] += -2.0 * _slope;

                  }
              } else {
                  int o = i  + (BUFSIZE * 2) + stride - (MYPAINT_NUM_CHANS * p);
                  float _slope = 0.0;
                  if (src[o+MYPAINT_NUM_CHANS-1] > 0.0) {
                    for (int c=0; c<MYPAINT_NUM_CHANS-1; c++) {
                      _slope += src[o+c];
                    }

                      slopes[0] += -2.0 * _slope;

                  }
              }
              // South
              if (i < BUFSIZE - stride * p) {
                  int o = i + stride * p;
                  float _slope = 0.0;
                  if (src[o+MYPAINT_NUM_CHANS-1] > 0.0) {
                    for (int c=0; c<MYPAINT_NUM_CHANS-1; c++) {
                      _slope += src[o+c];
                    }

                      slopes[1] += -2.0 * _slope;

                  }
              } else {
                  int o =  i + (BUFSIZE * 3) + (stride * p);
                  float _slope = 0.0;
                  if (src[o+MYPAINT_NUM_CHANS-1] > 0.0) {
                    for (int c=0; c<MYPAINT_NUM_CHANS-1; c++) {
                      _slope += src[o+c];
                    }

                    slopes[1] += -2.0 * _slope;

                }
              }
            }

            // amplify slope with options array
            float slope = abs(slopes[0]) + abs(slopes[1]);
            slope /= fastpow(MYPAINT_NUM_CHANS-1, opts[1]);

            if (slopes[1] != 0.0) {
              dir = atan(abs(slopes[0]) / abs(slopes[1]));
            }
            // reduce slope when dst alpha is very high, like thick paint hiding texture
            slope *= (1.0 - fastpow(dst[i+MYPAINT_NUM_CHANS-1], 16));
        
            float degrees = atan(slope * dir);
            float lambert = (fastcos(degrees) * (Oren_A + (Oren_B * fastsin(degrees) * fasttan(degrees)))) * Oren_exposure;
            Slopes_Array[i / MYPAINT_NUM_CHANS] = lambert;
        }
        for (unsigned int i=0; i<BUFSIZE; i+=MYPAINT_NUM_CHANS) {
          float lambert = Slopes_Array[i / MYPAINT_NUM_CHANS];
          if (lambert != 0.0) {
            for (int c=0; c<MYPAINT_NUM_CHANS-1; c++) {
                    dst[i+c] /= lambert;
            }
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
        for (unsigned int i=0; i<BUFSIZE; i+=MYPAINT_NUM_CHANS) {
            const float Sa = float_mul(src[i+MYPAINT_NUM_CHANS-1], opac);
            const float one_minus_Sa = 1.0 - Sa;
            for (int p=0; p<MYPAINT_NUM_CHANS-1; p++) {
                dst[i+p] *= Sa;
            }
            if (DSTALPHA) {
                dst[i+MYPAINT_NUM_CHANS-1] *= Sa;
            }
        }
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
        for (unsigned int i=0; i<BUFSIZE; i+=MYPAINT_NUM_CHANS) {
            const float Sa = float_mul(src[i+MYPAINT_NUM_CHANS-1], opac);
            const float one_minus_Sa = 1.0 - Sa;
            for (int p=0; p<MYPAINT_NUM_CHANS-1; p++) {
                dst[i+p] *= one_minus_Sa;
            }
            if (DSTALPHA) {
                dst[i+MYPAINT_NUM_CHANS-1] *= one_minus_Sa;
            }
        }
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

            // W3C spec:
            //   co = as*Cs*ab + ab*Cb*(1-as)
            // where
            //   src[n] = as*Cs    -- premultiplied
            //   dst[n] = ab*Cb    -- premultiplied

        for (unsigned int i=0; i<BUFSIZE; i+=MYPAINT_NUM_CHANS) {
            const float Sa = float_mul(src[i+MYPAINT_NUM_CHANS-1], opac);
            const float one_minus_Sa = 1.0 - Sa;
            const float Ba = dst[i+MYPAINT_NUM_CHANS-1];
            for (int p=0; p<MYPAINT_NUM_CHANS-1; p++) {
                dst[i+p] = float_sumprods(src[i+p] * opac, Ba, one_minus_Sa, dst[i+p] * Ba);
            }
            if (DSTALPHA) {
                dst[i+MYPAINT_NUM_CHANS-1] = float_sumprods(Sa, Ba, Ba, one_minus_Sa);
            }
        }
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

            // W3C Spec:
            //   co = as*Cs*(1-ab) + ab*Cb*as
            // where
            //   src[n] = as*Cs    -- premultiplied
            //   dst[n] = ab*Cb    -- premultiplied

        for (unsigned int i=0; i<BUFSIZE; i+=MYPAINT_NUM_CHANS) {
            const float Sa = float_mul(src[i+MYPAINT_NUM_CHANS-1], opac);
            const float Ba = dst[i+MYPAINT_NUM_CHANS-1];
            const float one_minus_Ba = 1.0 - Ba;

            for (int p=0; p<MYPAINT_NUM_CHANS-1; p++) {
                dst[i+p] = float_sumprods(src[i+p] * opac, one_minus_Ba, dst[i+p] * dst[i+MYPAINT_NUM_CHANS-1], Sa);
            }
            if (DSTALPHA) {
                dst[i+MYPAINT_NUM_CHANS-1] = Sa;
            }
        }
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

template <bool DSTALPHA, unsigned int BUFSIZE>
class BufferCombineFunc <DSTALPHA, BUFSIZE, BlendDarken, CompositeSourceOver>
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
            //if (Sa <= 0.0 || dst[i+MYPAINT_NUM_CHANS-1] <= 0.0) continue;
            const float one_minus_Sa = 1.0 - Sa;
            bool was_darker = false;
            for (int p=0; p<MYPAINT_NUM_CHANS-1; p++) {
                float dstp = dst[i+p];
                float srcp = src[i+p];
                if (dst[i+MYPAINT_NUM_CHANS-1] > 0.0) dstp /= dst[i+MYPAINT_NUM_CHANS-1];
                if (src[i+MYPAINT_NUM_CHANS-1] > 0.0) srcp /= src[i+MYPAINT_NUM_CHANS-1];
                if (srcp < dstp) {
                    dst[i+p] = src[i+p] * opac + one_minus_Sa * dst[i+p];
                    was_darker = true;
                }
            }
            if (DSTALPHA && was_darker) {
                dst[i+MYPAINT_NUM_CHANS-1] = (Sa + float_mul(dst[i+MYPAINT_NUM_CHANS-1], one_minus_Sa));
            }
        }
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

template <bool DSTALPHA, unsigned int BUFSIZE>
class BufferCombineFunc <DSTALPHA, BUFSIZE, BlendLighten, CompositeSourceOver>
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
            const float Sa = src[i+MYPAINT_NUM_CHANS-1] * opac;
            if (Sa <= 0.0 || dst[i+MYPAINT_NUM_CHANS-1] <= 0.0) continue;
            const float one_minus_Sa = 1.0 - Sa;
            bool was_lighter = false;
            for (int p=0; p<MYPAINT_NUM_CHANS-1; p++) {
                float dstp = dst[i+p];
                float srcp = src[i+p];
                if (dst[i+MYPAINT_NUM_CHANS-1] > 0.0) dstp /= dst[i+MYPAINT_NUM_CHANS-1];
                if (src[i+MYPAINT_NUM_CHANS-1] > 0.0) srcp /= src[i+MYPAINT_NUM_CHANS-1];
                if (srcp > dstp) {
                    dst[i+p] = src[i+p] * opac + one_minus_Sa * dst[i+p];
                    was_lighter = true;
                }
            }
            if (DSTALPHA && was_lighter) {
                dst[i+MYPAINT_NUM_CHANS-1] = (Sa + dst[i+MYPAINT_NUM_CHANS-1] * one_minus_Sa);
            }
        }
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
