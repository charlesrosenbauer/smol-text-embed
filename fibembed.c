/*
 * fibembed.c - Standalone implementation of Fibration NN v2 text embedding
 *
 * Self-contained single compilation unit. Includes:
 *   - Platform detection (TLS, SIMD)
 *   - UTF-8 decoding
 *   - Feature extraction (trigrams, words, bigrams, directional pairs)
 *   - Block-sparse and dense matrix operations
 *   - 4-layer MLP forward pass
 *   - Public API wrappers
 */

#include "fibembed.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* ================================================================
 * Platform detection
 * ================================================================ */

#ifdef _MSC_VER
    #define FIB_TLS __declspec(thread)
#else
    #define FIB_TLS __thread
#endif

/* SIMD support */
#if defined(__AVX2__)
    #include <immintrin.h>
    #define FIB_USE_AVX2 1
#elif defined(__SSE2__) || defined(_M_X64)
    #include <emmintrin.h>
    #define FIB_USE_SSE2 1
#endif

/* ================================================================
 * Model dimensions (must match weights header)
 * ================================================================ */

#ifndef EMB_V2_N_TRIGRAM_BUCKETS
#define EMB_V2_N_TRIGRAM_BUCKETS 12288
#endif
#ifndef EMB_V2_N_WORD_BUCKETS
#define EMB_V2_N_WORD_BUCKETS 4096
#endif
#ifndef EMB_V2_N_BIGRAM_BUCKETS
#define EMB_V2_N_BIGRAM_BUCKETS 4096
#endif
#ifndef EMB_V2_N_DIRPAIR_BUCKETS
#define EMB_V2_N_DIRPAIR_BUCKETS 4096
#endif
#ifndef EMB_V2_HIDDEN_DIM_1
#define EMB_V2_HIDDEN_DIM_1 384
#endif
#ifndef EMB_V2_HIDDEN_DIM_2
#define EMB_V2_HIDDEN_DIM_2 256
#endif
#ifndef EMB_V2_HIDDEN_DIM_3
#define EMB_V2_HIDDEN_DIM_3 192
#endif
#ifndef EMB_V2_EMBED_DIM
#define EMB_V2_EMBED_DIM 128
#endif
#ifndef EMB_V2_BLOCK_SIZE
#define EMB_V2_BLOCK_SIZE 16
#endif
#define EMB_V2_DIR_PAIR_WINDOW 3

/* ================================================================
 * Weights
 * ================================================================ */

#if __has_include("fibembed_weights.h")
    #include "fibembed_weights.h"
    #define FIB_WEIGHTS_AVAILABLE 1
#else
    #define FIB_WEIGHTS_AVAILABLE 0
    #define EMB_V2_TRI_N_BLOCKS 0
    #define EMB_V2_USE_LAYER_NORM 0
#endif

/* Compressed blob path */
#if defined(EMB_V2_COMPRESSED) && EMB_V2_COMPRESSED

    static const uint16_t* emb_v2_tri_row_perm = NULL;
    static const int8_t*   emb_v2_tri_blocks   = NULL;
    static const uint16_t* emb_v2_tri_indices  = NULL;
    static const float*    emb_v2_tri_scales   = NULL;
    static const int8_t*   emb_v2_w1_word      = NULL;
    static const int8_t*   emb_v2_w1_bigram    = NULL;
    static const int8_t*   emb_v2_w1_dirpair   = NULL;
    static const float*    emb_v2_b1           = NULL;
    static const float*    emb_v2_b2           = NULL;
    static const float*    emb_v2_b3           = NULL;
    static const float*    emb_v2_b4           = NULL;
    static const int8_t*   emb_v2_w2           = NULL;
    static const int8_t*   emb_v2_w3           = NULL;
    static const int8_t*   emb_v2_w4           = NULL;
    #if EMB_V2_USE_LAYER_NORM
    static const float*    emb_v2_ln1_gamma    = NULL;
    static const float*    emb_v2_ln1_beta     = NULL;
    static const float*    emb_v2_ln2_gamma    = NULL;
    static const float*    emb_v2_ln2_beta     = NULL;
    static const float*    emb_v2_ln3_gamma    = NULL;
    static const float*    emb_v2_ln3_beta     = NULL;
    #endif

    static char* fibEmbBlob = NULL;

#elif !FIB_WEIGHTS_AVAILABLE
    static const float emb_v2_w1_word_scale = 1.0f;
    static const float emb_v2_w1_bigram_scale = 1.0f;
    static const float emb_v2_w1_dirpair_scale = 1.0f;
    static const float emb_v2_w2_scale = 1.0f;
    static const float emb_v2_w3_scale = 1.0f;
    static const float emb_v2_w4_scale = 1.0f;

    static const uint16_t  _fib_stub_u16 = 0;
    static const int8_t    _fib_stub_i8  = 0;
    static const float     _fib_stub_f   = 0;
    static const uint16_t* emb_v2_tri_row_perm = &_fib_stub_u16;
    static const int8_t*   emb_v2_tri_blocks   = &_fib_stub_i8;
    static const uint16_t* emb_v2_tri_indices  = &_fib_stub_u16;
    static const float*    emb_v2_tri_scales   = &_fib_stub_f;
    static const int8_t*   emb_v2_w1_word      = &_fib_stub_i8;
    static const int8_t*   emb_v2_w1_bigram    = &_fib_stub_i8;
    static const int8_t*   emb_v2_w1_dirpair   = &_fib_stub_i8;
    static const float*    emb_v2_b1           = &_fib_stub_f;
    static const float*    emb_v2_b2           = &_fib_stub_f;
    static const float*    emb_v2_b3           = &_fib_stub_f;
    static const float*    emb_v2_b4           = &_fib_stub_f;
    static const int8_t*   emb_v2_w2           = &_fib_stub_i8;
    static const int8_t*   emb_v2_w3           = &_fib_stub_i8;
    static const int8_t*   emb_v2_w4           = &_fib_stub_i8;
#endif

/* ================================================================
 * Hash functions (FNV-1a)
 * ================================================================ */

#define FIB_FNV_OFFSET 0xcbf29ce484222325ULL
#define FIB_FNV_PRIME  0x100000001b3ULL

static inline uint64_t fib_fnv1aChar(uint64_t h, uint32_t c) {
    h ^= c;
    h *= FIB_FNV_PRIME;
    return h;
}

static inline uint64_t fib_hashTrigram(uint32_t c0, uint32_t c1, uint32_t c2) {
    uint64_t h = FIB_FNV_OFFSET;
    h = fib_fnv1aChar(h, c0);
    h = fib_fnv1aChar(h, c1);
    h = fib_fnv1aChar(h, c2);
    return h;
}

static inline uint64_t fib_hashWord(const uint32_t* text, int start, int len) {
    uint64_t h = FIB_FNV_OFFSET;
    h = fib_fnv1aChar(h, 'w');
    h = fib_fnv1aChar(h, ':');
    for (int i = 0; i < len; i++) {
        h = fib_fnv1aChar(h, text[start + i]);
    }
    return h;
}

static inline uint64_t fib_hashBigram(uint64_t h1, uint64_t h2) {
    uint64_t h = FIB_FNV_OFFSET;
    h = fib_fnv1aChar(h, 'b');
    h = fib_fnv1aChar(h, ':');
    h ^= h1;
    h *= FIB_FNV_PRIME;
    h = fib_fnv1aChar(h, '|');
    h ^= h2;
    h *= FIB_FNV_PRIME;
    return h;
}

static inline uint64_t fib_hashDirpair(uint64_t h1, uint64_t h2) {
    uint64_t h = FIB_FNV_OFFSET;
    h = fib_fnv1aChar(h, 'd');
    h = fib_fnv1aChar(h, ':');
    h ^= h1;
    h *= FIB_FNV_PRIME;
    h = fib_fnv1aChar(h, '>');
    h ^= h2;
    h *= FIB_FNV_PRIME;
    return h;
}

/* ================================================================
 * Character classification
 * ================================================================ */

static inline uint32_t fib_toLower(uint32_t c) {
    if (c >= 'A' && c <= 'Z') return c + 32;
    return c;
}

static inline int fib_isWhitespace(uint32_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static inline int fib_isWordChar(uint32_t c) {
    if (c >= 'a' && c <= 'z') return 1;
    if (c >= 'A' && c <= 'Z') return 1;
    if (c >= '0' && c <= '9') return 1;
    if (c >= 0x80) return 1;
    if (c == '\'' || c == '-') return 1;
    return 0;
}

/* ================================================================
 * Feature structures
 * ================================================================ */

#define FIB_MAX_TRIGRAMS 4096
#define FIB_MAX_WORDS 512
#define FIB_MAX_BIGRAMS 512
#define FIB_MAX_DIRPAIRS 1024
#define FIB_HASH_TABLE_SIZE 8192

typedef struct {
    uint16_t index;
    uint16_t count;
} FibFeature;

typedef struct {
    uint32_t hash;
    uint16_t start;
    uint16_t len;
} FibWordInfo;

typedef struct {
    FibFeature* trigrams;
    int nTrigrams;
    float trigramTotal;
    FibFeature* words;
    int nWords;
    float wordTotal;
    FibFeature* bigrams;
    int nBigrams;
    float bigramTotal;
    FibFeature* dirpairs;
    int nDirpairs;
    float dirpairTotal;
} FibExtractedFeatures;

/* ================================================================
 * Block index for trigram weights
 * ================================================================ */

#define FIB_MAX_TRI_INPUT_BLOCKS ((EMB_V2_N_TRIGRAM_BUCKETS + EMB_V2_BLOCK_SIZE - 1) / EMB_V2_BLOCK_SIZE)

static int fibInitialized = 0;
static uint16_t* fibTriBlockIndexLists = NULL;
static int* fibTriBlockIndexStarts = NULL;
static int* fibTriBlockIndexCounts = NULL;

static void fib_buildTriBlockIndex(void) {
#if FIB_WEIGHTS_AVAILABLE
    if (fibTriBlockIndexLists) return;

    fibTriBlockIndexCounts = calloc(FIB_MAX_TRI_INPUT_BLOCKS, sizeof(int));
    if (!fibTriBlockIndexCounts) return;

    for (int b = 0; b < EMB_V2_TRI_N_BLOCKS; b++) {
        int inputBlock = emb_v2_tri_indices[b * 2];
        if (inputBlock < FIB_MAX_TRI_INPUT_BLOCKS) {
            fibTriBlockIndexCounts[inputBlock]++;
        }
    }

    fibTriBlockIndexStarts = malloc(sizeof(int) * (FIB_MAX_TRI_INPUT_BLOCKS + 1));
    if (!fibTriBlockIndexStarts) {
        free(fibTriBlockIndexCounts); fibTriBlockIndexCounts = NULL;
        return;
    }

    fibTriBlockIndexStarts[0] = 0;
    for (int i = 0; i < FIB_MAX_TRI_INPUT_BLOCKS; i++) {
        fibTriBlockIndexStarts[i + 1] = fibTriBlockIndexStarts[i] + fibTriBlockIndexCounts[i];
    }

    int totalEntries = fibTriBlockIndexStarts[FIB_MAX_TRI_INPUT_BLOCKS];
    fibTriBlockIndexLists = malloc(sizeof(uint16_t) * totalEntries);
    if (!fibTriBlockIndexLists) {
        free(fibTriBlockIndexCounts); fibTriBlockIndexCounts = NULL;
        free(fibTriBlockIndexStarts); fibTriBlockIndexStarts = NULL;
        return;
    }

    int* fillCounts = calloc(FIB_MAX_TRI_INPUT_BLOCKS, sizeof(int));
    if (!fillCounts) {
        free(fibTriBlockIndexCounts); fibTriBlockIndexCounts = NULL;
        free(fibTriBlockIndexStarts); fibTriBlockIndexStarts = NULL;
        free(fibTriBlockIndexLists); fibTriBlockIndexLists = NULL;
        return;
    }

    for (int b = 0; b < EMB_V2_TRI_N_BLOCKS; b++) {
        int inputBlock = emb_v2_tri_indices[b * 2];
        if (inputBlock < FIB_MAX_TRI_INPUT_BLOCKS) {
            int offset = fibTriBlockIndexStarts[inputBlock] + fillCounts[inputBlock];
            fibTriBlockIndexLists[offset] = (uint16_t)b;
            fillCounts[inputBlock]++;
        }
    }
    free(fillCounts);
#endif
}

/* ================================================================
 * Feature extraction
 * ================================================================ */

static int fib_extractFeatures(const uint32_t* text, int len, FibExtractedFeatures* out) {
    if (len < 3 || !FIB_WEIGHTS_AVAILABLE) return 0;

    static FIB_TLS uint16_t triHashTable[FIB_HASH_TABLE_SIZE];
    static FIB_TLS uint16_t triHashCounts[FIB_HASH_TABLE_SIZE];
    static FIB_TLS uint8_t triHashUsed[FIB_HASH_TABLE_SIZE];

    static FIB_TLS uint16_t wordHashTable[FIB_HASH_TABLE_SIZE];
    static FIB_TLS uint16_t wordHashCounts[FIB_HASH_TABLE_SIZE];
    static FIB_TLS uint8_t wordHashUsed[FIB_HASH_TABLE_SIZE];

    static FIB_TLS uint16_t bigramHashTable[FIB_HASH_TABLE_SIZE];
    static FIB_TLS uint16_t bigramHashCounts[FIB_HASH_TABLE_SIZE];
    static FIB_TLS uint8_t bigramHashUsed[FIB_HASH_TABLE_SIZE];

    static FIB_TLS uint16_t dirpairHashTable[FIB_HASH_TABLE_SIZE];
    static FIB_TLS uint16_t dirpairHashCounts[FIB_HASH_TABLE_SIZE];
    static FIB_TLS uint8_t dirpairHashUsed[FIB_HASH_TABLE_SIZE];

    memset(triHashUsed, 0, sizeof(triHashUsed));
    memset(wordHashUsed, 0, sizeof(wordHashUsed));
    memset(bigramHashUsed, 0, sizeof(bigramHashUsed));
    memset(dirpairHashUsed, 0, sizeof(dirpairHashUsed));

    static FIB_TLS uint32_t normalized[8192];
    static FIB_TLS FibWordInfo wordInfos[FIB_MAX_WORDS];
    int normLen = 0;
    int nWordInfos = 0;

    int inWord = 0;
    int wordStart = 0;

    for (int i = 0; i < len && normLen < 8191; i++) {
        uint32_t c = text[i];
        c = fib_toLower(c);

        if (fib_isWhitespace(c)) {
            if (normLen > 0 && normalized[normLen-1] != ' ') {
                if (inWord && nWordInfos < FIB_MAX_WORDS) {
                    int wordLen = normLen - wordStart;
                    if (wordLen > 0) {
                        wordInfos[nWordInfos].start = wordStart;
                        wordInfos[nWordInfos].len = wordLen;
                        wordInfos[nWordInfos].hash = fib_hashWord(normalized, wordStart, wordLen);
                        nWordInfos++;
                    }
                }
                normalized[normLen++] = ' ';
                inWord = 0;
            }
        } else {
            if (!inWord) {
                wordStart = normLen;
                inWord = 1;
            }
            normalized[normLen++] = c;

            if (!fib_isWordChar(c) && inWord) {
                int wordLen = normLen - 1 - wordStart;
                if (wordLen > 0 && nWordInfos < FIB_MAX_WORDS) {
                    wordInfos[nWordInfos].start = wordStart;
                    wordInfos[nWordInfos].len = wordLen;
                    wordInfos[nWordInfos].hash = fib_hashWord(normalized, wordStart, wordLen);
                    nWordInfos++;
                }
                wordStart = normLen;
            }
        }
    }

    if (inWord && nWordInfos < FIB_MAX_WORDS) {
        int wordLen = normLen - wordStart;
        if (wordLen > 0) {
            wordInfos[nWordInfos].start = wordStart;
            wordInfos[nWordInfos].len = wordLen;
            wordInfos[nWordInfos].hash = fib_hashWord(normalized, wordStart, wordLen);
            nWordInfos++;
        }
    }

    /* Extract trigrams */
    int nTriUnique = 0;
    float triTotal = 0;
    uint32_t prev2 = ' ', prev1 = ' ';

    for (int i = 0; i < normLen; i++) {
        uint32_t c = normalized[i];
        if (prev2 != 0 && prev1 != 0) {
            uint64_t h = fib_hashTrigram(prev2, prev1, c);
            uint16_t bucket = h % EMB_V2_N_TRIGRAM_BUCKETS;
            uint16_t permuted = emb_v2_tri_row_perm[bucket];

            uint32_t slot = permuted % FIB_HASH_TABLE_SIZE;
            while (triHashUsed[slot] && triHashTable[slot] != permuted) {
                slot = (slot + 1) % FIB_HASH_TABLE_SIZE;
            }

            if (!triHashUsed[slot]) {
                triHashUsed[slot] = 1;
                triHashTable[slot] = permuted;
                triHashCounts[slot] = 1;
                nTriUnique++;
            } else {
                triHashCounts[slot]++;
            }
            triTotal++;
        }
        prev2 = prev1;
        prev1 = c;
    }

    /* Extract word unigrams */
    int nWordUnique = 0;
    float wordTotal = 0;
    for (int i = 0; i < nWordInfos; i++) {
        uint16_t bucket = wordInfos[i].hash % EMB_V2_N_WORD_BUCKETS;
        uint32_t slot = bucket % FIB_HASH_TABLE_SIZE;
        while (wordHashUsed[slot] && wordHashTable[slot] != bucket) {
            slot = (slot + 1) % FIB_HASH_TABLE_SIZE;
        }
        if (!wordHashUsed[slot]) {
            wordHashUsed[slot] = 1;
            wordHashTable[slot] = bucket;
            wordHashCounts[slot] = 1;
            nWordUnique++;
        } else {
            wordHashCounts[slot]++;
        }
        wordTotal++;
    }

    /* Extract word bigrams */
    int nBigramUnique = 0;
    float bigramTotal = 0;
    for (int i = 0; i < nWordInfos - 1; i++) {
        uint64_t h = fib_hashBigram(wordInfos[i].hash, wordInfos[i+1].hash);
        uint16_t bucket = h % EMB_V2_N_BIGRAM_BUCKETS;
        uint32_t slot = bucket % FIB_HASH_TABLE_SIZE;
        while (bigramHashUsed[slot] && bigramHashTable[slot] != bucket) {
            slot = (slot + 1) % FIB_HASH_TABLE_SIZE;
        }
        if (!bigramHashUsed[slot]) {
            bigramHashUsed[slot] = 1;
            bigramHashTable[slot] = bucket;
            bigramHashCounts[slot] = 1;
            nBigramUnique++;
        } else {
            bigramHashCounts[slot]++;
        }
        bigramTotal++;
    }

    /* Extract directional word pairs */
    int nDirpairUnique = 0;
    float dirpairTotal = 0;
    for (int i = 0; i < nWordInfos; i++) {
        for (int j = i + 1; j < nWordInfos && j <= i + EMB_V2_DIR_PAIR_WINDOW; j++) {
            uint64_t h = fib_hashDirpair(wordInfos[i].hash, wordInfos[j].hash);
            uint16_t bucket = h % EMB_V2_N_DIRPAIR_BUCKETS;
            uint32_t slot = bucket % FIB_HASH_TABLE_SIZE;
            while (dirpairHashUsed[slot] && dirpairHashTable[slot] != bucket) {
                slot = (slot + 1) % FIB_HASH_TABLE_SIZE;
            }
            if (!dirpairHashUsed[slot]) {
                dirpairHashUsed[slot] = 1;
                dirpairHashTable[slot] = bucket;
                dirpairHashCounts[slot] = 1;
                nDirpairUnique++;
            } else {
                dirpairHashCounts[slot]++;
            }
            dirpairTotal++;
        }
    }

    /* Collect features from hash tables */
    int nTri = 0;
    for (int i = 0; i < FIB_HASH_TABLE_SIZE && nTri < FIB_MAX_TRIGRAMS; i++) {
        if (triHashUsed[i]) {
            out->trigrams[nTri].index = triHashTable[i];
            out->trigrams[nTri].count = triHashCounts[i];
            nTri++;
        }
    }
    out->nTrigrams = nTri;
    out->trigramTotal = triTotal;

    int nWord = 0;
    for (int i = 0; i < FIB_HASH_TABLE_SIZE && nWord < FIB_MAX_WORDS; i++) {
        if (wordHashUsed[i]) {
            out->words[nWord].index = wordHashTable[i];
            out->words[nWord].count = wordHashCounts[i];
            nWord++;
        }
    }
    out->nWords = nWord;
    out->wordTotal = wordTotal;

    int nBigram = 0;
    for (int i = 0; i < FIB_HASH_TABLE_SIZE && nBigram < FIB_MAX_BIGRAMS; i++) {
        if (bigramHashUsed[i]) {
            out->bigrams[nBigram].index = bigramHashTable[i];
            out->bigrams[nBigram].count = bigramHashCounts[i];
            nBigram++;
        }
    }
    out->nBigrams = nBigram;
    out->bigramTotal = bigramTotal;

    int nDirpair = 0;
    for (int i = 0; i < FIB_HASH_TABLE_SIZE && nDirpair < FIB_MAX_DIRPAIRS; i++) {
        if (dirpairHashUsed[i]) {
            out->dirpairs[nDirpair].index = dirpairHashTable[i];
            out->dirpairs[nDirpair].count = dirpairHashCounts[i];
            nDirpair++;
        }
    }
    out->nDirpairs = nDirpair;
    out->dirpairTotal = dirpairTotal;

    (void)nTriUnique; (void)nWordUnique; (void)nBigramUnique; (void)nDirpairUnique;
    return 1;
}

/* ================================================================
 * SIMD helpers
 * ================================================================ */

#if defined(FIB_USE_SSE2)
static inline float fib_horizontal_sum_ps(__m128 v) {
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float result;
    _mm_store_ss(&result, sums);
    return result;
}

static inline float fib_dotProductInt8Float_SSE2(const float* hidden, const int8_t* weights, int len) {
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();
    __m128 acc2 = _mm_setzero_ps();
    __m128 acc3 = _mm_setzero_ps();

    for (int j = 0; j < len; j += 16) {
        __m128i w8 = _mm_loadu_si128((const __m128i*)&weights[j]);
        __m128i zero = _mm_setzero_si128();
        __m128i sign = _mm_cmpgt_epi8(zero, w8);
        __m128i w16_lo = _mm_unpacklo_epi8(w8, sign);
        __m128i w16_hi = _mm_unpackhi_epi8(w8, sign);
        __m128i sign_lo = _mm_srai_epi16(w16_lo, 15);
        __m128i sign_hi = _mm_srai_epi16(w16_hi, 15);
        __m128i w32_0 = _mm_unpacklo_epi16(w16_lo, sign_lo);
        __m128i w32_1 = _mm_unpackhi_epi16(w16_lo, sign_lo);
        __m128i w32_2 = _mm_unpacklo_epi16(w16_hi, sign_hi);
        __m128i w32_3 = _mm_unpackhi_epi16(w16_hi, sign_hi);
        __m128 wf0 = _mm_cvtepi32_ps(w32_0);
        __m128 wf1 = _mm_cvtepi32_ps(w32_1);
        __m128 wf2 = _mm_cvtepi32_ps(w32_2);
        __m128 wf3 = _mm_cvtepi32_ps(w32_3);
        __m128 h0 = _mm_loadu_ps(&hidden[j + 0]);
        __m128 h1 = _mm_loadu_ps(&hidden[j + 4]);
        __m128 h2 = _mm_loadu_ps(&hidden[j + 8]);
        __m128 h3 = _mm_loadu_ps(&hidden[j + 12]);
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(h0, wf0));
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(h1, wf1));
        acc2 = _mm_add_ps(acc2, _mm_mul_ps(h2, wf2));
        acc3 = _mm_add_ps(acc3, _mm_mul_ps(h3, wf3));
    }
    __m128 acc = _mm_add_ps(_mm_add_ps(acc0, acc1), _mm_add_ps(acc2, acc3));
    return fib_horizontal_sum_ps(acc);
}
#endif

#if defined(FIB_USE_AVX2)
static inline float fib_horizontal_sum_ps_avx(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float result;
    _mm_store_ss(&result, sums);
    return result;
}

static inline float fib_dotProductInt8Float_AVX2(const float* hidden, const int8_t* weights, int len) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    for (int j = 0; j < len; j += 16) {
        __m128i w8 = _mm_loadu_si128((const __m128i*)&weights[j]);
        __m256i w32_lo = _mm256_cvtepi8_epi32(w8);
        __m256i w32_hi = _mm256_cvtepi8_epi32(_mm_srli_si128(w8, 8));
        __m256 wf_lo = _mm256_cvtepi32_ps(w32_lo);
        __m256 wf_hi = _mm256_cvtepi32_ps(w32_hi);
        __m256 h_lo = _mm256_loadu_ps(&hidden[j]);
        __m256 h_hi = _mm256_loadu_ps(&hidden[j + 8]);
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(h_lo, wf_lo));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(h_hi, wf_hi));
    }
    return fib_horizontal_sum_ps_avx(_mm256_add_ps(acc0, acc1));
}
#endif

/* ================================================================
 * Matrix operations
 * ================================================================ */

static inline void fib_blockMatmulAccum(const float* blockInput, const int8_t* block,
                                        float scale, float* hidden, int hiddenStart, int hiddenDim) {
    for (int h = 0; h < EMB_V2_BLOCK_SIZE; h++) {
        int hiddenIdx = hiddenStart + h;
        if (hiddenIdx >= hiddenDim) break;
        float sum = 0.0f;
        for (int j = 0; j < EMB_V2_BLOCK_SIZE; j++) {
            sum += blockInput[j] * (float)block[j * EMB_V2_BLOCK_SIZE + h];
        }
        hidden[hiddenIdx] += sum * scale;
    }
}

static void fib_sparseMatmulAccum(const FibFeature* features, int nFeatures, float totalCount,
                                  const int8_t* W, float scale, int hiddenDim, int inputDim,
                                  float* hidden) {
    for (int f = 0; f < nFeatures; f++) {
        int idx = features[f].index;
        float val = features[f].count / (totalCount + 1e-8f);
        for (int h = 0; h < hiddenDim; h++) {
            hidden[h] += val * (float)W[h * inputDim + idx] * scale;
        }
    }
}

static void fib_layerNorm(float* x, int dim, const float* gamma, const float* beta) {
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) mean += x[i];
    mean /= dim;
    float var = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= dim;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < dim; i++) {
        x[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

static void fib_relu(float* x, int dim) {
#if defined(FIB_USE_AVX2)
    __m256 zero = _mm256_setzero_ps();
    for (int i = 0; i < dim; i += 8) {
        __m256 v = _mm256_loadu_ps(&x[i]);
        v = _mm256_max_ps(v, zero);
        _mm256_storeu_ps(&x[i], v);
    }
#elif defined(FIB_USE_SSE2)
    __m128 zero = _mm_setzero_ps();
    for (int i = 0; i < dim; i += 4) {
        __m128 v = _mm_loadu_ps(&x[i]);
        v = _mm_max_ps(v, zero);
        _mm_storeu_ps(&x[i], v);
    }
#else
    for (int i = 0; i < dim; i++) {
        if (x[i] < 0) x[i] = 0;
    }
#endif
}

static void fib_denseMatmul(const float* hidden, int hiddenDim,
                            const int8_t* W, float scale, const float* bias,
                            float* out, int outDim) {
#if defined(FIB_USE_AVX2) && FIB_WEIGHTS_AVAILABLE
    for (int i = 0; i < outDim; i++) {
        const int8_t* row = &W[i * hiddenDim];
        float sum = fib_dotProductInt8Float_AVX2(hidden, row, hiddenDim);
        out[i] = sum * scale + bias[i];
    }
#elif defined(FIB_USE_SSE2) && FIB_WEIGHTS_AVAILABLE
    for (int i = 0; i < outDim; i++) {
        const int8_t* row = &W[i * hiddenDim];
        float sum = fib_dotProductInt8Float_SSE2(hidden, row, hiddenDim);
        out[i] = sum * scale + bias[i];
    }
#else
    for (int i = 0; i < outDim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < hiddenDim; j++) {
            sum += hidden[j] * (float)W[i * hiddenDim + j];
        }
        out[i] = sum * scale + bias[i];
    }
#endif
}

/* ================================================================
 * Forward pass
 * ================================================================ */

static void fib_forwardPass(const FibExtractedFeatures* feat, float* embedding) {
    float h1[EMB_V2_HIDDEN_DIM_1];
    memcpy(h1, emb_v2_b1, sizeof(float) * EMB_V2_HIDDEN_DIM_1);

#if FIB_WEIGHTS_AVAILABLE
    /* Trigram features (block-sparse) */
    if (fibTriBlockIndexLists && fibTriBlockIndexStarts && fibTriBlockIndexCounts && feat->nTrigrams > 0) {
        uint8_t activeBlocks[FIB_MAX_TRI_INPUT_BLOCKS];
        memset(activeBlocks, 0, sizeof(activeBlocks));
        for (int f = 0; f < feat->nTrigrams; f++) {
            int block = feat->trigrams[f].index / EMB_V2_BLOCK_SIZE;
            if (block < FIB_MAX_TRI_INPUT_BLOCKS) activeBlocks[block] = 1;
        }

        float blockInput[EMB_V2_BLOCK_SIZE];
        for (int ib = 0; ib < FIB_MAX_TRI_INPUT_BLOCKS; ib++) {
            if (!activeBlocks[ib] || fibTriBlockIndexCounts[ib] == 0) continue;

            memset(blockInput, 0, sizeof(blockInput));
            int inputStart = ib * EMB_V2_BLOCK_SIZE;
            for (int f = 0; f < feat->nTrigrams; f++) {
                int idx = feat->trigrams[f].index;
                if (idx >= inputStart && idx < inputStart + EMB_V2_BLOCK_SIZE) {
                    blockInput[idx - inputStart] = feat->trigrams[f].count / (feat->trigramTotal + 1e-8f);
                }
            }

            int start = fibTriBlockIndexStarts[ib];
            int count = fibTriBlockIndexCounts[ib];
            for (int i = 0; i < count; i++) {
                int b = fibTriBlockIndexLists[start + i];
                int colBlock = emb_v2_tri_indices[b * 2 + 1];
                float blockScale = emb_v2_tri_scales[b];
                const int8_t* blockData = &emb_v2_tri_blocks[b * EMB_V2_BLOCK_SIZE * EMB_V2_BLOCK_SIZE];
                int hiddenStart = colBlock * EMB_V2_BLOCK_SIZE;
                fib_blockMatmulAccum(blockInput, blockData, blockScale, h1, hiddenStart, EMB_V2_HIDDEN_DIM_1);
            }
        }
    }

    if (feat->nWords > 0) {
        fib_sparseMatmulAccum(feat->words, feat->nWords, feat->wordTotal,
                              emb_v2_w1_word, emb_v2_w1_word_scale,
                              EMB_V2_HIDDEN_DIM_1, EMB_V2_N_WORD_BUCKETS, h1);
    }
    if (feat->nBigrams > 0) {
        fib_sparseMatmulAccum(feat->bigrams, feat->nBigrams, feat->bigramTotal,
                              emb_v2_w1_bigram, emb_v2_w1_bigram_scale,
                              EMB_V2_HIDDEN_DIM_1, EMB_V2_N_BIGRAM_BUCKETS, h1);
    }
    if (feat->nDirpairs > 0) {
        fib_sparseMatmulAccum(feat->dirpairs, feat->nDirpairs, feat->dirpairTotal,
                              emb_v2_w1_dirpair, emb_v2_w1_dirpair_scale,
                              EMB_V2_HIDDEN_DIM_1, EMB_V2_N_DIRPAIR_BUCKETS, h1);
    }
#endif

#if defined(EMB_V2_USE_LAYER_NORM) && EMB_V2_USE_LAYER_NORM
    fib_layerNorm(h1, EMB_V2_HIDDEN_DIM_1, emb_v2_ln1_gamma, emb_v2_ln1_beta);
#endif
    fib_relu(h1, EMB_V2_HIDDEN_DIM_1);

    float h2[EMB_V2_HIDDEN_DIM_2];
    fib_denseMatmul(h1, EMB_V2_HIDDEN_DIM_1, emb_v2_w2, emb_v2_w2_scale, emb_v2_b2, h2, EMB_V2_HIDDEN_DIM_2);
#if defined(EMB_V2_USE_LAYER_NORM) && EMB_V2_USE_LAYER_NORM
    fib_layerNorm(h2, EMB_V2_HIDDEN_DIM_2, emb_v2_ln2_gamma, emb_v2_ln2_beta);
#endif
    fib_relu(h2, EMB_V2_HIDDEN_DIM_2);

    float h3[EMB_V2_HIDDEN_DIM_3];
    fib_denseMatmul(h2, EMB_V2_HIDDEN_DIM_2, emb_v2_w3, emb_v2_w3_scale, emb_v2_b3, h3, EMB_V2_HIDDEN_DIM_3);
#if defined(EMB_V2_USE_LAYER_NORM) && EMB_V2_USE_LAYER_NORM
    fib_layerNorm(h3, EMB_V2_HIDDEN_DIM_3, emb_v2_ln3_gamma, emb_v2_ln3_beta);
#endif
    fib_relu(h3, EMB_V2_HIDDEN_DIM_3);

    fib_denseMatmul(h3, EMB_V2_HIDDEN_DIM_3, emb_v2_w4, emb_v2_w4_scale, emb_v2_b4, embedding, EMB_V2_EMBED_DIM);

    /* L2 normalize */
    float mag = 0.0f;
    for (int i = 0; i < EMB_V2_EMBED_DIM; i++) {
        mag += embedding[i] * embedding[i];
    }
    if (mag > 1e-8f) {
        mag = 1.0f / sqrtf(mag);
        for (int i = 0; i < EMB_V2_EMBED_DIM; i++) {
            embedding[i] *= mag;
        }
    }
}

/* ================================================================
 * Compressed weight decompression
 * ================================================================ */

#if defined(EMB_V2_COMPRESSED) && EMB_V2_COMPRESSED

static void fib_bitmapExpand(const unsigned char* packed, int packedLen,
                             unsigned char* out, int outLen) {
    int ip = 0, op = 0;
    while (ip < packedLen && op < outLen) {
        unsigned char bm = packed[ip++];
        for (int j = 0; j < 8 && op < outLen; j++) {
            if (bm & (1 << j))
                out[op++] = packed[ip++];
            else
                out[op++] = 0;
        }
    }
}

static void fib_decompressWeights(void) {
    if (fibEmbBlob) return;

    fibEmbBlob = malloc(EMB_V2_BLOB_SIZE);
    if (!fibEmbBlob) {
        fprintf(stderr, "fibembed: failed to allocate %d bytes\n", EMB_V2_BLOB_SIZE);
        return;
    }

    memcpy(fibEmbBlob, emb_v2_data, EMB_V2_DATA_SIZE);

    fib_bitmapExpand(emb_v2_tri_packed, EMB_V2_TRI_PACKED_SIZE,
                     (unsigned char*)fibEmbBlob + emb_v2_tri_blocks_OFFSET,
                     emb_v2_tri_blocks_COUNT);

    #define FIB_BLOB_PTR(type, name) \
        name = (const type*)(fibEmbBlob + name##_OFFSET)

    FIB_BLOB_PTR(float,    emb_v2_b1);
    FIB_BLOB_PTR(float,    emb_v2_b2);
    FIB_BLOB_PTR(float,    emb_v2_b3);
    FIB_BLOB_PTR(float,    emb_v2_b4);
#if EMB_V2_USE_LAYER_NORM
    FIB_BLOB_PTR(float,    emb_v2_ln1_gamma);
    FIB_BLOB_PTR(float,    emb_v2_ln1_beta);
    FIB_BLOB_PTR(float,    emb_v2_ln2_gamma);
    FIB_BLOB_PTR(float,    emb_v2_ln2_beta);
    FIB_BLOB_PTR(float,    emb_v2_ln3_gamma);
    FIB_BLOB_PTR(float,    emb_v2_ln3_beta);
#endif
    FIB_BLOB_PTR(float,    emb_v2_tri_scales);
    FIB_BLOB_PTR(uint16_t, emb_v2_tri_row_perm);
    FIB_BLOB_PTR(uint16_t, emb_v2_tri_indices);
    FIB_BLOB_PTR(int8_t,   emb_v2_tri_blocks);
    FIB_BLOB_PTR(int8_t,   emb_v2_w1_word);
    FIB_BLOB_PTR(int8_t,   emb_v2_w1_bigram);
    FIB_BLOB_PTR(int8_t,   emb_v2_w1_dirpair);
    FIB_BLOB_PTR(int8_t,   emb_v2_w2);
    FIB_BLOB_PTR(int8_t,   emb_v2_w3);
    FIB_BLOB_PTR(int8_t,   emb_v2_w4);

    #undef FIB_BLOB_PTR
}
#endif

/* ================================================================
 * Public API
 * ================================================================ */

void fibembed_init(void) {
    if (fibInitialized) return;
#if defined(EMB_V2_COMPRESSED) && EMB_V2_COMPRESSED
    fib_decompressWeights();
#endif
    fib_buildTriBlockIndex();
    fibInitialized = 1;
}

FibEmbedding fibembed_compute(const char* text, int len) {
    FibEmbedding result;
    memset(&result, 0, sizeof(result));

    if (!text || len <= 0 || !FIB_WEIGHTS_AVAILABLE) return result;

    /* Decode UTF-8 to UTF-32 */
    uint32_t utf32[8192];
    int utf32Len = 0;
    int i = 0;
    while (i < len && utf32Len < 8191) {
        uint32_t c;
        unsigned char b = (unsigned char)text[i];
        if (b < 0x80) {
            c = b;
            i++;
        } else if ((b & 0xE0) == 0xC0 && i + 1 < len) {
            c = ((b & 0x1F) << 6) | (text[i+1] & 0x3F);
            i += 2;
        } else if ((b & 0xF0) == 0xE0 && i + 2 < len) {
            c = ((b & 0x0F) << 12) | ((text[i+1] & 0x3F) << 6) | (text[i+2] & 0x3F);
            i += 3;
        } else if ((b & 0xF8) == 0xF0 && i + 3 < len) {
            c = ((b & 0x07) << 18) | ((text[i+1] & 0x3F) << 12) |
                ((text[i+2] & 0x3F) << 6) | (text[i+3] & 0x3F);
            i += 4;
        } else {
            c = b;
            i++;
        }
        utf32[utf32Len++] = c;
    }

    FibFeature trigrams[FIB_MAX_TRIGRAMS];
    FibFeature words[FIB_MAX_WORDS];
    FibFeature bigrams[FIB_MAX_BIGRAMS];
    FibFeature dirpairs[FIB_MAX_DIRPAIRS];

    FibExtractedFeatures feat = {
        .trigrams = trigrams, .nTrigrams = 0, .trigramTotal = 0,
        .words = words, .nWords = 0, .wordTotal = 0,
        .bigrams = bigrams, .nBigrams = 0, .bigramTotal = 0,
        .dirpairs = dirpairs, .nDirpairs = 0, .dirpairTotal = 0,
    };

    if (!fib_extractFeatures(utf32, utf32Len, &feat)) return result;

    fib_forwardPass(&feat, result.v);
    return result;
}

float fibembed_similarity(const FibEmbedding* a, const FibEmbedding* b) {
    float dot = 0.0f;
#if defined(FIB_USE_SSE2)
    __m128 vdot = _mm_setzero_ps();
    for (int i = 0; i + 4 <= FIBEMBED_DIM; i += 4) {
        __m128 va = _mm_loadu_ps(&a->v[i]);
        __m128 vb = _mm_loadu_ps(&b->v[i]);
        vdot = _mm_add_ps(vdot, _mm_mul_ps(va, vb));
    }
    __m128 shuf = _mm_shuffle_ps(vdot, vdot, _MM_SHUFFLE(2, 3, 0, 1));
    vdot = _mm_add_ps(vdot, shuf);
    shuf = _mm_movehl_ps(shuf, vdot);
    vdot = _mm_add_ss(vdot, shuf);
    _mm_store_ss(&dot, vdot);
#else
    for (int i = 0; i < FIBEMBED_DIM; i++) {
        dot += a->v[i] * b->v[i];
    }
#endif
    return dot;
}

void fibembed_normalize(FibEmbedding* e) {
    float mag = 0.0f;
    for (int i = 0; i < FIBEMBED_DIM; i++) {
        mag += e->v[i] * e->v[i];
    }
    if (mag > 1e-8f) {
        mag = 1.0f / sqrtf(mag);
        for (int i = 0; i < FIBEMBED_DIM; i++) {
            e->v[i] *= mag;
        }
    }
}

void fibembed_add(FibEmbedding* a, const FibEmbedding* b) {
    for (int i = 0; i < FIBEMBED_DIM; i++) {
        a->v[i] += b->v[i];
    }
}

void fibembed_zero(FibEmbedding* e) {
    memset(e->v, 0, sizeof(e->v));
}
