/*
 * fibembed.h - Standalone text embedding library
 *
 * Extracts the Fibration NN v2 embedding system as a reusable library.
 * Produces 128-dimensional text embeddings using character trigrams,
 * word unigrams, bigrams, and directional pairs through a 4-layer MLP.
 *
 * Usage:
 *   fibembed_init();  // Call once at startup
 *   FibEmbedding e = fibembed_compute("hello world", 11);
 *   float sim = fibembed_similarity(&e1, &e2);
 */

#ifndef FIBEMBED_H
#define FIBEMBED_H

#define FIBEMBED_DIM 128

typedef struct {
    float v[FIBEMBED_DIM];
} FibEmbedding;

/* Initialize the embedding system. Call once at startup. */
void         fibembed_init(void);

/* Compute embedding for UTF-8 text. Returns L2-normalized 128-dim vector. */
FibEmbedding fibembed_compute(const char* text, int len);

/* Cosine similarity between two normalized embeddings (-1 to 1). */
float        fibembed_similarity(const FibEmbedding* a, const FibEmbedding* b);

/* Normalize embedding to unit length. */
void         fibembed_normalize(FibEmbedding* e);

/* Add embedding b into a (for accumulation/averaging). */
void         fibembed_add(FibEmbedding* a, const FibEmbedding* b);

/* Zero out an embedding. */
void         fibembed_zero(FibEmbedding* e);

#endif /* FIBEMBED_H */
