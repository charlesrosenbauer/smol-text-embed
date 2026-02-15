/*
 * example.c - Demonstrate the fibembed standalone embedding library
 *
 * Build: make
 * Run:   ./example
 */

#include <stdio.h>
#include <string.h>
#include "fibembed.h"

int main(void) {
    fibembed_init();

    const char* texts[] = {
        "The quick brown fox jumps over the lazy dog",
        "A fast auburn fox leaps across a sleepy hound",
        "Quantum computing enables exponential speedups",
        "Machine learning models require large datasets",
    };
    int n = sizeof(texts) / sizeof(texts[0]);

    FibEmbedding embeddings[4];
    for (int i = 0; i < n; i++) {
        embeddings[i] = fibembed_compute(texts[i], strlen(texts[i]));
        printf("Embedded: \"%s\"\n", texts[i]);
        printf("  First 8 dims: [");
        for (int d = 0; d < 8; d++) {
            printf("%.4f%s", embeddings[i].v[d], d < 7 ? ", " : "");
        }
        printf("]\n\n");
    }

    printf("Similarity matrix:\n");
    printf("%40s", "");
    for (int j = 0; j < n; j++) {
        printf("  [%d]  ", j);
    }
    printf("\n");

    for (int i = 0; i < n; i++) {
        printf("[%d] %-36.36s", i, texts[i]);
        for (int j = 0; j < n; j++) {
            float sim = fibembed_similarity(&embeddings[i], &embeddings[j]);
            printf(" %5.3f ", sim);
        }
        printf("\n");
    }

    /* Demonstrate accumulation */
    printf("\nAccumulation test:\n");
    FibEmbedding avg;
    fibembed_zero(&avg);
    fibembed_add(&avg, &embeddings[0]);
    fibembed_add(&avg, &embeddings[1]);
    fibembed_normalize(&avg);
    printf("Average of [0]+[1] similarity to [0]: %.3f\n",
           fibembed_similarity(&avg, &embeddings[0]));
    printf("Average of [0]+[1] similarity to [2]: %.3f\n",
           fibembed_similarity(&avg, &embeddings[2]));

    return 0;
}
