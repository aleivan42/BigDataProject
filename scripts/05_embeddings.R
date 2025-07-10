# ==== PARAMETRI ====
library(here)

input_rds   <- here("results", "processed_data", "lemmatized_texts_final.rds")
output_dir  <- here("results", "embeddings")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)


# Parametri modificabili
top_n_words <- 100      # Numero parole da visualizzare
embedding_dim <- 50     # Dimensionalità embeddings
perplexity <- 30        # Parametro t-SNE (adattivo se necessario)
min_word_freq <- 5      # Frequenza minima parole
max_doc_prop <- 0.8     # Massima proporzione documenti in cui appare una parola

set.seed(123) # Riproducibilità

# ==== 1. CARICAMENTO E PREPROCESSING ====
cat("=== CARICAMENTO DATI ===\n")
clean_texts <- readRDS(input_rds)
corpus_lemmatized <- corpus(clean_texts, text_field = "lemma_text")

# Tokenizzazione
tokens_list <- tokens(corpus_lemmatized) %>% 
  as.list() %>% 
  lapply(function(x) x[x != ""]) # Rimuovi stringhe vuote

# ==== 2. COSTRUZIONE VOCABOLO ====
cat("=== COSTRUZIONE VOCABOLO ===\n")
it <- itoken(tokens_list, progressbar = TRUE)
vocab <- create_vocabulary(it) %>% 
  prune_vocabulary(
    term_count_min = min_word_freq,
    doc_proportion_max = max_doc_prop
  )

# ==== 3. COSTRUZIONE EMBEDDINGS (GloVe) ====
cat("=== ADDESTRAMENTO EMBEDDINGS ===\n")
vectorizer <- vocab_vectorizer(vocab)
tcm <- create_tcm(it, vectorizer, skip_grams_window = 5L)

glove <- GlobalVectors$new(
  rank = embedding_dim,
  x_max = 10,
  learning_rate = 0.05
)

word_vectors <- glove$fit_transform(tcm, n_iter = 50, convergence_tol = 0.001) + 
  t(glove$components)

# ==== 4. SELEZIONE PAROLE TOP ====
cat("=== SELEZIONE PAROLE ===\n")
freq_df <- data.frame(
  word = vocab$term,
  freq = vocab$term_count,
  stringsAsFactors = FALSE
) %>% 
  arrange(desc(freq))

top_words <- freq_df %>% 
  filter(freq > quantile(freq, 0.9)) %>% 
  pull(word) %>% 
  head(top_n_words)

filtered_vectors <- word_vectors[rownames(word_vectors) %in% top_words, ]

# ==== 5. RIDUZIONE DIMENSIONALITÀ (PCA) ====
cat("=== ANALISI PCA ===\n")
pca <- prcomp(filtered_vectors, center = TRUE, scale. = TRUE)
pca_df <- as.data.frame(pca$x[, 1:2])
pca_df$word <- rownames(filtered_vectors)

# Visualizzazione PCA migliorata
p1 <- ggplot(pca_df, aes(x = PC1, y = PC2, label = word)) +
  geom_point(color = "#2E86C1", alpha = 0.8, size = 2.5) +
  geom_text_repel(
    size = 3.5,
    max.overlaps = 25,
    segment.color = "grey30",
    segment.size = 0.3,
    box.padding = 0.5,
    bg.color = "white",
    bg.r = 0.15,
    color = "black"
  ) +
  labs(title = "Word Embeddings - PCA",
       subtitle = paste("Top", length(top_words), "parole per frequenza")) +
  theme_minimal(base_size = 12) +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = "grey80"),
    panel.grid.major = element_line(color = "grey90", linewidth = 0.2),
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5)
  )

ggsave(file.path(output_dir, "pca_plot.png"), p1, width = 14, height = 12, dpi = 350, bg = "white")

# ==== 6. t-SNE ====
cat("=== ANALISI t-SNE ===\n")
perplexity_adj <- min(perplexity, floor(nrow(filtered_vectors)/3))

tsne_out <- Rtsne(
  filtered_vectors,
  dims = 2,
  perplexity = perplexity_adj,
  theta = 0.1,
  max_iter = 1000,
  verbose = TRUE
)

tsne_df <- data.frame(
  X = tsne_out$Y[,1],
  Y = tsne_out$Y[,2],
  word = rownames(filtered_vectors)
)

# Visualizzazione t-SNE migliorata
p2 <- ggplot(tsne_df, aes(x = X, y = Y, label = word)) +
  geom_point(color = "#E74C3C", alpha = 0.8, size = 2.5) +
  geom_text_repel(
    size = 3.5,
    max.overlaps = 20,
    segment.color = "grey30",
    segment.size = 0.3,
    box.padding = 0.5,
    bg.color = "white",
    bg.r = 0.15,
    color = "black"
  ) +
  labs(title = "Word Embeddings - t-SNE",
       subtitle = paste("Perplexity =", perplexity_adj)) +
  theme_minimal(base_size = 12) +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = "grey80"),
    panel.grid.major = element_line(color = "grey90", linewidth = 0.2),
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5)
  )

ggsave(file.path(output_dir, "tsne_plot.png"), p2, width = 14, height = 12, dpi = 350, bg = "white")

# ==== 7. CLUSTERING ====
cat("=== CLUSTERING ===\n")
# Determinazione ottimo numero cluster
wss <- sapply(1:10, function(k){kmeans(filtered_vectors, k, nstart = 25)$tot.withinss})
elbow_plot <- fviz_nbclust(filtered_vectors, kmeans, method = "wss", k.max = 10) +
  geom_vline(xintercept = which.min(diff(wss)) + 1, linetype = 2, color = "red") +
  labs(title = "Elbow Method for Optimal k") +
  theme_minimal() +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = "grey80"),
    panel.grid.major = element_line(color = "grey90"),
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5)
  )

optimal_k <- max(3, which.min(diff(wss)) + 1) # Almeno 3 cluster

# K-means clustering
set.seed(123)
km_res <- kmeans(filtered_vectors, centers = optimal_k, nstart = 25)

# Palette ad alto contrasto per cluster
cluster_palette <- c("#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", 
                     "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF")

# Aggiungi cluster ai dati
pca_df$cluster <- as.factor(km_res$cluster)
tsne_df$cluster <- as.factor(km_res$cluster)

# Visualizzazione cluster PCA migliorata
p3 <- ggplot(pca_df, aes(x = PC1, y = PC2, color = cluster, label = word)) +
  geom_point(alpha = 0.8, size = 2.5) +
  geom_text_repel(
    size = 3.5,
    max.overlaps = 15,
    show.legend = FALSE,
    segment.color = "grey30",
    segment.size = 0.3,
    box.padding = 0.5,
    bg.color = "white",
    bg.r = 0.15,
    color = "black"
  ) +
  scale_color_manual(values = cluster_palette) +
  labs(title = "Word Clusters (PCA Space)",
       subtitle = paste("k =", optimal_k, "clusters")) +
  theme_minimal(base_size = 12) +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = "grey80"),
    panel.grid.major = element_line(color = "grey90", linewidth = 0.2),
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5),
    legend.position = "bottom"
  )

# Visualizzazione cluster t-SNE migliorata
p4 <- ggplot(tsne_df, aes(x = X, y = Y, color = cluster, label = word)) +
  geom_point(alpha = 0.8, size = 2.5) +
  geom_text_repel(
    size = 3.5,
    max.overlaps = 15,
    show.legend = FALSE,
    segment.color = "grey30",
    segment.size = 0.3,
    box.padding = 0.5,
    bg.color = "white",
    bg.r = 0.15,
    color = "black"
  ) +
  scale_color_manual(values = cluster_palette) +
  labs(title = "Word Clusters (t-SNE Space)",
       subtitle = paste("k =", optimal_k, "clusters")) +
  theme_minimal(base_size = 12) +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = "grey80"),
    panel.grid.major = element_line(color = "grey90", linewidth = 0.2),
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5),
    legend.position = "bottom"
  )

# Salva plots con qualità migliore
ggsave(file.path(output_dir, "cluster_pca.png"), p3, width = 14, height = 12, dpi = 350, bg = "white")
ggsave(file.path(output_dir, "cluster_tsne.png"), p4, width = 14, height = 12, dpi = 350, bg = "white")
ggsave(file.path(output_dir, "elbow_plot.png"), elbow_plot, width = 10, height = 8, dpi = 350, bg = "white")

# ==== 8. ANALISI SEMANTICA ====
cat("=== ANALISI SEMANTICA ===\n")
find_similar_words <- function(word, vectors, n = 5) {
  if (!word %in% rownames(vectors)) {
    cat("Parola non trovata nel vocabolario.\n")
    return(NULL)
  }
  
  sims <- sim2(vectors, vectors[word, , drop = FALSE], method = "cosine")
  sims_df <- data.frame(
    word = rownames(sims),
    similarity = as.numeric(sims),
    stringsAsFactors = FALSE
  ) %>% 
    arrange(desc(similarity)) %>% 
    filter(word != !!word) %>% 
    head(n)
  
  return(sims_df)
}

# Esempi di analisi
example_words <- c("love", "time", "man", "house", "good")
similarity_results <- lapply(example_words, function(w) {
  res <- find_similar_words(w, word_vectors)
  if (!is.null(res)) {
    write.csv(res, file.path(output_dir, paste0("similar_to_", w, ".csv")), row.names = FALSE)
  }
  return(res)
})

# ==== 9. SALVATAGGIO RISULTATI ====
cat("=== SALVATAGGIO RISULTATI ===\n")
write.csv(pca_df, file.path(output_dir, "pca_coordinates.csv"), row.names = FALSE)
write.csv(tsne_df, file.path(output_dir, "tsne_coordinates.csv"), row.names = FALSE)
saveRDS(word_vectors, file.path(output_dir, "word_vectors.rds"))

# Report finale
sink(file.path(output_dir, "analysis_report.txt"))
cat("=== REPORT ANALISI EMBEDDINGS ===\n\n")
cat("Parametri usati:\n")
cat("- Dimensionalità embeddings:", embedding_dim, "\n")
cat("- Top parole analizzate:", length(top_words), "\n")
cat("- Perplexity t-SNE:", perplexity_adj, "\n")
cat("- Numero cluster ottimale:", optimal_k, "\n\n")

cat("Statistiche base:\n")
cat("- Parole totali nel vocabolario:", nrow(vocab), "\n")
cat("- Parole filtrate per analisi:", nrow(filtered_vectors), "\n")
cat("- Varianza spiegata PCA (2 componenti):", 
    round(sum(summary(pca)$importance[2,1:2])*100, 1), "%\n")

sink()

# ==== 10. VISUALIZZAZIONE FINALE ====
cat("=== ANALISI COMPLETATA ===\n")
cat("File salvati in:", output_dir, "\n")
cat("Visualizza i file PNG per i risultati grafici.\n")

# Apri automaticamente una visualizzazione
if (interactive()) {
  browseURL(file.path(output_dir, "cluster_tsne.png"))
}