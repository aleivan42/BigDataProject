# === SELEZIONE DEI DOCUMENTI PIÙ SIGNIFICATIVI ===
# Calcola la similarità media per documento
doc_similarity <- rowMeans(sim_matrix)
top_docs <- names(sort(doc_similarity, decreasing = TRUE)[1:20])

# Filtra la matrice per i documenti selezionati
sim_matrix_filtered <- sim_matrix_ordered[rownames(sim_matrix_ordered) %in% top_docs, 
                                          colnames(sim_matrix_ordered) %in% top_docs]

# === HEATMAP ===
sim_df_filtered <- as.data.frame(sim_matrix_filtered) %>%
  rownames_to_column(var = "Doc1") %>%
  pivot_longer(cols = -Doc1, names_to = "Doc2", values_to = "Similarity") %>%
  mutate(
    Doc1 = factor(Doc1, levels = rownames(sim_matrix_filtered)),
    Doc2 = factor(Doc2, levels = colnames(sim_matrix_filtered)),
    Similarity_label = ifelse(Similarity > 0.25 & Doc1 != Doc2, 
                              sprintf("%.2f", Similarity), "")
  )

ggplot(sim_df_filtered, aes(x = Doc1, y = Doc2, fill = Similarity)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = Similarity_label), color = "black", size = 3) +
  scale_fill_gradientn(colors = viridis_pal(option = "D")(10),
                       limits = c(0, 1),
                       name = "Similarità") +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    axis.text.y = element_text(size = 10),
    panel.grid = element_blank(),
    plot.title = element_text(hjust = 0.5, face = "bold")
  ) +
  labs(title = "Similarità tra i 20 documenti più correlati",
       x = "", y = "") +
  coord_fixed()