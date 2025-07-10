# === 0. LIBRERIE NECESSARIE ===
library(quanteda)
library(topicmodels)
library(tidytext)
library(tidyverse)
library(ggplot2)
library(scales)
library(forcats)

# === 1. CARICAMENTO TESTI LEMMATIZZATI ===
clean_texts <- readRDS("C:/Users/aleca/Documents/BigDataProject/results/processed_data/lemmatized_texts_final.rds")

dfm <- corpus(clean_texts, text_field = "lemma_text") %>%
  tokens() %>%
  dfm()

# === 2. CONVERSIONE A DOCUMENT-TERM MATRIX PER LDA ===
dtm <- convert(dfm, to = "topicmodels")

# === 3. STIMA MODELLO LDA ===
k_topics <- 10  # Puoi cambiare il numero di topic
lda_model <- LDA(dtm, k = k_topics, control = list(seed = 1234))

# === 4. TOP 10 LEMMI PER TOPIC ===
top_terms <- tidy(lda_model, matrix = "beta") %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  ungroup() %>%
  arrange(topic, -beta)

# === 5. TOPIC DOMINANTE PER DOCUMENTO ===
topic_distribution <- tidy(lda_model, matrix = "gamma")

dominant_topic <- topic_distribution %>%
  group_by(document) %>%
  slice_max(gamma, n = 1) %>%
  ungroup()

# === 6. GRAFICO: TOP PAROLE PER TOPIC ===
top_terms %>%
  mutate(term = fct_reorder(term, beta)) %>%
  ggplot(aes(x = beta, y = term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free_y") +
  scale_x_continuous(labels = percent_format(accuracy = 1)) +
  labs(title = "Top 10 Lemmi per Topic",
       x = "Probabilità (beta)",
       y = NULL) +
  theme_minimal(base_size = 13)

# === 7. SALVATAGGI ===
saveRDS(lda_model, "results/topic_model.rds")
write_csv(top_terms, "results/top_terms_by_topic.csv")
write_csv(dominant_topic, "results/dominant_topic_by_document.csv")

# === 8. OUTPUT DI CONTROLLO ===
cat("Modello salvato. Ecco un esempio di top lemmi per ciascun topic:\n")
print(top_terms %>% group_by(topic) %>% summarise(lemmi = paste(term, collapse = ", "), .groups = "drop"))
