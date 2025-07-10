clean_texts <- readRDS("C:/Users/aleca/Documents/BigDataProject/results/processed_data/lemmatized_texts_final.rds")

# Carico le librerie necessarie
library(quanteda)
library(ggplot2)

# Creazione del corpus e DFM
corpus_lemmatized <- corpus(clean_texts, text_field = "lemma_text")
dfm <- tokens(corpus_lemmatized) %>% 
  dfm()

# 1. Legge di Zipf
textstat_frequency(dfm) %>% 
  ggplot(aes(x = rank, y = frequency)) + 
  geom_line() +
  scale_x_log10() +
  scale_y_log10()

# 2. Ricchezza lessicale (Type-Token Ratio)
lexdiv <- textstat_lexdiv(dfm)
print(lexdiv)

# Salva risultati
write.csv(lexdiv, "results/lexical_stats.csv")