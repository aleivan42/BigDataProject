# 1. LIBRERIE ----
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  readtext,
  quanteda,
  quanteda.textstats,
  udpipe,
  tidyverse,
  stringr,
  stopwords
)

# 2. CONFIGURAZIONE ----

text_path  <- here("texts_folder")
output_dir <- here("results")
model_file <- here("english-ewt-ud-2.5-191206.udpipe")


# 3. CREAZIONE STRUTTURA CARTELLE ----
dir.create(file.path(output_dir, "processed_data"), showWarnings = FALSE)
dir.create(file.path(output_dir, "stats"), showWarnings = FALSE)
dir.create(file.path(output_dir, "samples"), showWarnings = FALSE)

# 4. PERCORSI DI OUTPUT ----
output_file_rds <- file.path(output_dir, "processed_data", "lemmatized_texts_final.rds")
output_top_csv <- file.path(output_dir, "stats", "top_20_lemmi.csv")
sample_output <- file.path(output_dir, "samples", "sample_lemmatized.txt")

# 5. STOPWORDS ESTESA ----
stopwords_list <- c(
  stopwords::stopwords("en", source = "stopwords-iso"),
  stopwords::stopwords("en", source = "smart"),
  stopwords::stopwords("en", source = "snowball"),
  c("thou", "thee", "thy", "hath", "shall", "unto", "would",
    "could", "might", "must", "may", "upon", "also", "even",
    "every", "said", "one", "like", "much", "many", "go",
    "come", "went", "came", "us", "oh", "ah", "ha", "ho")
) %>% unique() %>% tolower()

# 6. FUNZIONI ----
clean_text <- function(text) {
  text %>%
    str_replace_all("[^[:alnum:][:space:]]", " ") %>%
    str_remove_all("\\b\\d+\\b") %>%
    str_remove_all("\\b[a-z]\\b") %>%
    str_squish() %>%
    tolower()
}

lemmatize_final <- function(text, model, stopwords) {
  clean_txt <- clean_text(text)
  
  tokens <- tokens(clean_txt,
                   remove_punct = TRUE,
                   remove_numbers = TRUE,
                   remove_symbols = TRUE) %>%
    tokens_remove(stopwords, valuetype = "fixed") %>%
    tokens_tolower()
  
  tryCatch({
    anno <- udpipe_annotate(model, as.character(tokens))
    lemmas <- as.data.frame(anno) %>%
      filter(
        !upos %in% c("PUNCT", "SYM", "NUM", "DET", "AUX", "PART"),
        nchar(lemma) > 2,
        !str_detect(lemma, "'|\\d")
      ) %>%
      pull(lemma)
    
    lemmas <- lemmas[!lemmas %in% stopwords]
    paste(lemmas, collapse = " ")
  }, error = function(e) {
    message("Errore: ", e$message)
    return(NA_character_)
  })
}

# 7. ESECUZIONE ----
cat("=== INIZIO PROCESSING ===\n")

# Caricamento modello
if (!file.exists(model_file)) {
  udpipe_download_model(language = "english")
}
ud_model <- udpipe_load_model(model_file)

# Caricamento testi
raw_texts <- readtext(paste0(text_path, "/*.txt"), encoding = "UTF-8")

# Lemmatizzazione
clean_texts <- raw_texts %>%
  mutate(
    lemma_text = map_chr(
      text,
      ~ lemmatize_final(.x, ud_model, stopwords_list),
      .progress = TRUE
    )
  )

# 8. VERIFICA QUALITÀ ----
cat("\n=== VERIFICA STOPWORDS ===\n")
final_check <- clean_texts %>%
  corpus(text_field = "lemma_text") %>%
  tokens() %>%
  tokens_keep(pattern = stopwords_list, valuetype = "fixed") %>%
  dfm() %>%
  colSums()

if (sum(final_check) > 0) {
  warning_msg <- data.frame(
    word = names(final_check[final_check > 0]),
    count = unname(final_check[final_check > 0])
  )
  write.csv(warning_msg, 
            file.path(output_dir, "stats", "residual_stopwords.csv"),
            row.names = FALSE)
  cat("Stopwords residue salvate in: stats/residual_stopwords.csv\n")
} else {
  cat("Z ero stopwords residue rilevate\n")
}

# 9. ANALISI E SALVATAGGIO ----
# Top 20 lemmi
top_clean <- clean_texts %>%
  corpus(text_field = "lemma_text") %>%
  tokens() %>%
  dfm() %>%
  quanteda.textstats::textstat_frequency(n = 20)

write.csv(top_clean, output_top_csv, row.names = FALSE)
cat("\n Top 20 lemmi salvati in: stats/top_20_lemmi.csv\n")

# Salvataggio dati completi
saveRDS(clean_texts, output_file_rds)
cat("\n Dati completi salvati in: processed_data/lemmatized_texts_final.rds\n")

# Esempio di output testuale
sample_content <- substr(clean_texts$lemma_text[1], 1, 1000)
writeLines(sample_content, sample_output)
cat("\n Esempio di testo salvato in: samples/sample_lemmatized.txt\n")

# 10. RIEPILOGO ----
cat("\n=== RIEPILOGO OUTPUT ===\n")
cat("- processed_data/lemmatized_texts_final.rds : Dati lemmatizzati completi\n")
cat("- stats/top_20_lemmi.csv : Frequenze dei lemmi principali\n")
cat("- stats/residual_stopwords.csv : Eventuali stopwords residue\n")
cat("- samples/sample_lemmatized.txt : Esempio di testo processato\n")